"""Run the frozen Phase 5 selective-review protocol with vLLM on a V100."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402

sys.path.insert(0, str(ROOT))
from phase4.lib.transition import parse_review, transition_name  # noqa: E402


PROTOCOL_PATH = ROOT / "phase5/configs/review_protocol_v1.yaml"
LOCK_PATH = ROOT / "phase5/data/protocol/review_protocol_v1_lock.json"
REGISTRY_PATH = ROOT / "phase5/configs/experiments.yaml"
SPLITS = {
    "smoke": ROOT / "phase5/data/protocol/smoke_v1.jsonl",
    "train": ROOT / "phase5/data/splits/v1/train.jsonl",
    "development": ROOT / "phase5/data/splits/v1/development.jsonl",
    "protected_test": ROOT / "phase5/data/splits/v1/protected_test.jsonl",
}


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify_lock(lock: dict[str, Any]) -> None:
    for record in lock["files"].values():
        path = ROOT / record["path"]
        observed = sha256_lf(path)
        if observed != record["sha256_lf"]:
            raise RuntimeError(f"Frozen input changed: {record['path']} ({observed})")
    if not lock.get("all_passed"):
        raise RuntimeError("Frozen protocol lock is not passing")


def resolve_adapter(repo: str, revision: str, expected_sha: str, token: str | None) -> str:
    path = Path(snapshot_download(repo_id=repo, revision=revision, token=token))
    observed = sha256_file(path / "adapter_model.safetensors")
    if observed != expected_sha:
        raise RuntimeError(f"Adapter hash mismatch for {repo}: {observed}")
    return str(path)


def build_problem(row: dict[str, Any]) -> Problem:
    return Problem(
        id=row["problem_id"], domain="math", question=row["question"],
        reference_answer=row["reference_answer"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        choices=("original_solver", "warmstart_v2", "correction_sft_v3"))
    parser.add_argument("--split", default="smoke", choices=tuple(SPLITS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-v2", type=Path,
                        default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--open-protected", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    if args.split == "protected_test" and not args.open_protected:
        raise RuntimeError("Protected test is sealed; pass --open-protected explicitly once")

    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    verify_lock(lock)
    if protocol["generation"]["backend"] != "vllm_0_7_0":
        raise RuntimeError("Protocol backend is not frozen to vllm_0_7_0")

    rows = read_jsonl(SPLITS[args.split])
    if args.split == "smoke":
        counts = Counter(bool(row["initial_correct"]) for row in rows)
        if len(rows) != 8 or counts != Counter({True: 4, False: 4}):
            raise RuntimeError(f"Invalid smoke composition: rows={len(rows)}, counts={counts}")

    repos = registry["checkpoint_hub_repos"]
    revisions = registry["checkpoint_hub_revisions"]
    hashes = registry["checkpoint_adapter_sha256"]
    adapter_path: str | None = None
    model_revision: str | None = None
    if args.checkpoint == "original_solver":
        model_path = repos["original_solver"]
        model_revision = revisions["original_solver"]
    elif args.checkpoint == "warmstart_v2":
        model_path = repos["original_solver"]
        model_revision = revisions["original_solver"]
        adapter_path = resolve_adapter(
            repos["phase4_warmstart_v2"], revisions["phase4_warmstart_v2"],
            hashes["phase4_warmstart_v2"], token,
        )
    else:
        marker = args.merged_v2 / "phase5_lineage.json"
        if not marker.exists():
            raise RuntimeError(f"Missing verified merged V2 parent: {marker}")
        lineage = json.loads(marker.read_text(encoding="utf-8"))
        expected = {
            "base_repo": repos["original_solver"],
            "base_revision": revisions["original_solver"],
            "adapter_repo": repos["phase4_warmstart_v2"],
            "adapter_revision": revisions["phase4_warmstart_v2"],
            "adapter_weight_sha256": hashes["phase4_warmstart_v2"],
            "weight_dtype": "float16",
        }
        if lineage != expected:
            raise RuntimeError(f"Merged V2 lineage mismatch: {lineage}")
        model_path = str(args.merged_v2)
        adapter_path = resolve_adapter(
            repos["phase4_correction_sft_v3"], revisions["phase4_correction_sft_v3"],
            hashes["phase4_correction_sft_v3"], token,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=model_revision, token=token,
    )
    conversations: list[dict[str, Any]] = []
    condition_messages = protocol["conversation"]["condition_messages"]
    contract = protocol["conversation"]["contract_instruction"]
    system = protocol["conversation"]["system_prompt"]
    for row in rows:
        problem = build_problem(row)
        for condition in protocol["scope"]["primary_conditions"]:
            if condition == "neutral":
                condition_text = condition_messages["neutral"]["all"]
            else:
                key = "initially_correct" if row["initial_correct"] else "initially_wrong"
                condition_text = condition_messages[condition][key]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": build_prompt(problem)},
                {"role": "assistant", "content": row["initial_output"]},
                {"role": "user", "content": f"{condition_text}\n\n{contract}"},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            conversations.append({"source": row, "condition": condition,
                                  "messages": messages, "rendered_prompt": rendered})

    generation = protocol["generation"]
    llm = LLM(
        model=model_path,
        revision=model_revision,
        tokenizer=model_path,
        tokenizer_revision=model_revision,
        dtype="half",
        max_model_len=int(generation["max_sequence_length"]),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=adapter_path is not None,
        max_lora_rank=16,
        enforce_eager=True,
        trust_remote_code=False,
    )
    request = LoRARequest(args.checkpoint, 1, adapter_path) if adapter_path else None
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
    )
    outputs = llm.generate(
        [item["rendered_prompt"] for item in conversations], sampling,
        lora_request=request,
    )

    verifier = MathVerifier()
    records: list[dict[str, Any]] = []
    for item, output in zip(conversations, outputs):
        row = item["source"]
        raw = output.outputs[0].text
        action = parse_review(raw)
        if action.valid:
            final_answer = row["initial_output"] if action.decision == "KEEP" else action.revised_answer
            verification = verifier.verify(build_problem(row), final_answer or "")
            final_correct: bool | None = verification.passed
            detail: str | None = verification.detail
            transition: str | None = transition_name(bool(row["initial_correct"]), final_correct)
        else:
            final_answer = None
            final_correct = None
            detail = None
            transition = None
        records.append({
            "schema_version": "phase5_guided_review_v1",
            "checkpoint": args.checkpoint,
            "split": args.split,
            "condition": item["condition"],
            "problem_id": row["problem_id"],
            "initial_correct": bool(row["initial_correct"]),
            "messages": item["messages"],
            "rendered_prompt": item["rendered_prompt"],
            "raw_model_output": raw,
            "generated_token_ids": list(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
            "strict_contract_valid": action.valid,
            "decision": action.decision,
            "revised_answer": action.revised_answer,
            "final_answer": final_answer,
            "final_correct": final_correct,
            "final_verifier_detail": detail,
            "transition": transition,
            "protocol_lock_sha256_lf": sha256_lf(LOCK_PATH),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"{args.checkpoint}_{args.split}.jsonl"
    summary_path = args.output_dir / f"{args.checkpoint}_{args.split}_summary.json"
    results_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                            encoding="utf-8")
    by_condition: dict[str, dict[str, Any]] = {}
    for condition in protocol["scope"]["primary_conditions"]:
        subset = [record for record in records if record["condition"] == condition]
        valid = [record for record in subset if record["strict_contract_valid"]]
        by_condition[condition] = {
            "rows": len(subset),
            "strict_valid": len(valid),
            "strict_contract_validity": len(valid) / len(subset),
            "keep": sum(record["decision"] == "KEEP" for record in valid),
            "revise": sum(record["decision"] == "REVISE" for record in valid),
            "final_correct": sum(record["final_correct"] is True for record in subset),
            "transitions": dict(Counter(record["transition"] for record in valid)),
        }
    valid_total = sum(record["strict_contract_valid"] for record in records)
    minimum = float(protocol["smoke_gate"]["minimum_contract_validity"])
    summary = {
        "schema_version": "phase5_guided_review_summary_v1",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "backend": "vllm_0_7_0",
        "dtype": "float16",
        "model": model_path,
        "model_revision": model_revision,
        "adapter_enabled": adapter_path is not None,
        "rows": len(records),
        "source_rows": len(rows),
        "strict_valid": valid_total,
        "strict_contract_validity": valid_total / len(records),
        "minimum_contract_validity": minimum,
        "smoke_gate_passed": args.split != "smoke" or valid_total / len(records) >= minimum,
        "by_condition": by_condition,
        "results_sha256_lf": sha256_lf(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.split == "smoke" and not summary["smoke_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
