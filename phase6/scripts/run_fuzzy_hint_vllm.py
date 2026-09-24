"""Run the frozen fuzzy-hint curve for one checkpoint with vLLM."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
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


CONFIG = ROOT / "phase6/configs/fuzzy_hint_pilot_v1.yaml"
MANIFEST = ROOT / "phase6/data/fuzzy_hint_pilot_v1.jsonl"
LOCK = ROOT / "phase6/data/fuzzy_hint_pilot_v1_lock.json"
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
KEEP_RE = re.compile(r"\A\s*<decision>KEEP</decision>\s*\Z", re.DOTALL)
REVISE_RE = re.compile(
    r"\A\s*<decision>REVISE</decision>\s*<answer>(.+)</answer>\s*\Z", re.DOTALL
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        observed = sha256_lf(ROOT / relative)
        if observed != expected["sha256_lf"]:
            raise RuntimeError(f"Frozen input changed: {relative}")
    return lock


def resolve_adapter(repo: str, revision: str, expected_sha: str,
                    token: str | None) -> str:
    path = Path(snapshot_download(repo_id=repo, revision=revision, token=token))
    observed = sha256_file(path / "adapter_model.safetensors")
    if observed != expected_sha:
        raise RuntimeError(f"Adapter hash mismatch for {repo}: {observed}")
    return str(path)


def build_problem(row: dict[str, Any]) -> Problem:
    return Problem(id=row["problem_id"], domain="math", question=row["question"],
                   reference_answer=row["reference_answer"])


def parse_contract(raw: str) -> tuple[bool, str, str | None]:
    if KEEP_RE.fullmatch(raw):
        return True, "KEEP", None
    match = REVISE_RE.fullmatch(raw)
    if match and match.group(1).strip():
        return True, "REVISE", match.group(1).strip()
    return False, "INVALID", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        choices=("original_solver", "warmstart_v2", "correction_sft_v3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-v2", type=Path,
                        default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    verify_lock()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    rows = read_jsonl(MANIFEST)
    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
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
        lineage = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        expected = {
            "base_repo": repos["original_solver"],
            "base_revision": revisions["original_solver"],
            "adapter_repo": repos["phase4_warmstart_v2"],
            "adapter_revision": revisions["phase4_warmstart_v2"],
            "adapter_weight_sha256": hashes["phase4_warmstart_v2"],
            "weight_dtype": "float16",
        }
        if lineage != expected:
            raise RuntimeError("Merged V2 lineage mismatch")
        model_path = str(args.merged_v2)
        adapter_path = resolve_adapter(
            repos["phase4_correction_sft_v3"], revisions["phase4_correction_sft_v3"],
            hashes["phase4_correction_sft_v3"], token,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision, token=token)
    conversations: list[dict[str, Any]] = []
    for row in rows:
        for probability in config["intervention"]["probability_wrong_percent"]:
            hint = config["intervention"]["estimator_statement"].format(
                probability=probability)
            messages = [
                {"role": "system", "content": config["conversation"]["system_prompt"]},
                {"role": "user", "content": build_prompt(build_problem(row))},
                {"role": "assistant", "content": row["initial_output"]},
                {"role": "user", "content":
                 f"{hint}\n\n{config['conversation']['contract_instruction']}"},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            conversations.append({"source": row, "probability": int(probability),
                                  "messages": messages, "rendered_prompt": rendered})

    generation = config["generation"]
    llm = LLM(
        model=model_path, revision=model_revision, tokenizer=model_path,
        tokenizer_revision=model_revision, dtype="half",
        max_model_len=int(generation["max_model_len"]),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=adapter_path is not None, max_lora_rank=16,
        enforce_eager=True, trust_remote_code=False,
    )
    request = LoRARequest(args.checkpoint, 1, adapter_path) if adapter_path else None
    sampling = SamplingParams(temperature=0.0, max_tokens=int(generation["max_new_tokens"]),
                              seed=int(generation["seed"]))
    outputs = llm.generate([item["rendered_prompt"] for item in conversations], sampling,
                           lora_request=request)
    verifier = MathVerifier()
    records: list[dict[str, Any]] = []
    for item, output in zip(conversations, outputs):
        row = item["source"]
        result = output.outputs[0]
        valid, decision, revised = parse_contract(result.text)
        if valid:
            final_answer = row["initial_output"] if decision == "KEEP" else revised
            verification = verifier.verify(build_problem(row), final_answer or "")
            final_correct: bool | None = bool(verification.passed)
            detail: str | None = verification.detail
            if row["initial_correct"] and final_correct:
                transition = "correct_to_correct"
            elif row["initial_correct"]:
                transition = "correct_to_wrong"
            elif final_correct:
                transition = "wrong_to_correct"
            else:
                transition = "wrong_to_wrong"
        else:
            final_answer = None
            final_correct = None
            detail = None
            transition = None
        records.append({
            "schema_version": "phase6_fuzzy_hint_result_v1",
            "checkpoint": args.checkpoint,
            "problem_id": row["problem_id"],
            "initial_correct": bool(row["initial_correct"]),
            "probability_wrong_percent": item["probability"],
            "messages": item["messages"],
            "rendered_prompt": item["rendered_prompt"],
            "raw_model_output": result.text,
            "generated_token_ids": list(result.token_ids),
            "finish_reason": result.finish_reason,
            "strict_contract_valid": valid,
            "decision": decision,
            "revised_answer": revised,
            "final_answer": final_answer,
            "final_correct": final_correct,
            "final_verifier_detail": detail,
            "transition": transition,
            "pilot_lock_sha256_lf": sha256_lf(LOCK),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.checkpoint}.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                      encoding="utf-8", newline="\n")
    by_probability = {}
    for probability in config["intervention"]["probability_wrong_percent"]:
        subset = [row for row in records if row["probability_wrong_percent"] == probability]
        by_probability[str(probability)] = {
            "rows": len(subset),
            "strict_valid": sum(row["strict_contract_valid"] for row in subset),
            "wrong_to_correct": sum(row["transition"] == "wrong_to_correct" for row in subset),
            "correct_to_wrong": sum(row["transition"] == "correct_to_wrong" for row in subset),
            "decisions": dict(Counter(row["decision"] for row in subset)),
        }
    summary = {
        "schema_version": "phase6_fuzzy_hint_summary_v1",
        "checkpoint": args.checkpoint,
        "sources": len(rows), "rows": len(records),
        "by_probability": by_probability,
        "output_sha256_lf": sha256_lf(output),
    }
    (args.output_dir / f"{args.checkpoint}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

