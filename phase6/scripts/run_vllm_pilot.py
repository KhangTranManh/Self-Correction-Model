"""Run confidence verbalization and rationale review with one Phase 5 checkpoint."""

from __future__ import annotations

import argparse
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


CONFIG = ROOT / "phase6/configs/verbalization_pilot_v1.yaml"
MANIFEST = ROOT / "phase6/data/pilot_v1.jsonl"
LOCK = ROOT / "phase6/data/pilot_v1_lock.json"
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
CONFIDENCE_RE = re.compile(r"\A\s*<confidence>(100|[0-9]{1,2})</confidence>\s*\Z")
REVIEW_RE = re.compile(
    r"\A\s*<review>(.+?)</review>\s*<decision>(KEEP|REVISE)</decision>"
    r"(?:\s*<answer>(.*?)</answer>)?\s*\Z",
    flags=re.DOTALL,
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


def problem(row: dict[str, Any]) -> Problem:
    return Problem(id=row["problem_id"], domain="math", question=row["question"],
                   reference_answer=row["reference_answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        choices=("original_solver", "warmstart_v2", "correction_sft_v3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-v2", type=Path,
                        default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    lock = verify_lock()
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
        if not marker.exists():
            raise RuntimeError(f"Missing merged V2 lineage marker: {marker}")
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
            raise RuntimeError("Merged V2 lineage mismatch")
        model_path = str(args.merged_v2)
        adapter_path = resolve_adapter(
            repos["phase4_correction_sft_v3"], revisions["phase4_correction_sft_v3"],
            hashes["phase4_correction_sft_v3"], token,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision, token=token)
    rendered: dict[str, list[str]] = {"confidence": [], "rationale_review": []}
    messages_by_kind: dict[str, list[list[dict[str, str]]]] = {
        "confidence": [], "rationale_review": [],
    }
    for row in rows:
        prefix = [
            {"role": "user", "content": build_prompt(problem(row))},
            {"role": "assistant", "content": row["initial_output"]},
        ]
        for kind, section in (("confidence", config["confidence"]),
                              ("rationale_review", config["rationale_review"])):
            messages = ([{"role": "system", "content": section["system_prompt"]}]
                        + prefix + [{"role": "user", "content": section["user_prompt"]}])
            messages_by_kind[kind].append(messages)
            rendered[kind].append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

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
    generated = {}
    for kind, max_key in (("confidence", "confidence_max_tokens"),
                          ("rationale_review", "rationale_max_tokens")):
        params = SamplingParams(temperature=0.0, max_tokens=int(generation[max_key]),
                                seed=int(generation["seed"]))
        generated[kind] = llm.generate(rendered[kind], params, lora_request=request)

    verifier = MathVerifier()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        for kind in ("confidence", "rationale_review"):
            result = generated[kind][index].outputs[0]
            raw = result.text
            record: dict[str, Any] = {
                "schema_version": "phase6_verbalization_result_v1",
                "checkpoint": args.checkpoint, "kind": kind,
                "problem_id": row["problem_id"],
                "initial_correct": bool(row["initial_correct"]),
                "messages": messages_by_kind[kind][index],
                "rendered_prompt": rendered[kind][index],
                "raw_model_output": raw,
                "generated_token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason,
                "pilot_lock_sha256_lf": sha256_lf(LOCK),
            }
            if kind == "confidence":
                match = CONFIDENCE_RE.fullmatch(raw)
                record.update({
                    "strict_contract_valid": match is not None,
                    "confidence_correct": int(match.group(1)) if match else None,
                })
            else:
                match = REVIEW_RE.fullmatch(raw)
                review = match.group(1).strip() if match else None
                decision = match.group(2) if match else None
                answer = match.group(3).strip() if match and match.group(3) else None
                valid = bool(match and review and
                             ((decision == "KEEP" and answer is None)
                              or (decision == "REVISE" and answer)))
                final_answer = (row["initial_output"] if decision == "KEEP" else answer) if valid else None
                verified = verifier.verify(problem(row), final_answer or "") if valid else None
                final_correct = bool(verified.passed) if verified else None
                if valid and not row["initial_correct"] and final_correct:
                    transition = "wrong_to_correct"
                elif valid and row["initial_correct"] and not final_correct:
                    transition = "correct_to_wrong"
                elif valid and row["initial_correct"] and final_correct:
                    transition = "correct_to_correct"
                elif valid:
                    transition = "wrong_to_wrong"
                else:
                    transition = None
                record.update({
                    "strict_contract_valid": valid, "review_text": review,
                    "decision": decision, "revised_answer": answer,
                    "final_answer": final_answer, "final_correct": final_correct,
                    "final_verifier_detail": verified.detail if verified else None,
                    "transition": transition,
                })
            records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.checkpoint}.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                      encoding="utf-8", newline="\n")
    summary = {
        "schema_version": "phase6_verbalization_summary_v1",
        "checkpoint": args.checkpoint,
        "sources": len(rows), "rows": len(records),
        "confidence_valid": sum(r["strict_contract_valid"] for r in records
                                if r["kind"] == "confidence"),
        "rationale_valid": sum(r["strict_contract_valid"] for r in records
                               if r["kind"] == "rationale_review"),
        "rationale_fixes": sum(r.get("transition") == "wrong_to_correct" for r in records),
        "rationale_harms": sum(r.get("transition") == "correct_to_wrong" for r in records),
        "output_sha256_lf": sha256_lf(output),
    }
    summary_path = args.output_dir / f"{args.checkpoint}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
