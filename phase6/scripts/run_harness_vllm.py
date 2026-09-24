"""Run initial, confidence, or routed recheck generations for Phase 6 holdout."""

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


CONFIG = ROOT / "phase6/configs/harness_confirmation_v1.yaml"
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
SOURCE_POOL = ROOT / "phase6/data/harness_confirmation_source_pool_v1.jsonl"
SOURCE_LOCK = ROOT / "phase6/data/harness_confirmation_source_pool_v1_lock.json"
HOLDOUT = ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl"
HOLDOUT_LOCK = ROOT / "phase6/data/harness_confirmation_holdout_v1_lock.json"
CONFIDENCE_RE = re.compile(r"\A\s*<confidence>(100|[0-9]{1,2})</confidence>\s*\Z")


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


def verify_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256_lf(ROOT / relative) != expected["sha256_lf"]:
            raise RuntimeError(f"Frozen input changed: {relative}")
    return lock


def build_problem(row: dict[str, Any]) -> Problem:
    return Problem(id=row["problem_id"], domain="math", question=row["question"],
                   reference_answer=row["reference_answer"])


def resolve_adapter(repo: str, revision: str, expected_sha: str, token: str | None) -> str:
    path = Path(snapshot_download(repo_id=repo, revision=revision, token=token))
    if sha256_file(path / "adapter_model.safetensors") != expected_sha:
        raise RuntimeError(f"Adapter hash mismatch: {repo}")
    return str(path)


def model_spec(checkpoint: str, registry: dict[str, Any], token: str | None,
               merged_v2: Path) -> tuple[str, str | None, str | None]:
    repos = registry["checkpoint_hub_repos"]
    revisions = registry["checkpoint_hub_revisions"]
    hashes = registry["checkpoint_adapter_sha256"]
    if checkpoint == "original_solver":
        return repos["original_solver"], revisions["original_solver"], None
    if checkpoint == "warmstart_v2":
        return repos["original_solver"], revisions["original_solver"], resolve_adapter(
            repos["phase4_warmstart_v2"], revisions["phase4_warmstart_v2"],
            hashes["phase4_warmstart_v2"], token)
    marker = merged_v2 / "phase5_lineage.json"
    if not marker.exists():
        raise RuntimeError(f"Missing verified V2 merged parent: {marker}")
    return str(merged_v2), None, resolve_adapter(
        repos["phase4_correction_sft_v3"], revisions["phase4_correction_sft_v3"],
        hashes["phase4_correction_sft_v3"], token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("initial", "confidence", "recheck"))
    parser.add_argument("--checkpoint", choices=("original_solver", "warmstart_v2", "correction_sft_v3"))
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--routes-lock", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-v2", type=Path, default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()
    if args.stage == "initial" and args.checkpoint is not None:
        raise RuntimeError("Initial generation always uses original_solver")
    if args.stage != "initial" and args.checkpoint is None:
        raise RuntimeError("Confidence/recheck require --checkpoint")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    if args.stage == "initial":
        verify_lock(SOURCE_LOCK)
        rows = read_jsonl(SOURCE_POOL)
        checkpoint = "original_solver"
    else:
        verify_lock(HOLDOUT_LOCK)
        rows = read_jsonl(HOLDOUT)
        checkpoint = str(args.checkpoint)
    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_path, revision, adapter_path = model_spec(checkpoint, registry, token, args.merged_v2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, token=token)
    messages: list[list[dict[str, str]]] = []
    metadata: list[dict[str, Any]] = []
    if args.stage == "initial":
        for row in rows:
            messages.append([{"role": "user", "content": build_prompt(build_problem(row))}])
            metadata.append({"source": row})
        max_tokens = int(config["generation"]["initial_max_tokens"])
    elif args.stage == "confidence":
        for row in rows:
            messages.append([
                {"role": "system", "content": "Estimate correctness only; do not revise."},
                {"role": "user", "content": build_prompt(build_problem(row))},
                {"role": "assistant", "content": row["initial_output"]},
                {"role": "user", "content": "Give the probability from 0 to 100 that the previous answer is fully correct. Output exactly <confidence>N</confidence>, where N is one integer from 0 through 100."},
            ])
            metadata.append({"source": row})
        max_tokens = int(config["generation"]["confidence_max_tokens"])
    else:
        if args.routes is None:
            raise RuntimeError("--routes is required for recheck")
        if args.routes_lock is None:
            raise RuntimeError("--routes-lock is required for recheck")
        route_lock = json.loads(args.routes_lock.read_text(encoding="utf-8"))
        # CLI invocations commonly pass a repository-relative path.  Resolve
        # both paths before deriving the lock key so the provenance check is
        # identical for relative and absolute invocations.
        relative_routes = args.routes.resolve().relative_to(ROOT.resolve()).as_posix()
        expected_routes = route_lock["files"].get(relative_routes, {}).get("sha256_lf")
        if expected_routes is None or sha256_lf(args.routes) != expected_routes:
            raise RuntimeError("Frozen routes mismatch")
        routes = [row for row in read_jsonl(args.routes) if row["checkpoint"] == checkpoint]
        by_id = {row["problem_id"]: row for row in rows}
        for route in routes:
            row = by_id[route["problem_id"]]
            prompt_key = {
                "self_confidence": "self_confidence_router",
                "frozen_probe": "frozen_probe_router",
                "oracle_known_wrong": "oracle_known_wrong",
            }[route["condition"]]
            messages.append([
                {"role": "system", "content": "Return only a complete answer to the original problem."},
                {"role": "user", "content": build_prompt(build_problem(row))},
                {"role": "assistant", "content": row["initial_output"]},
                {"role": "user", "content": config[prompt_key]["routing_prompt"]},
            ])
            metadata.append({"source": row, "route": route})
        max_tokens = int(config["generation"]["recheck_max_tokens"])
    rendered = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
                for item in messages]
    llm = LLM(model=model_path, revision=revision, tokenizer=model_path,
              tokenizer_revision=revision, dtype="half",
              max_model_len=int(config["generation"]["max_model_len"]),
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_lora=adapter_path is not None, max_lora_rank=16,
              enforce_eager=True, trust_remote_code=False)
    request = LoRARequest(checkpoint, 1, adapter_path) if adapter_path else None
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            seed=int(config["generation"]["seed"]))
    outputs = llm.generate(rendered, params, lora_request=request)
    verifier = MathVerifier()
    records: list[dict[str, Any]] = []
    for item, prompt, output in zip(metadata, messages, outputs):
        row = item["source"]
        raw = output.outputs[0].text
        record: dict[str, Any] = {
            "schema_version": "phase6_harness_confirmation_generation_v1",
            "stage": args.stage, "checkpoint": checkpoint,
            "problem_id": row["problem_id"], "initial_correct": row.get("initial_correct"),
            "messages": prompt, "raw_model_output": raw,
            "generated_token_ids": list(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
        }
        if args.stage == "initial":
            verification = verifier.verify(build_problem(row), raw)
            record.update({**row, "initial_output": raw,
                           "initial_correct": bool(verification.passed),
                           "initial_verifier_detail": verification.detail})
        elif args.stage == "confidence":
            match = CONFIDENCE_RE.fullmatch(raw)
            record.update({"strict_contract_valid": match is not None,
                           "confidence_correct": int(match.group(1)) if match else None})
        else:
            verification = verifier.verify(build_problem(row), raw)
            record.update({"condition": item["route"]["condition"],
                           "route_score_probability_wrong": item["route"]["probability_wrong"],
                           "final_correct": bool(verification.passed),
                           "final_verifier_detail": verification.detail})
        records.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.stage}_{checkpoint}.jsonl"
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                           encoding="utf-8", newline="\n")
    summary = {"stage": args.stage, "checkpoint": checkpoint, "rows": len(records),
               "output_sha256_lf": sha256_lf(output_path)}
    (args.output_dir / f"{args.stage}_{checkpoint}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
