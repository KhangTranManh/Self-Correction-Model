"""Collect frozen Phase 7 paired outputs, one checkpoint and split at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem
from phase7.scripts.collect_initials_4bit import (
    append_durable, read_jsonl, sha256, write_atomic,
)
from phase7.scripts.paired_prompts import paired_prompt


CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"
REGISTRY = ROOT / "phase7/configs/experiments.yaml"
LOCK = ROOT / "phase7/data/protocol/paired_generation_v1_lock.json"
ARMS = ("blind_resolve", "answer_visible_resolve")
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def request_seed(base: int, split: str, checkpoint: str, problem_id: str) -> int:
    value = f"phase7-paired-v1|{base}|{split}|{checkpoint}|{problem_id}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "protected"), required=True)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/phase7_paired_v1")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use CPython 3.10 for Phase 7")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if vllm.__version__ != "0.7.0" or not torch.cuda.is_available():
        raise RuntimeError("Pinned vLLM 0.7.0 and CUDA are required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Paired protocol hash mismatch: {relative}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    split_dir = ROOT / "phase7/data/split_v1"
    split_report = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))
    ids_path = split_dir / f"{args.split}_ids.json"
    if sha256(ids_path) != split_report[f"{args.split}_ids_sha256"]:
        raise ValueError("Selected split hash mismatch")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    initial_dir = ROOT / "outputs/phase7_initials_v1"
    if sha256(initial_dir / "initial_rollouts.jsonl") != split_report["initial_rollouts_sha256"]:
        raise ValueError("Initial answers differ from the frozen split")
    initials = {row["problem_id"]: row for row in
                read_jsonl(initial_dir / "initial_rollouts.jsonl")}
    source_plan = config["source_plan"]
    source_path = ROOT / source_plan["candidate_manifest"]
    if sha256(source_path) != source_plan["candidate_manifest_sha256"]:
        raise ValueError("Candidate source manifest changed")
    sources = {row["id"]: row for row in read_jsonl(source_path)}
    if len(ids) != len(set(ids)) or any(pid not in initials or pid not in sources for pid in ids):
        raise ValueError("Split contains missing or duplicate IDs")
    generation = config["generation_draft"]
    if generation["backend"] != "vllm_0_7_0" or generation["dtype"] != "float16":
        raise ValueError("Unexpected paired generation backend")
    max_tokens = int(generation["max_new_tokens"])
    max_len = int(generation["max_model_len"])
    base_seed = int(generation["seed"])
    original = registry["models"]["original_solver"]
    if args.checkpoint == "original_solver":
        model_path, revision = original["repo"], original["revision"]
        lora_path = None
    else:
        model_path = str(ROOT / "models/phase7_v2_merged_fp16")
        marker = json.loads((Path(model_path) / "phase7_lineage.json").read_text(encoding="utf-8"))
        if (marker["original_revision"] != original["revision"] or
                marker["adapter_weight_sha256"] !=
                registry["models"]["warmstart_v2"]["adapter_sha256"]):
            raise ValueError("V2 merged model lineage mismatch")
        revision = None
        lora_path = None
        if args.checkpoint == "correction_sft_v3":
            lora_path = ROOT / "outputs/phase4_correction_sft_v3/final_adapter"
            if sha256(lora_path / "adapter_model.safetensors") != registry["models"]["correction_sft_v3"]["adapter_sha256"]:
                raise ValueError("V3 LoRA weight hash mismatch")
    output_dir = args.output_root / args.split / args.checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "paired_outputs.audit.jsonl"
    settings = {
        "schema_version": "phase7_paired_outputs_v1", "split": args.split,
        "checkpoint": args.checkpoint, "model_path": model_path,
        "revision": revision, "lora_path": str(lora_path) if lora_path else None,
        "split_ids_sha256": sha256(ids_path),
        "initial_rollouts_sha256": sha256(initial_dir / "initial_rollouts.jsonl"),
        "protocol_lock_sha256": sha256(LOCK), "vllm": vllm.__version__,
        "temperature": float(generation["temperature"]),
        "max_tokens": max_tokens, "max_model_len": max_len,
        "gpu_memory_utilization": float(generation["gpu_memory_utilization"]),
        "seed": base_seed, "dtype": "float16",
    }
    expected_tasks = [(pid, arm) for pid in ids for arm in ARMS]
    if audit_path.exists():
        audit = read_jsonl(audit_path)
        if not audit or audit[0] != {"type": "metadata", "settings": settings}:
            raise ValueError("Existing paired audit settings differ")
        completed = audit[1:]
        for index, record in enumerate(completed):
            if (index >= len(expected_tasks) or record["type"] != "completion" or
                    record["index"] != index or
                    (record["row"]["problem_id"], record["row"]["arm"]) != expected_tasks[index]):
                raise ValueError("Existing paired audit is not ordered prefix")
    else:
        append_durable(audit_path, {"type": "metadata", "settings": settings})
        completed = []
    if len(completed) < len(expected_tasks):
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        llm = LLM(model=model_path, revision=revision, tokenizer=model_path,
                  tokenizer_revision=revision, dtype="half", max_model_len=max_len,
                  gpu_memory_utilization=float(generation["gpu_memory_utilization"]),
                  enforce_eager=True, trust_remote_code=False,
                  enable_lora=bool(lora_path), max_loras=1, max_lora_rank=16)
        lora_request = (LoRARequest("phase7_v3", 1, str(lora_path)) if lora_path else None)
        for index in range(len(completed), len(expected_tasks)):
            pid, arm = expected_tasks[index]
            source = sources[pid]
            problem = Problem(id=pid, domain="math", question=source["question"],
                              reference_answer=source["reference_answer"])
            previous = initials[pid]["initial_output"]
            blind = paired_prompt(problem, "blind_resolve")
            if blind != paired_prompt(problem, "blind_resolve", "SENTINEL_UNUSED"):
                raise RuntimeError("Blind prompt depends on previous answer")
            prompt = blind if arm == "blind_resolve" else paired_prompt(
                problem, arm, previous)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + max_tokens > max_len:
                raise ValueError(f"Context budget exceeded: {pid} {arm}")
            seed = request_seed(base_seed, args.split, args.checkpoint, pid)
            sampling = SamplingParams(temperature=float(generation["temperature"]),
                                      max_tokens=max_tokens, seed=seed)
            result = llm.generate([rendered], sampling, lora_request=lora_request,
                                  use_tqdm=False)[0].outputs[0]
            row = {"problem_id": pid, "arm": arm, "index": index,
                   "request_seed": seed, "output": result.text,
                   "generated_tokens": len(result.token_ids),
                   "finish_reason": result.finish_reason,
                   "hit_token_cap": len(result.token_ids) >= max_tokens}
            append_durable(audit_path, {"type": "completion", "index": index, "row": row})
            completed.append({"row": row})
            if (index + 1) % 10 == 0:
                print(f"SAVED {index + 1}/{len(expected_tasks)} {args.split} {args.checkpoint}",
                      flush=True)
    rows = [item["row"] for item in completed]
    output_path = output_dir / "paired_outputs.jsonl"
    write_atomic(output_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                  for row in rows))
    summary = {"status": "complete", "tasks": len(rows),
               "audit_sha256": sha256(audit_path),
               "outputs_sha256": sha256(output_path), "settings": settings}
    write_atomic(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "tasks": len(rows),
                      "checkpoint": args.checkpoint, "split": args.split}))


if __name__ == "__main__":
    main()
