"""Collect frozen Phase 8 blind, own-answer, and distractor arms without gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit  # noqa: E402


SOURCE = ROOT / "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl"
FIRST = ROOT / "outputs/phase8_first_pass_v2/initial/answers.jsonl"
DONORS = ROOT / "phase8/data/distractors_v2/assignments.jsonl"
MODELS = ("original_solver", "warmstart_v2", "correction_sft_v3")
ARMS = ("blind", "own_visible", "distractor")
BASE = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def seed_for(checkpoint: str, pid: str) -> int:
    return int(hashlib.sha256(f"phase8_three_arms_v1|{checkpoint}|{pid}".encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=MODELS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/phase8_three_arms_v2")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not torch.cuda.is_available() or vllm.__version__ != "0.7.0":
        raise RuntimeError("Pinned vLLM 0.7.0 and CUDA required")
    source_report = json.loads(SOURCE.with_name("candidate_report.json").read_text(encoding="utf-8"))
    first_report = json.loads(FIRST.with_name("summary.json").read_text(encoding="utf-8"))
    donor_report = json.loads(DONORS.with_name("report.json").read_text(encoding="utf-8"))
    if (sha256(SOURCE) != source_report["candidate_manifest_sha256"] or
            sha256(FIRST) != first_report["answers_sha256"] or
            sha256(DONORS) != donor_report["assignments_sha256"]):
        raise RuntimeError("Frozen source, first-answer, or distractor hash mismatch")
    sources = read_jsonl(SOURCE)
    first = {row["problem_id"]: row for row in read_jsonl(FIRST)}
    donors = {row["problem_id"]: row for row in read_jsonl(DONORS)}
    if len(sources) != len(first) or len(sources) != len(donors) or len(sources) != 400:
        raise RuntimeError("Source/answer/donor counts differ")
    lora = None
    if args.checkpoint == "original_solver":
        model_path, revision = BASE, REVISION
    else:
        model_path, revision = str(ROOT / "models/phase7_v2_merged_fp16"), None
        marker = json.loads((Path(model_path) / "phase7_lineage.json").read_text(encoding="utf-8"))
        registry = json.loads((ROOT / "phase8/data/model_lineage_v1.json").read_text(encoding="utf-8"))
        if marker != registry["v2_merged_lineage"]:
            raise RuntimeError("V2 merged lineage changed")
        if args.checkpoint == "correction_sft_v3":
            lora = ROOT / "outputs/phase4_correction_sft_v3/final_adapter"
            if sha256(lora / "adapter_model.safetensors") != registry["v3_adapter_sha256"]:
                raise RuntimeError("V3 adapter hash mismatch")
    settings = {
        "schema_version": "phase8_three_arms_v2", "checkpoint": args.checkpoint,
        "batch_size": BATCH_SIZE,
        "model_path": model_path, "revision": revision,
        "lora_path": str(lora) if lora else None,
        "source_sha256": sha256(SOURCE), "first_answers_sha256": sha256(FIRST),
        "distractor_map_sha256": sha256(DONORS),
        "temperature": 0.0, "max_new_tokens": 768, "max_model_len": 4096,
        "gpu": torch.cuda.get_device_name(0), "gold_labels_opened": False,
    }
    tasks = [(source["id"], arm) for source in sources for arm in ARMS
             if arm != "distractor" or donors[source["id"]]["donor_problem_id"]]
    output_dir = args.output_root / args.checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = output_dir / "audit.jsonl"
    completed = open_audit(audit, settings, tasks, lambda row: (row["problem_id"], row["arm"]))
    if len(completed) < len(tasks):
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        llm = LLM(model=model_path, revision=revision, tokenizer=model_path,
                  tokenizer_revision=revision, dtype="half", max_model_len=4096,
                  gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=False,
                  enable_lora=bool(lora), max_loras=1, max_lora_rank=16)
        lora_request = LoRARequest("phase8_v3", 1, str(lora)) if lora else None
        source_by_id = {row["id"]: row for row in sources}
        requests = []
        for pid, arm in tasks:
            problem = Problem(id=pid, domain="math", question=source_by_id[pid]["question"])
            if arm == "blind":
                prompt = paired_prompt(problem, "blind_resolve")
            elif arm == "own_visible":
                prompt = paired_prompt(problem, "answer_visible_resolve", first[pid]["output"])
            else:
                prompt = paired_prompt(problem, "answer_visible_resolve", donors[pid]["donor_output"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + 768 > 4096:
                raise RuntimeError(f"Context exceeds budget: {pid}, {arm}")
            requests.append((rendered, SamplingParams(
                temperature=0.0, max_tokens=768, seed=seed_for(args.checkpoint, pid))))

        def make_row(index: int, result) -> dict:
            pid, arm = tasks[index]
            return {"problem_id": pid, "arm": arm, "index": index,
                    "request_seed": seed_for(args.checkpoint, pid), "output": result.text,
                    "generated_tokens": len(result.token_ids),
                    "finish_reason": result.finish_reason,
                    "hit_token_cap": len(result.token_ids) >= 768}

        completed = generate_ordered(llm, requests, len(completed), audit, make_row,
                                     args.checkpoint, lora_request=lora_request, done=completed)
    output = output_dir / "answers.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                              for row in completed), encoding="utf-8", newline="\n")
    summary = {"status": "complete", "tasks": len(completed),
               "audit_sha256": sha256(audit), "answers_sha256": sha256(output),
               "settings": settings}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
