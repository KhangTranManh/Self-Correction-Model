"""Per-checkpoint Phase 9 generation: four blind attempts, then the self-check judge.

Blind attempts use the exact Phase 7/8 blind prompt at temperature 0.7 so they
are independent samples. The judge sees the problem and two unverified
solutions (the shared first answer A0 and this checkpoint's first blind
attempt B1) and writes its own final solution greedily. The judge output is
generated for every source; the analysis applies it only where A0 and B1
disagree, a gold-free rule fixed in advance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit, read_jsonl  # noqa: E402
from phase9.scripts.collect_first import sha256, sources  # noqa: E402

FIRST = ROOT / "outputs/phase9_first_v1"
OUT_ROOT = ROOT / "outputs/phase9_checkpoints_v1"
BASE = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
MERGED_V2 = ROOT / "models/phase7_v2_merged_fp16"
V3_ADAPTER = ROOT / "outputs/phase4_correction_sft_v3/final_adapter"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
ATTEMPTS = 4
JUDGE_HEADER = (
    "\n\nTwo independent solutions to this problem are shown below. Either or both "
    "may be wrong, and their final answers may differ. Check each solution step by "
    "step, then write your own complete, correct solution."
)


def judge_prompt(problem: Problem, first: str, blind: str) -> str:
    return (build_prompt(problem) + JUDGE_HEADER + "\n\nSolution A:\n" + first
            + "\n\nSolution B:\n" + blind)


def seed(kind: str, checkpoint: str, pid: str, attempt: int = 0) -> int:
    key = f"phase9_{kind}_v1|{checkpoint}|{pid}|{attempt}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=CHECKPOINTS)
    args = parser.parse_args()
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if sys.version_info[:2] != (3, 10) or vllm.__version__ != "0.7.0":
        raise RuntimeError("Python 3.10 and vLLM 0.7.0 required")
    rows = sources()
    first_summary = json.loads((FIRST / "summary.json").read_text(encoding="utf-8"))
    if first_summary["status"] != "complete" or \
            sha256(FIRST / "answers.jsonl") != first_summary["answers_sha256"]:
        raise RuntimeError("First answers incomplete or changed")
    first = {row["problem_id"]: row["output"] for row in read_jsonl(FIRST / "answers.jsonl")}
    lineage = json.loads((ROOT / "phase8/data/model_lineage_v1.json").read_text(encoding="utf-8"))
    lora = None
    if args.checkpoint == "original_solver":
        model_path, revision = BASE, REVISION
    else:
        model_path, revision = str(MERGED_V2), None
        marker = json.loads((MERGED_V2 / "phase7_lineage.json").read_text(encoding="utf-8"))
        if marker != lineage["v2_merged_lineage"]:
            raise RuntimeError("Merged V2 lineage changed")
        if args.checkpoint == "correction_sft_v3":
            lora = V3_ADAPTER
            if sha256(lora / "adapter_model.safetensors") != lineage["v3_adapter_sha256"]:
                raise RuntimeError("V3 adapter hash mismatch")
    out = OUT_ROOT / args.checkpoint
    common = {"checkpoint": args.checkpoint, "model_path": model_path, "revision": revision,
              "lora_path": str(lora) if lora else None, "max_new_tokens": 768,
              "max_model_len": 4096, "batch_size": BATCH_SIZE,
              "first_answers_sha256": first_summary["answers_sha256"],
              "gold_labels_opened": False}
    blind_settings = dict(common, schema_version="phase9_blind_v1", temperature=0.7,
                          top_p=1.0, top_k=-1, attempts=ATTEMPTS)
    blind_tasks = [(row["id"], k) for row in rows for k in range(1, ATTEMPTS + 1)]
    blind_audit = out / "blind_audit.jsonl"
    blind_done = open_audit(blind_audit, blind_settings, blind_tasks,
                            lambda row: (row["problem_id"], row["attempt"]))
    judge_settings = dict(common, schema_version="phase9_judge_v1", temperature=0.0,
                          judge_header=JUDGE_HEADER)
    judge_audit = out / "judge_audit.jsonl"
    judge_done = open_audit(judge_audit, judge_settings, [row["id"] for row in rows],
                            lambda row: row["problem_id"])
    by_id = {row["id"]: row for row in rows}

    if len(blind_done) < len(blind_tasks) or len(judge_done) < len(rows):
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        llm = LLM(model=model_path, revision=revision, tokenizer=model_path,
                  tokenizer_revision=revision, dtype="half", max_model_len=4096,
                  gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=False,
                  enable_lora=bool(lora), max_loras=1, max_lora_rank=16)
        lora_request = LoRARequest("phase9_v3", 1, str(lora)) if lora else None

        def render(prompt: str) -> str:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + 768 > 4096:
                raise RuntimeError("Context exceeds budget")
            return rendered

        blind_requests = []
        for pid, k in blind_tasks:
            problem = Problem(id=pid, domain="math", question=by_id[pid]["question"])
            blind_requests.append((render(paired_prompt(problem, "blind_resolve")), SamplingParams(
                temperature=0.7, top_p=1.0, top_k=-1, max_tokens=768,
                seed=seed("blind", args.checkpoint, pid, k))))

        def blind_row(index: int, result) -> dict:
            pid, k = blind_tasks[index]
            return {"problem_id": pid, "attempt": k, "index": index,
                    "request_seed": seed("blind", args.checkpoint, pid, k),
                    "output": result.text, "generated_tokens": len(result.token_ids),
                    "hit_token_cap": len(result.token_ids) >= 768}

        blind_done = generate_ordered(llm, blind_requests, len(blind_done), blind_audit,
                                      blind_row, f"{args.checkpoint}/blind",
                                      lora_request=lora_request, done=blind_done)
        b1 = {row["problem_id"]: row["output"] for row in blind_done if row["attempt"] == 1}
        judge_requests = []
        for row in rows:
            problem = Problem(id=row["id"], domain="math", question=row["question"])
            judge_requests.append((render(judge_prompt(problem, first[row["id"]], b1[row["id"]])),
                                   SamplingParams(temperature=0.0, max_tokens=768,
                                                  seed=seed("judge", args.checkpoint, row["id"]))))

        def judge_row(index: int, result) -> dict:
            pid = rows[index]["id"]
            return {"problem_id": pid, "index": index,
                    "request_seed": seed("judge", args.checkpoint, pid),
                    "output": result.text, "generated_tokens": len(result.token_ids),
                    "hit_token_cap": len(result.token_ids) >= 768}

        judge_done = generate_ordered(llm, judge_requests, len(judge_done), judge_audit,
                                      judge_row, f"{args.checkpoint}/judge",
                                      lora_request=lora_request, done=judge_done)
    summary = {"status": "complete"}
    for name, rows_out, audit in (("blind", blind_done, blind_audit),
                                  ("judge", judge_done, judge_audit)):
        path = out / f"{name}_answers.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows_out), encoding="utf-8", newline="\n")
        summary[f"{name}_rows"] = len(rows_out)
        summary[f"{name}_audit_sha256"] = sha256(audit)
        summary[f"{name}_answers_sha256"] = sha256(path)
    summary["settings"] = {"blind": blind_settings, "judge": judge_settings}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}))


if __name__ == "__main__":
    main()
