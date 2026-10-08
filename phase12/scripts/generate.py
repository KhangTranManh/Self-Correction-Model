"""Phase 12 holdout generation on SVAMP (original solver; batched, resumable).

Stages:
  samples        s1 (Phase 1 prompt) + s2..s5 (blind prompt), temperature 0.7
  judge_base     greedy untrained judge on pairs (s1, s2) and (s1, s3) whose final
                 answers differ (gold-free rule), each in both orders:
                 "ab" shows s1 as Solution A; "ba" shows s1 as Solution B
  judge_dpo      the same tasks with the order-swapped DPO LoRA
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
import phase8.scripts.batched as batched  # noqa: E402
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402

SOURCES = ROOT / "phase12/data/sources_v1"
OUT = ROOT / "outputs/phase12_v1"
ADAPTER = OUT / "dpo_lora/final_adapter"
MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
MAX_TOKENS = 768
SAMPLES = 5
JUDGED = (2, 3)  # blind attempts paired with s1 for judging


def read_jsonl(path: Path) -> list[dict]:
    """Newline-only split: str.splitlines() also breaks on U+2028 inside JSON strings."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


batched.read_jsonl = read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def holdout() -> list[dict]:
    report = json.loads((SOURCES / "report.json").read_text(encoding="utf-8"))
    path = SOURCES / "holdout.jsonl"
    if sha256(path) != report["holdout"]["sha256"]:
        raise RuntimeError("Frozen Phase 12 holdout changed")
    return read_jsonl(path)


def seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, ("phase12_v1", *parts))).encode()).hexdigest()[:8], 16)


def completed(stage: str) -> dict[tuple, str]:
    folder = OUT / stage
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or sha256(folder / "answers.jsonl") != summary["answers_sha256"]:
        raise RuntimeError(f"Stage {stage} incomplete or changed")
    return {tuple(r["task"]): r["output"] for r in read_jsonl(folder / "answers.jsonl")}


def judge_tasks(samples: dict) -> list[tuple[str, int, str]]:
    """(source, k, order) for k in JUDGED where s1 and sk give different final answers."""
    tasks = []
    for row in holdout():
        first = parse(samples[(row["id"], 1)])
        for k in JUDGED:
            if not same(first, parse(samples[(row["id"], k)])):
                tasks += [(row["id"], k, "ab"), (row["id"], k, "ba")]
    return tasks


def run(stage: str, tasks: list[tuple], prompts: list[str], temperature: float,
        seeds: list[int], lora: Path | None = None) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if sys.version_info[:2] != (3, 10) or vllm.__version__ != "0.7.0":
        raise RuntimeError("Python 3.10 and vLLM 0.7.0 required")
    settings = {"schema_version": f"phase12_{stage}_v1", "model": MODEL, "revision": REVISION,
                "lora_sha256": sha256(lora / "adapter_model.safetensors") if lora else None,
                "temperature": temperature, "top_p": 1.0, "top_k": -1,
                "max_new_tokens": MAX_TOKENS, "max_model_len": 4096, "batch_size": BATCH_SIZE,
                "gpu": torch.cuda.get_device_name(0), "gold_labels_opened": False}
    folder = OUT / stage
    audit = folder / "audit.jsonl"
    done = open_audit(audit, settings, [list(t) for t in tasks], lambda row: row["task"])
    if len(done) < len(tasks):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        llm = LLM(model=MODEL, revision=REVISION, tokenizer=MODEL, tokenizer_revision=REVISION,
                  dtype="half", max_model_len=4096, gpu_memory_utilization=0.85,
                  enforce_eager=True, trust_remote_code=False,
                  enable_lora=bool(lora), max_loras=1, max_lora_rank=16)
        requests = []
        for prompt, request_seed in zip(prompts, seeds):
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + MAX_TOKENS > 4096:
                raise RuntimeError("Context exceeds budget")
            requests.append((rendered, SamplingParams(temperature=temperature, top_p=1.0, top_k=-1,
                                                      max_tokens=MAX_TOKENS, seed=request_seed)))

        def make_row(index: int, result) -> dict:
            return {"task": list(tasks[index]), "index": index, "request_seed": seeds[index],
                    "output": result.text, "generated_tokens": len(result.token_ids),
                    "hit_token_cap": len(result.token_ids) >= MAX_TOKENS}

        done = generate_ordered(llm, requests, len(done), audit, make_row, stage,
                                lora_request=LoRARequest("phase12_dpo", 1, str(lora)) if lora else None,
                                done=done)
    answers = folder / "answers.jsonl"
    answers.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in done),
                       encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(done), "audit_sha256": sha256(audit),
               "answers_sha256": sha256(answers), "settings": settings}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"stage": stage, "rows": len(done)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("samples", "judge_base", "judge_dpo"))
    args = parser.parse_args()
    rows = holdout()
    if args.stage == "samples":
        tasks, prompts, seeds = [], [], []
        for row in rows:
            problem = Problem(id=row["id"], domain="math", question=row["question"])
            for k in range(1, SAMPLES + 1):
                tasks.append((row["id"], k))
                prompts.append(build_prompt(problem) if k == 1 else paired_prompt(problem, "blind_resolve"))
                seeds.append(seed("samples", row["id"], k))
        run("samples", tasks, prompts, 0.7, seeds)
        return
    samples = completed("samples")
    by_id = {row["id"]: row for row in rows}
    tasks = judge_tasks(samples)
    prompts = []
    for pid, k, order in tasks:
        problem = Problem(id=pid, domain="math", question=by_id[pid]["question"])
        first, other = samples[(pid, 1)], samples[(pid, k)]
        prompts.append(judge_prompt(problem, first, other) if order == "ab"
                       else judge_prompt(problem, other, first))
    seeds = [seed(args.stage, pid, k, order) for pid, k, order in tasks]
    run(args.stage, tasks, prompts, 0.0, seeds, ADAPTER if args.stage == "judge_dpo" else None)


if __name__ == "__main__":
    main()
