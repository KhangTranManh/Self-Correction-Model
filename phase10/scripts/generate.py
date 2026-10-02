"""Phase 10 vLLM generation stages (original solver; batched, resumable).

Stages:
  holdout          A0 (initial prompt) + B1..B4 (blind prompt) at temperature 0.7,
                   then the untrained greedy judge J_base on (A0, B1)
  train_samples    s1 (initial prompt) + s2..s4 (blind prompt) at temperature 0.7
  judge_candidates four temperature-0.7 judge samples per training pair
  holdout_trained  greedy judge J_trained with the trained LoRA on (A0, B1)
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
import phase8.scripts.batched as batched  # noqa: E402
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    """Split on "\\n" only: str.splitlines() also breaks on U+2028 inside JSON strings,
    which some GSM8K questions contain. Patched into the shared helper so that
    open_audit() resumes Phase 10 audits with the same safe reader."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


batched.read_jsonl = read_jsonl

SOURCES = ROOT / "phase10/data/sources_v1"
OUT = ROOT / "outputs/phase10_v1"
PAIRS = OUT / "train_pairs.jsonl"
ADAPTER = OUT / "judge_lora/final_adapter"
MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
MAX_TOKENS = 768


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rows(name: str) -> list[dict]:
    report = json.loads((SOURCES / "report.json").read_text(encoding="utf-8"))
    key = "train_pool" if name == "train_pool" else "holdout"
    path = SOURCES / f"{name}.jsonl"
    if sha256(path) != report[key]["sha256"]:
        raise RuntimeError(f"Frozen Phase 10 {name} changed")
    return read_jsonl(path)


def seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, ("phase10_v1", *parts))).encode()).hexdigest()[:8], 16)


def run(stage: str, tasks: list[tuple], prompts: list[str], temperatures: list[float],
        seeds: list[int], extra: dict, lora: Path | None = None) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if sys.version_info[:2] != (3, 10) or vllm.__version__ != "0.7.0":
        raise RuntimeError("Python 3.10 and vLLM 0.7.0 required")
    settings = {"schema_version": f"phase10_{stage}_v1", "model": MODEL, "revision": REVISION,
                "lora": str(lora) if lora else None,
                "lora_sha256": sha256(lora / "adapter_model.safetensors") if lora else None,
                "max_new_tokens": MAX_TOKENS, "max_model_len": 4096, "top_p": 1.0, "top_k": -1,
                "batch_size": BATCH_SIZE, "gpu": torch.cuda.get_device_name(0),
                "gold_labels_opened": stage in ("train_samples", "judge_candidates"), **extra}
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
        for prompt, temperature, request_seed in zip(prompts, temperatures, seeds):
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
                                lora_request=LoRARequest("phase10_judge", 1, str(lora)) if lora else None,
                                done=done)
    answers = folder / "answers.jsonl"
    answers.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in done),
                       encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(done), "audit_sha256": sha256(audit),
               "answers_sha256": sha256(answers), "settings": settings}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"stage": stage, "rows": len(done)}))


def completed(stage: str) -> dict[tuple, str]:
    folder = OUT / stage
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or sha256(folder / "answers.jsonl") != summary["answers_sha256"]:
        raise RuntimeError(f"Stage {stage} incomplete or changed")
    return {tuple(r["task"]): r["output"] for r in read_jsonl(folder / "answers.jsonl")}


def samples_stage(stage: str, sources: list[dict]) -> None:
    tasks, prompts, temps, seeds = [], [], [], []
    for row in sources:
        problem = Problem(id=row["id"], domain="math", question=row["question"])
        for k in range(1, 6 if stage == "holdout" else 5):
            tasks.append((row["id"], k))
            prompts.append(build_prompt(problem) if k == 1 else paired_prompt(problem, "blind_resolve"))
            temps.append(0.7)
            seeds.append(seed(stage, row["id"], k))
    run(f"{stage}_samples" if stage == "holdout" else stage, tasks, prompts, temps, seeds,
        {"sample_1_prompt": "phase1 build_prompt", "samples_2plus_prompt": "phase7 blind_resolve"})


def judge_stage(stage: str, sources: list[dict], samples: dict, lora: Path | None) -> None:
    tasks, prompts, temps, seeds = [], [], [], []
    for row in sources:
        problem = Problem(id=row["id"], domain="math", question=row["question"])
        tasks.append((row["id"],))
        prompts.append(judge_prompt(problem, samples[(row["id"], 1)], samples[(row["id"], 2)]))
        temps.append(0.0)
        seeds.append(seed(stage, row["id"]))
    run(stage, tasks, prompts, temps, seeds, {"judge": "greedy on (A0=sample 1, B1=sample 2)"}, lora)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("holdout", "train_samples", "judge_candidates", "holdout_trained"))
    args = parser.parse_args()
    if args.stage == "holdout":
        holdout = rows("holdout")
        samples_stage("holdout", holdout)
        judge_stage("holdout_judge_base", holdout, completed("holdout_samples"), None)
    elif args.stage == "train_samples":
        samples_stage("train_samples", rows("train_pool"))
    elif args.stage == "judge_candidates":
        pairs = read_jsonl(PAIRS)
        by_id = {row["id"]: row for row in rows("train_pool")}
        samples = completed("train_samples")
        tasks, prompts, temps, seeds = [], [], [], []
        for pair in pairs:
            problem = Problem(id=pair["problem_id"], domain="math",
                              question=by_id[pair["problem_id"]]["question"])
            text = judge_prompt(problem, samples[(pair["problem_id"], pair["a"])],
                                samples[(pair["problem_id"], pair["b"])])
            for j in range(1, 5):
                tasks.append((pair["pair_id"], j))
                prompts.append(text)
                temps.append(0.7)
                seeds.append(seed("judge_candidates", pair["pair_id"], j))
        run("judge_candidates", tasks, prompts, temps, seeds,
            {"pairs_sha256": sha256(PAIRS), "samples_per_pair": 4})
    else:
        judge_stage("holdout_judge_trained", rows("holdout"), completed("holdout_samples"), ADAPTER)


if __name__ == "__main__":
    main()
