"""Phase 14 generation for one model x dataset x split (batched, resumable).

The protocol is Phase 13's, unchanged except for the model:
  samples              s1 = Phase 1 prompt, s2..s5 = Phase 7/8 blind prompt,
                       temperature 0.7, per-request seeds. Generated up to
                       GENERATION_CAP tokens; the primary view is the first
                       PRIMARY_CAP (768) tokens, which equals a 768-cap run.
  judge --judge NAME   greedy judge on pairs (s1, s2) and (s1, s3) whose
                       primary-view final answers differ, each in both orders
                       ("ab": s1 shown as Solution A; "ba": s1 as Solution B):
                         base    Phase 9 judge prompt
                         base_c  Phase 13 constrained-verdict prompt; also records
                                 the log-probabilities of "A" and "B" at the
                                 verdict token (top-k logprobs do not change the
                                 greedy output)
Prompts are imported from Phases 1, 7, 9 and 13, never retyped. Judges see the
primary (768-token) view of each solution, as in Phases 9-13.

Datasets "gsm8k13" and "svamp12" are the Phase 13 and Phase 12 holdouts (the
report's GSM8K 561 and SVAMP 960), solved and judged by every model itself with
the identical protocol (main split only). "gsm8k13_report" and "svamp12_report"
are regression-only: the original solver's attempts are read from the report's
runs and its base judgments imported, so the report's numbers can be checked;
base_c is re-run there. "bridge13" reuses Phase 13's original-solver attempts,
so other models re-judge exactly the Phase 13 pairs (report section 4.5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
import phase8.scripts.batched as batched  # noqa: E402
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402
from phase13.scripts.constrained import constrained_prompt  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase14.scripts.config import (BRIDGE, COMPARISON, DATASETS, GENERATION_CAP, REGRESSION_SETS, JUDGED, MAX_MODEL_LEN,  # noqa: E402
                                    MODELS, OUT, PHASE, PRIMARY_CAP, SAMPLES, SOURCES, TEMPERATURE,
                                    VERDICT_TOP_LOGPROBS)

JUDGES = {"base": judge_prompt, "base_c": constrained_prompt}
MAIN_SIZE = PHASE / "data/main_size.json"   # written by the pilot amendment, before any main run
_VERDICT_PREFIX = re.compile(r"Verdict:\s*Solution")


def read_jsonl(path: Path) -> list[dict]:
    """Newline-only split: str.splitlines() also breaks on U+2028 inside JSON strings."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


batched.read_jsonl = read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def problems(dataset: str, split: str) -> list[dict]:
    if dataset in (BRIDGE, *COMPARISON, *REGRESSION_SETS):  # earlier holdouts, hash-checked by their own loaders
        if split != "main":
            raise RuntimeError(f"{dataset} has no pilot")
        if dataset in ("svamp12", "svamp12_report"):
            from phase12.scripts.generate import holdout as holdout12
            return holdout12()
        from phase13.scripts.generate import holdout
        return holdout()
    report = json.loads((SOURCES / "report.json").read_text(encoding="utf-8"))["datasets"][dataset]
    if split == "pilot":
        path, expected = SOURCES / f"{dataset}_pilot.jsonl", report["pilot"]["sha256"]
    else:
        path, expected = SOURCES / f"{dataset}_main_pool.jsonl", report["main_pool"]["sha256"]
    if sha256(path) != expected:
        raise RuntimeError(f"Frozen Phase 14 sources changed: {path.name}")
    rows = read_jsonl(path)
    if split == "main":
        if not MAIN_SIZE.exists():
            raise RuntimeError("Main size is fixed only by the pilot amendment (data/main_size.json)")
        rows = rows[:json.loads(MAIN_SIZE.read_text(encoding="utf-8"))["main_size"][dataset]]
    return rows


def folder(model: str, dataset: str, split: str) -> Path:
    return OUT / split / model / dataset


def seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, ("phase14_v1", *parts))).encode()).hexdigest()[:8], 16)


def completed(base: Path, stage: str) -> dict[tuple, dict]:
    path = base / stage
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or sha256(path / "answers.jsonl") != summary["answers_sha256"]:
        raise RuntimeError(f"Stage {stage} incomplete or changed: {path}")
    return {tuple(r["task"]): r for r in read_jsonl(path / "answers.jsonl")}


REUSED_MODEL = "original_solver"
REUSED = {**REGRESSION_SETS, BRIDGE: "phase13"}


def earlier(dataset: str):
    """The Phase 13 or Phase 12 generator module that owns this dataset's original outputs."""
    if REUSED[dataset] == "phase12":
        import phase12.scripts.generate as module
    else:
        import phase13.scripts.generate as module
    return module


def reused(model: str, dataset: str) -> bool:
    """True when this cell reads the report's original outputs instead of generating new ones.

    Only the regression-only sets (Self_Correction's report outputs on GSM8K and
    SVAMP) and the bridge set do so. The main cells of every model, Self_Correction
    included, are generated fresh with the identical Phase 14 protocol.
    """
    return dataset == BRIDGE or (model == REUSED_MODEL and dataset in REGRESSION_SETS)


def load_samples(model: str, dataset: str, split: str) -> dict[tuple, dict]:
    """Attempts for judging (the reused ones carry no cap information)."""
    if not reused(model, dataset):
        return completed(folder(model, dataset, split), "samples")
    return {task: {"task": list(task), "output": text, "output_full": text,
                   "hit_primary_cap": None, "hit_token_cap": None}
            for task, text in earlier(dataset).completed("samples").items()}


def import_base_judge(model: str, dataset: str) -> None:
    """Copy the earlier phase's untrained base judgments (Phase 9 prompt) for a reused cell."""
    path = folder(model, dataset, "main") / "judge_base"
    if (path / "summary.json").exists():
        print(json.dumps({"model": model, "dataset": dataset, "stage": "judge_base", "status": "imported"}))
        return
    module = earlier(dataset)
    source = module.OUT / "judge_base"
    rows = [{"task": list(task), "output": text, "imported": True}
            for task, text in sorted(module.completed("judge_base").items())]
    path.mkdir(parents=True, exist_ok=True)
    answers = path / "answers.jsonl"
    answers.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                       encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(rows), "answers_sha256": sha256(answers),
               "imported_from": (str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)).replace("\\", "/"),
               "source_answers_sha256": sha256(source / "answers.jsonl")}
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"model": model, "dataset": dataset, "stage": "judge_base", "imported_rows": len(rows)}))


def judge_tasks(rows: list[dict], samples: dict) -> list[tuple[str, int, str]]:
    """(source, k, order) for k in JUDGED where s1 and sk (primary view) give different final answers."""
    tasks = []
    for row in rows:
        first = parse(samples[(row["id"], 1)]["output"])
        for k in JUDGED:
            if not same(first, parse(samples[(row["id"], k)]["output"])):
                tasks += [(row["id"], k, "ab"), (row["id"], k, "ba")]
    return tasks


def verdict_logprobs(result) -> dict | None:
    """Log-probabilities of "A" and "B" at the token after the last "Verdict: Solution"."""
    if not result.logprobs:
        return None
    offsets, text = [], ""
    for token_id, candidates in zip(result.token_ids, result.logprobs):
        offsets.append(len(text))
        text += candidates[token_id].decoded_token or ""
    matches = list(_VERDICT_PREFIX.finditer(text))
    if not matches:
        return None
    end = matches[-1].end()
    for position, start in enumerate(offsets):
        if start >= end:
            found = {}
            for candidate in result.logprobs[position].values():
                letter = (candidate.decoded_token or "").strip()
                if letter in ("A", "B"):
                    found[letter] = max(found.get(letter, float("-inf")), candidate.logprob)
            return {"position": position, "A": found.get("A"), "B": found.get("B")}
    return None


def run(model: str, base: Path, stage: str, tasks: list[tuple], prompts: list[str], temperature: float,
        max_tokens: int, seeds: list[int], with_logprobs: bool = False) -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if sys.version_info[:2] != (3, 10) or vllm.__version__ != "0.7.0":
        raise RuntimeError("Python 3.10 and vLLM 0.7.0 required")
    spec = MODELS[model]
    settings = {"schema_version": f"phase14_{stage}_v1", "model": spec["repo"], "revision": spec["revision"],
                "temperature": temperature, "top_p": 1.0, "top_k": -1, "max_new_tokens": max_tokens,
                "primary_cap": PRIMARY_CAP, "max_model_len": MAX_MODEL_LEN, "batch_size": BATCH_SIZE,
                "logprobs": VERDICT_TOP_LOGPROBS if with_logprobs else None,
                "gpu": torch.cuda.get_device_name(0), "gold_labels_opened": False}
    path = base / stage
    audit = path / "audit.jsonl"
    done = open_audit(audit, settings, [list(t) for t in tasks], lambda row: row["task"])
    if len(done) < len(tasks):
        tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        llm = LLM(model=spec["repo"], revision=spec["revision"], tokenizer=spec["repo"],
                  tokenizer_revision=spec["revision"], dtype="half", max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=False)
        requests = []
        for prompt, request_seed in zip(prompts, seeds):
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + max_tokens > MAX_MODEL_LEN:
                raise RuntimeError("Context exceeds budget")
            requests.append((rendered, SamplingParams(
                temperature=temperature, top_p=1.0, top_k=-1, max_tokens=max_tokens, seed=request_seed,
                logprobs=VERDICT_TOP_LOGPROBS if with_logprobs else None)))

        def make_row(index: int, result) -> dict:
            ids = list(result.token_ids)
            row = {"task": list(tasks[index]), "index": index, "request_seed": seeds[index],
                   "generated_tokens": len(ids), "hit_token_cap": len(ids) >= max_tokens}
            if max_tokens > PRIMARY_CAP:  # attempts: primary view = first 768 tokens
                row["output_full"] = result.text
                row["output"] = result.text if len(ids) < PRIMARY_CAP else \
                    tokenizer.decode(ids[:PRIMARY_CAP], skip_special_tokens=True)
                row["hit_primary_cap"] = len(ids) >= PRIMARY_CAP
            else:
                row["output"] = result.text
            if with_logprobs:
                row["verdict_logprobs"] = verdict_logprobs(result)
            return row

        done = generate_ordered(llm, requests, len(done), audit, make_row, stage, done=done)
    answers = path / "answers.jsonl"
    answers.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in done),
                       encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(done), "audit_sha256": sha256(audit),
               "answers_sha256": sha256(answers), "settings": settings}
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"model": model, "stage": stage, "rows": len(done)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--dataset", required=True, choices=(*DATASETS, *COMPARISON, *REGRESSION_SETS, BRIDGE))
    parser.add_argument("--split", required=True, choices=("pilot", "main"))
    parser.add_argument("--stage", required=True, choices=("samples", "judge"))
    parser.add_argument("--judge", choices=tuple(JUDGES))
    args = parser.parse_args()
    rows = problems(args.dataset, args.split)
    base = folder(args.model, args.dataset, args.split)
    tag = (args.model, args.dataset, args.split)
    if args.stage == "samples":
        if reused(args.model, args.dataset):  # report's original attempts; nothing to generate
            print(json.dumps({"model": args.model, "dataset": args.dataset, "stage": "samples", "status": "reused"}))
            return
        tasks, prompts, seeds = [], [], []
        for row in rows:
            problem = Problem(id=row["id"], domain="math", question=row["question"])
            for k in range(1, SAMPLES + 1):
                tasks.append((row["id"], k))
                prompts.append(build_prompt(problem) if k == 1 else paired_prompt(problem, "blind_resolve"))
                seeds.append(seed("samples", *tag, row["id"], k))
        run(args.model, base, "samples", tasks, prompts, TEMPERATURE, GENERATION_CAP, seeds)
        return
    if args.judge is None:
        raise SystemExit("--judge is required for --stage judge")
    if args.judge == "base" and reused(args.model, args.dataset) and args.model == REUSED_MODEL:
        import_base_judge(args.model, args.dataset)  # the report's own base judgments
        return
    make = JUDGES[args.judge]
    samples = load_samples(args.model, args.dataset, args.split)
    by_id = {row["id"]: row for row in rows}
    tasks = judge_tasks(rows, samples)
    prompts = []
    for pid, k, order in tasks:
        problem = Problem(id=pid, domain="math", question=by_id[pid]["question"])
        first, other = samples[(pid, 1)]["output"], samples[(pid, k)]["output"]
        prompts.append(make(problem, first, other) if order == "ab" else make(problem, other, first))
    seeds = [seed(args.judge, *tag, pid, k, order) for pid, k, order in tasks]
    run(args.model, base, f"judge_{args.judge}", tasks, prompts, 0.0, PRIMARY_CAP, seeds,
        with_logprobs=args.judge == "base_c")


if __name__ == "__main__":
    main()
