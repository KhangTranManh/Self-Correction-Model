"""T1: the parent model judges its OWN conflicting solutions (vLLM, one model load).

Qwen2.5-7B-Instruct, Phase 13 GSM8K holdout (561 problems), exactly the Phase 13
recipe with the model swapped:
  samples  s1 (Phase 1 prompt) + s2, s3 (blind_resolve prompt), temperature 0.7,
           768 tokens, Phase 13 seeds with a "t1" tag
  judge    pairs (s1, s2), (s1, s3) whose final answers differ, both orders, Phase 9
           judge prompt, greedy, 768 tokens (same cap as table D)
Outputs: outputs/paper/t1_samples.jsonl, outputs/paper/judge_t1_parent_self.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402
from phase13.scripts import generate as g13  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = ROOT / "outputs/paper"
SAMPLES = 3


def write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = g13.holdout()
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, tokenizer=MODEL, dtype="half", max_model_len=6144,
              gpu_memory_utilization=0.85, enforce_eager=True)
    render = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True)
    OUT.mkdir(parents=True, exist_ok=True)
    spath = OUT / "t1_samples.jsonl"
    if not spath.exists():
        tasks, prompts, params = [], [], []
        for row in rows:
            problem = Problem(id=row["id"], domain="math", question=row["question"])
            for k in range(1, SAMPLES + 1):
                tasks.append([row["id"], k])
                prompts.append(render(build_prompt(problem) if k == 1 else paired_prompt(problem, "blind_resolve")))
                params.append(SamplingParams(temperature=0.7, top_p=1.0, top_k=-1, max_tokens=g13.MAX_TOKENS,
                                             seed=g13.seed("t1_samples", row["id"], k)))
        results = llm.generate(prompts, params)
        write(spath, [{"task": t, "output": r.outputs[0].text, "generated_tokens": len(r.outputs[0].token_ids),
                       "hit_token_cap": len(r.outputs[0].token_ids) >= g13.MAX_TOKENS}
                      for t, r in zip(tasks, results)])
        print(json.dumps({"stage": "t1_samples", "rows": len(tasks)}), flush=True)
    samples = {tuple(r["task"]): r["output"] for r in g13.read_jsonl(spath)}
    jpath = OUT / "judge_t1_parent_self.jsonl"
    if jpath.exists():
        return
    by_id = {row["id"]: row for row in rows}
    tasks, prompts = [], []
    for row in rows:
        pid = row["id"]
        problem = Problem(id=pid, domain="math", question=row["question"])
        first = samples[(pid, 1)]
        for k in (2, 3):
            other = samples[(pid, k)]
            if same(parse(first), parse(other)):
                continue
            for order in ("ab", "ba"):
                tasks.append([pid, k, order])
                prompts.append(render(judge_prompt(problem, first, other) if order == "ab"
                                      else judge_prompt(problem, other, first)))
    params = SamplingParams(temperature=0.0, max_tokens=g13.MAX_TOKENS, seed=20261014)
    results = llm.generate(prompts, params)
    write(jpath, [{"task": t, "output": r.outputs[0].text, "generated_tokens": len(r.outputs[0].token_ids),
                   "hit_token_cap": len(r.outputs[0].token_ids) >= g13.MAX_TOKENS}
                  for t, r in zip(tasks, results)])
    print(json.dumps({"stage": "judge_t1_parent_self", "rows": len(tasks), "problems": len(by_id)}), flush=True)


if __name__ == "__main__":
    main()
