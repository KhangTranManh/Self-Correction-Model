"""T3 and T5: re-judge the T1 pairs (Instruct's own conflicting solutions).

  t3  Qwen2.5-7B-Instruct, 1536-token cap (T1 hit 768 tokens on 20% of judgments)
  t5  Self_Correction_v1 judging Instruct's solutions (another model's work), 768
      tokens as in d0
Same tasks (pid, k, order) as judge_t1_parent_self.jsonl, Phase 9 judge prompt,
greedy. Output: outputs/paper/judge_t3_parent_self_long.jsonl,
outputs/paper/judge_t5_sc_on_parent.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402
from phase13.scripts import generate as g13  # noqa: E402

OUT = ROOT / "outputs/paper"
RUNS = {"t3": ("Qwen/Qwen2.5-7B-Instruct", None, 1536, "judge_t3_parent_self_long.jsonl"),
        "t5": ("Kxck/Self_Correction_v1", "6437f947999168a0ce2a98a86e4252fc77160a33", 768,
               "judge_t5_sc_on_parent.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=tuple(RUNS))
    args = parser.parse_args()
    model, revision, cap, name = RUNS[args.run]
    if (OUT / name).exists():
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = {r["id"]: r for r in g13.holdout()}
    samples = {tuple(r["task"]): r["output"] for r in g13.read_jsonl(OUT / "t1_samples.jsonl")}
    tasks = [r["task"] for r in g13.read_jsonl(OUT / "judge_t1_parent_self.jsonl")]
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    llm = LLM(model=model, revision=revision, tokenizer=model, tokenizer_revision=revision, dtype="half",
              max_model_len=6144, gpu_memory_utilization=0.85, enforce_eager=True)
    prompts = []
    for pid, k, order in tasks:
        problem = Problem(id=pid, domain="math", question=rows[pid]["question"])
        first, other = samples[(pid, 1)], samples[(pid, k)]
        p = judge_prompt(problem, first, other) if order == "ab" else judge_prompt(problem, other, first)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True))
    results = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=cap, seed=20261015))
    tmp = OUT / (name + ".tmp")
    tmp.write_text("".join(json.dumps({"task": t, "output": r.outputs[0].text,
                                       "generated_tokens": len(r.outputs[0].token_ids),
                                       "hit_token_cap": len(r.outputs[0].token_ids) >= cap},
                                      ensure_ascii=False, sort_keys=True) + "\n"
                           for t, r in zip(tasks, results)), encoding="utf-8", newline="\n")
    tmp.replace(OUT / name)
    print(json.dumps({"run": args.run, "rows": len(tasks)}), flush=True)


if __name__ == "__main__":
    main()
