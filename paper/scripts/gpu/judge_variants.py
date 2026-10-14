"""D0-D3 and E1: judge-prompt and judge-model variants on Phase 13 GSM8K (vLLM, greedy).

Same judged pairs (s1, sk), k in (2, 3), both orders, as Phase 13 judge_base.
Variants:
  d0_base      Self_Correction base, Phase 9 prompt (3090 control for V100 judge_base)
  d1_instruct  Qwen2.5-7B-Instruct, Phase 9 prompt (does the untrained parent show the bias?)
  d2_relabel   base, labels reversed: first shown solution is named "Solution B",
               second "Solution A" (does the preference follow position or label?)
  d3_restate   base, Phase 9 prompt + the problem restated after both solutions (recency)
  e1_four      base, four candidates s1..s4 as Solutions A-D on problems whose four
               answers are not all equal, in the 4 cyclic rotations (each candidate
               appears once in each position)
Picks are read by matching the judge's final answer to a shown solution, so they do
not depend on the label names. Output: outputs/paper/judge_<variant>.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.collect_checkpoint import JUDGE_HEADER, judge_prompt  # noqa: E402
from phase13.scripts import generate as g13  # noqa: E402

MODELS = {"base": ("Kxck/Self_Correction_v1", "6437f947999168a0ce2a98a86e4252fc77160a33"),
          "instruct": ("Qwen/Qwen2.5-7B-Instruct", None),
          "olmo": ("allenai/OLMo-2-1124-7B-Instruct", None)}
CONTEXT = {"olmo": 4096}  # OLMo-2 has a 4096-token context
VARIANTS = {"d0_base": "base", "d1_instruct": "instruct", "d2_relabel": "base",
            "d3_restate": "base", "e1_four": "base",
            "d1_instruct_long": "instruct", "d1_instruct_svamp": "instruct", "g1_olmo": "olmo"}
# d1_instruct_long: GSM8K with a 1536-token cap (d1_instruct hit 768 on 20.6%);
# d1_instruct_svamp: Phase 12 SVAMP pairs (vs the V100 Phase 12 judge_base), 1536 tokens.
MAX_TOKENS = {"d1_instruct_long": 1536, "d1_instruct_svamp": 1536, "g1_olmo": 1024}
FOUR_HEADER = (
    "\n\nFour independent solutions to this problem are shown below. Any of them "
    "may be wrong, and their final answers may differ. Check each solution step by "
    "step, then write your own complete, correct solution."
)
LETTERS = "ABCD"
OUT = ROOT / "outputs/paper"


def relabel_prompt(problem: Problem, shown_first: str, shown_second: str) -> str:
    return (build_prompt(problem) + JUDGE_HEADER + "\n\nSolution B:\n" + shown_first
            + "\n\nSolution A:\n" + shown_second)


def restate_prompt(problem: Problem, shown_first: str, shown_second: str) -> str:
    return judge_prompt(problem, shown_first, shown_second) + "\n\nThe problem again:\n" + problem.question


def four_prompt(problem: Problem, shown: list[str]) -> str:
    body = "".join(f"\n\nSolution {LETTERS[i]}:\n{text}" for i, text in enumerate(shown))
    return build_prompt(problem) + FOUR_HEADER + body


def tasks_for(variant: str):
    g = g13
    if variant.endswith("_svamp"):
        from phase12.scripts import generate as g
    samples = g.completed("samples")
    rows = {row["id"]: row for row in g.holdout()}
    out = []
    if variant == "e1_four":
        for pid in rows:
            values = [parse(samples[(pid, k)]) for k in range(1, 5)]
            if all(same(values[0], v) for v in values[1:]):
                continue
            problem = Problem(id=pid, domain="math", question=rows[pid]["question"])
            for r in range(4):
                order = [1 + (r + i) % 4 for i in range(4)]  # rotation r: s_{order[i]} shown at position i
                out.append(([pid, r, "".join(map(str, order))],
                            four_prompt(problem, [samples[(pid, k)] for k in order])))
        return out
    make = {"d2_relabel": relabel_prompt, "d3_restate": restate_prompt}.get(variant, judge_prompt)
    for pid, k, order in sorted(g.completed("judge_base")):
        problem = Problem(id=pid, domain="math", question=rows[pid]["question"])
        first, other = samples[(pid, 1)], samples[(pid, k)]
        out.append(([pid, k, order], make(problem, first, other) if order == "ab" else make(problem, other, first)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    args = parser.parse_args()
    todo = [v for v, m in VARIANTS.items() if m == args.model and not (OUT / f"judge_{v}.jsonl").exists()]
    if not todo:
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    name, revision = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    llm = LLM(model=name, revision=revision, tokenizer=name, tokenizer_revision=revision, dtype="half",
              max_model_len=CONTEXT.get(args.model, 6144), gpu_memory_utilization=0.85, enforce_eager=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in todo:
        cap = MAX_TOKENS.get(variant, g13.MAX_TOKENS)
        items = tasks_for(variant)
        rendered = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                            add_generation_prompt=True) for _, p in items]
        lengths = [len(tok.encode(r, add_special_tokens=False)) for r in rendered]
        keep = [i for i, n in enumerate(lengths) if n + cap <= CONTEXT.get(args.model, 6144)]
        params = SamplingParams(temperature=0.0, max_tokens=cap, seed=20261011)
        results = llm.generate([rendered[i] for i in keep], params)
        lines = []
        for i, res in zip(keep, results):
            o = res.outputs[0]
            lines.append(json.dumps({"task": items[i][0], "output": o.text, "prompt_tokens": lengths[i],
                                     "generated_tokens": len(o.token_ids),
                                     "hit_token_cap": len(o.token_ids) >= cap},
                                    ensure_ascii=False, sort_keys=True))
        tmp = OUT / f"judge_{variant}.jsonl.tmp"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(OUT / f"judge_{variant}.jsonl")
        print(json.dumps({"variant": variant, "rows": len(lines), "skipped_too_long": len(items) - len(keep)}),
              flush=True)


if __name__ == "__main__":
    main()
