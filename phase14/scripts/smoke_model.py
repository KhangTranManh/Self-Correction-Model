"""Technical smoke test for one Phase 14 model on the GPU host (no project data).

Uses six hand-written toy problems that belong to no dataset, so no pilot or
holdout row is touched. Checks, with the exact Phase 14 settings:
  1. the model loads in FP16 under vLLM 0.7.0 with max_model_len 4096;
  2. the first 768 tokens of a 1536-cap attempt equal a 768-cap attempt with
     the same seed (the basis of the primary 768-token view);
  3. tokenizer.decode(token_ids) reproduces vLLM's text, so the decoded
     768-token prefix is what a 768-cap run would print;
  4. answers are extractable (lenient and strict) and mostly correct, i.e. no
     FP16 overflow garbage;
  5. the constrained judge emits a verdict and its A/B log-probabilities are
     found (both orders).
The toy attempts and judgments are saved (outputs/phase14_v1/smoke/<model>_toy.json)
for scripts/smoke_hidden.py, which re-reads them with Transformers.
A model that fails is replaced by its fallback in config.py before any pilot
run, and the swap is recorded in RUN_STATUS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
from phase13.scripts.constrained import constrained_prompt  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase14.scripts.config import (GENERATION_CAP, MAX_MODEL_LEN, MODELS, OUT, PRIMARY_CAP,  # noqa: E402
                                    VERDICT_TOP_LOGPROBS, tokenizer_source)
from phase14.scripts.generate import decode_matches, primary_view, verdict_logprobs  # noqa: E402

TOY = [("A box holds 12 pencils. Mai buys 7 boxes and gives away 19 pencils. How many pencils does she keep?", "65"),
       ("A train travels 84 km in 1.5 hours. At the same speed, how many km does it travel in 4 hours?", "224"),
       ("What is 3/4 of 2/9? Give the answer as a fraction.", "1/6"),
       ("A rectangle has perimeter 46 cm and width 8 cm. What is its area in square cm?", "120"),
       ("Tom is 3 times as old as Ann. In 6 years their ages will add up to 52. How old is Ann now?", "10"),
       ("How many positive divisors does 180 have?", "18")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    args = parser.parse_args()
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if sys.version_info[:2] != (3, 10) or vllm.__version__ != "0.7.0":
        raise RuntimeError("Python 3.10 and vLLM 0.7.0 required")
    spec = MODELS[args.model]
    tok_repo, tok_revision = tokenizer_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(tok_repo, revision=tok_revision)
    llm = LLM(model=spec["repo"], revision=spec["revision"], tokenizer=tok_repo,
              tokenizer_revision=tok_revision, dtype="half", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=False)
    render = lambda text: tokenizer.apply_chat_template([{"role": "user", "content": text}],  # noqa: E731
                                                        tokenize=False, add_generation_prompt=True)
    problems = [Problem(id=f"toy{i}", domain="math", question=q) for i, (q, _) in enumerate(TOY)]
    prompts = [render(build_prompt(p)) for p in problems] + [render(paired_prompt(p, "blind_resolve")) for p in problems]
    params = lambda cap: [SamplingParams(temperature=0.7, top_p=1.0, top_k=-1, max_tokens=cap, seed=1000 + i)  # noqa: E731
                          for i in range(len(prompts))]
    long = llm.generate(prompts, params(GENERATION_CAP), use_tqdm=False)
    short = llm.generate(prompts, params(PRIMARY_CAP), use_tqdm=False)
    prefix_equal = [list(s.outputs[0].token_ids) == list(l.outputs[0].token_ids)[:PRIMARY_CAP] for s, l in zip(short, long)]
    decode_equal = [decode_matches(tokenizer, list(l.outputs[0].token_ids), l.outputs[0].text) for l in long]
    view_equal = [primary_view(tokenizer, list(l.outputs[0].token_ids), l.outputs[0].text) == s.outputs[0].text
                  for s, l in zip(short, long)]  # the stored primary view equals a 768-cap run's text
    answers = TOY + TOY
    texts = [l.outputs[0].text for l in long]
    lenient_ok = [same(parse(t), parse(f"Đáp số: {a}")) for t, (_, a) in zip(texts, answers)]
    strict_found = [parse(t, "strict") is not None for t in texts]
    views = [s.outputs[0].text for s in short]  # primary (768-token) views, as judges see them
    n = len(TOY)
    judge_prompts = ([render(constrained_prompt(problems[i], views[i], views[i + n])) for i in range(n)]
                     + [render(constrained_prompt(problems[i], views[i + n], views[i])) for i in range(n)])
    judged = llm.generate(judge_prompts, [SamplingParams(temperature=0.0, max_tokens=PRIMARY_CAP, seed=7,
                                                         logprobs=VERDICT_TOP_LOGPROBS)] * len(judge_prompts),
                          use_tqdm=False)
    verdicts = [verdict_logprobs(j.outputs[0]) for j in judged]
    toy = {"problems": [{"id": p.id, "question": p.question} for p in problems],
           "attempts": {p.id: {"1": views[i], "2": views[i + n]} for i, p in enumerate(problems)},
           "judgments": {f"{p.id}|2|{order}": {"output": judged[i + off].outputs[0].text,
                                                "verdict_logprobs": verdicts[i + off]}
                         for i, p in enumerate(problems) for order, off in (("ab", 0), ("ba", n))}}
    report = {"model": spec["repo"], "revision": spec["revision"], "gpu": torch.cuda.get_device_name(0),
              "prefix_768_equal": prefix_equal, "decode_equal": decode_equal,
              "primary_view_equals_768_run_text": view_equal,
              "lenient_correct": sum(lenient_ok), "strict_answer_found": sum(strict_found), "attempts": len(texts),
              "verdict_logprobs_found": sum(v is not None and v["A"] is not None and v["B"] is not None for v in verdicts),
              "judges": len(verdicts), "sample_output": texts[0][:1500], "sample_judge": judged[0].outputs[0].text[-600:],
              "sample_rendered_prompt": prompts[0]}
    report["pass"] = bool(all(prefix_equal) and all(decode_equal) and all(view_equal) and report["lenient_correct"] >= 8
                          and report["verdict_logprobs_found"] >= 8)
    out = OUT / "smoke" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "smoke" / f"{args.model}_toy.json").write_text(json.dumps(toy, ensure_ascii=False, indent=1) + "\n",
                                                          encoding="utf-8", newline="\n")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("sample")}, indent=2))


if __name__ == "__main__":
    main()
