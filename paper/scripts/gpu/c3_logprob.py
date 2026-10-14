"""C3 (part 2): likelihood baselines for choosing between two conflicting solutions.

For every judged one-right pair (s1, sk), k in (2, 3):
  (a) logprob-mean / logprob-sum: pick the solution with the higher mean / summed
      token log-probability under the original solver, given the Phase 1 math
      prompt (the same prompt for both, so only the solution text differs);
  (b) greedy-match: pick the solution whose final answer matches a fresh greedy
      solve (c3_greedy.py); 0.5 when neither or both match.
Reported next to the judge's single-order and both-orders results.
Output: outputs/paper/c3_likelihood.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase13.layer.probe_analysis import load  # noqa: E402

MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
OUT = ROOT / "outputs/paper"


def boot(values: np.ndarray, groups: np.ndarray) -> list[float]:
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(20261011)
    means = [values[np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])].mean() for _ in range(5000)]
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16,
                                                 device_map={"": 0}, attn_implementation="sdpa").eval()
    report = {"schema_version": "paper_c3_v1", "post_hoc": True}
    for name in ("phase12", "phase13"):
        g, samples, rows, correct = load(name)
        greedy = json.loads((OUT / f"c3_greedy_{name}.json").read_text(encoding="utf-8"))
        pairs = sorted({(pid, k) for (pid, k, _) in g.completed("judge_base")})
        pairs = [(pid, k) for pid, k in pairs if correct[(pid, 1)] != correct[(pid, k)]]
        need = sorted({(pid, 1) for pid, _ in pairs} | set(pairs))
        lp = {}
        with torch.inference_mode():
            for pid, k in need:
                head = tok.apply_chat_template([{"role": "user", "content": build_prompt(
                    Problem(id=pid, domain="math", question=rows[pid]["question"]))}], tokenize=False,
                    add_generation_prompt=True)
                ids_head = tok(head, add_special_tokens=False)["input_ids"]
                ids_sol = tok(samples[(pid, k)], add_special_tokens=False)["input_ids"]
                ids = torch.tensor([ids_head + ids_sol], device="cuda")
                logits = model(input_ids=ids).logits[0, len(ids_head) - 1:-1].float()
                token_lp = torch.log_softmax(logits, -1).gather(1, ids[0, len(ids_head):, None]).squeeze(1)
                lp[(pid, k)] = (float(token_lp.mean()), float(token_lp.sum()))
        hits = {"logprob_mean": [], "logprob_sum": [], "greedy_match": []}
        groups = []
        greedy_ok = 0
        for pid, k in pairs:
            first_right = correct[(pid, 1)]
            groups.append(pid)
            for j, key in enumerate(("logprob_mean", "logprob_sum")):
                hits[key].append(float((lp[(pid, 1)][j] > lp[(pid, k)][j]) == first_right))
            gv = parse(greedy[pid])
            m1, mk = same(gv, parse(samples[(pid, 1)])), same(gv, parse(samples[(pid, k)]))
            hits["greedy_match"].append(0.5 if m1 == mk else float(m1 == first_right))
            greedy_ok += m1 != mk
        groups = np.array(groups)
        report[name] = {"one_right_pairs": len(pairs),
                        "greedy_decisive_share": greedy_ok / len(pairs),
                        **{key: {"accuracy": float(np.mean(v)), "ci": boot(np.array(v), groups)}
                           for key, v in hits.items()}}
        print(name, json.dumps({k: v for k, v in report[name].items()}), flush=True)
    (OUT / "c3_likelihood.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
