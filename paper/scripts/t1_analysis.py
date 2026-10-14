"""T1 analysis (CPU): Qwen2.5-7B-Instruct judging its own conflicting solutions.

Table-D metrics on Instruct's own one-right pairs, next to d0 (Self_Correction on
its own solutions, 3090) and d1 (Instruct on Self_Correction's solutions).
Different pair sets, so differences use independent problem-clustered bootstraps
per arm. Primary metric: tie-break. Rule (fixed before the result): >= ~0.70
supports "SFT reduced the skill"; ~0.60 means the D1 gap is mostly the parent
judging another model's work.
Output: paper/results/t1_parent_self.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import cpu_checks as cc  # noqa: E402
from paper.scripts.d1_g1_robust import KEEP, add  # noqa: E402
from paper.scripts.t2_parent_b5 import t1_data  # noqa: E402

OUT = ROOT / "paper/results/t1_parent_self.json"


def boot_means(tie: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    return np.array([tie[np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])].mean() for _ in range(10000)])


def main() -> None:
    t1 = t1_data()
    tt = cc.judge_tables(t1)
    gsm = cc.load("gsm8k")
    for n in ("d0_base", "d1_instruct"):
        add(gsm, n)
    tg = cc.judge_tables(gsm)
    arms = {"t1_parent_self": tt["t1_parent_self"], "d0_base": tg["d0_base"], "d1_instruct": tg["d1_instruct"]}
    res = {"schema_version": "paper_t1_v1", "post_hoc": False,
           "t1_one_right_pairs": tt["_pairs"], "t1_problems_with_one_right_pair": tt["_problems"],
           "t1_solver_accuracy_s1": float(np.mean([t1["correct"][(p, 1)] for p in t1["rows"]])),
           "t1_token_cap_rate_judge": float(np.mean([r["hit_token_cap"] for r in t1["meta"]["t1_parent_self"].values()])),
           "tables": {k: {m: v[m] for m in KEEP} for k, v in arms.items()}, "differences": {}}
    boots = {k: boot_means(v["_tie"], v["_tg"], 20261070 + i) for i, (k, v) in enumerate(arms.items())}
    for ref in ("d0_base", "d1_instruct"):
        diff = boots["t1_parent_self"] - boots[ref]
        res["differences"][f"t1_minus_{ref}"] = {
            "tie_break_difference": float(arms["t1_parent_self"]["B2_tie_break"] - arms[ref]["B2_tie_break"]),
            "ci": [float(x) for x in np.quantile(diff, [0.025, 0.975])]}
    tie = arms["t1_parent_self"]["B2_tie_break"]
    res["rule_reading"] = ("supports 'SFT reduced the skill'" if tie >= 0.70 else
                           "gap mostly from judging another model's work" if tie <= 0.62 else "in between")
    OUT.write_text(json.dumps(cc.strip(res), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(cc.strip(res), indent=1))


if __name__ == "__main__":
    main()
