"""T6: CI for the first-position pick rate (share of A/B picks that go to the first
shown solution) on one-right pairs, per judge arm, strict and lenient scoring.

Problem-clustered bootstrap (10,000 resamples); two-sided bootstrap p for != 0.5.
Output: paper/results/t6_position_ci.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import cpu_checks as cc  # noqa: E402
from paper.scripts import t3_t5_analysis as t35  # noqa: E402

OUT = ROOT / "paper/results/t6_position_ci.json"


def first_rate(d: dict, judge: str, seed: int) -> dict:
    vals, groups = [], []
    for pid, k in cc.one_right_pairs(d):
        pk = cc.picks_for(d, judge, pid, k)
        for order in ("ab", "ba"):
            raw = pk[f"{order}_raw"]
            if raw in ("A", "B"):
                vals.append(float(raw == "A")); groups.append(pid)
    vals, groups = np.array(vals), np.array(groups)
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    means = np.array([vals[np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])].mean()
                      for _ in range(10000)])
    return {"ab_picks": int(len(vals)), "first_position_rate": float(vals.mean()),
            "ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "p_vs_half": float(min(1.0, 2 * min((means <= 0.5).mean(), (means >= 0.5).mean())))}


def main() -> None:
    res = {"schema_version": "paper_t6_v1", "post_hoc": True}
    for scoring in ("strict", "lenient"):
        t35.set_scoring(scoring == "lenient")
        out = {}
        for i, solutions in enumerate(("self_correction", "instruct")):
            d = t35.dataset(solutions, scoring == "lenient")
            for j, arm in enumerate(a for a in d["judges"] if a != "base"):
                out[arm] = first_rate(d, arm, 20261090 + 10 * i + j)
        res[scoring] = out
    t35.set_scoring(False)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    for s, v in res.items():
        if isinstance(v, dict):
            for arm, e in v.items():
                print(s, arm, e["ab_picks"], round(e["first_position_rate"], 3), [round(x, 3) for x in e["ci"]], e["p_vs_half"])


if __name__ == "__main__":
    main()
