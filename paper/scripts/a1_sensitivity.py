"""A1 amendment: recompute the Phase 13 GSM8K numbers without the 42 problems that
appeared in the 2026-09-02 P0 evaluation log (outputs/p0_base_100_log.jsonl).

Those problems were evaluated (never trained on) before Phase 13 froze its
holdout; this script checks that removing them leaves every conclusion intact.
Output: paper/results/a1_sensitivity.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import cpu_checks as cc  # noqa: E402

OUT = ROOT / "paper/results/a1_sensitivity.json"


def main() -> None:
    checks = json.loads((ROOT / "paper/results/cpu_checks.json").read_text(encoding="utf-8"))
    dropped = set(checks["A1_contamination_audit"]["near_duplicates_jaccard_8gram_ge_0.5"])
    d = cc.load("gsm8k")
    keep_ids = [pid for pid in d["rows"] if pid not in dropped]
    d["rows"] = {pid: d["rows"][pid] for pid in keep_ids}
    for judge in d["judges"]:
        d["judges"][judge] = {t: v for t, v in d["judges"][judge].items() if t[0] in d["rows"]}
    tables = cc.judge_tables(d)
    res = {"dropped_problems": len(dropped), "remaining_problems": len(d["rows"]),
           "one_right_pairs": tables["_pairs"], "judges": {}}
    for j in ("base", "p12", "base_c", "p13b"):
        e = tables[j]
        cons_pairs = round(e["B2_coverage"] * tables["_pairs"])
        right = round(e["B2_consistent_accuracy"] * cons_pairs)
        res["judges"][j] = {"consistent_accuracy": e["B2_consistent_accuracy"], "coverage": e["B2_coverage"],
                            "p_one_sided": float(binomtest(right, cons_pairs, 0.5, alternative="greater").pvalue),
                            "tie_break": e["B2_tie_break"], "single_order": e["B4_single_order_accuracy"],
                            "skill_index": e["B1_skill_index"],
                            "invented_rate": (e["B1_position_table"]["right_is_A"]["picks_invented"] +
                                              e["B1_position_table"]["right_is_B"]["picks_invented"]) / 2}
    res["B3"] = cc.b3_paired(tables)
    res["B5_F3"] = cc.b5_f3(d)
    OUT.write_text(json.dumps(cc.strip(res), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(cc.strip(res), indent=1)[:4000])


if __name__ == "__main__":
    main()
