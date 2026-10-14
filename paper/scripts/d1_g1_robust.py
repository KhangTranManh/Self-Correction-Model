"""D1 robustness and G1 (CPU): Instruct judge with a 1536-token cap (GSM8K), on SVAMP,
and an untrained OLMo-2-7B-Instruct judge (second model family, GSM8K).

Same tables as cpu_checks (B1/B2/B4) and the D1 test (net position-2 preference
of the variant minus the Self_Correction base judge, problem-clustered bootstrap).
GSM8K reference: the 3090 base run (d0); SVAMP reference: the V100 Phase 12 base
judge (D0 showed 95% pick agreement between the two GPUs).
Output: paper/results/d1_g1_robust.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import cpu_checks as cc  # noqa: E402
from paper.scripts.d_e_analysis import RUNS, net_pos2  # noqa: E402

OUT = ROOT / "paper/results/d1_g1_robust.json"
KEEP = ("B1_skill_index", "B1_skill_index_ci", "B1_position2_rate_among_AB", "B2_tie_break",
        "B2_consistent_accuracy", "B2_coverage", "B4_single_order_accuracy")


def add(d: dict, name: str) -> bool:
    path = RUNS / f"judge_{name}.jsonl"
    if not path.exists():
        return False
    recs = cc.read_jsonl(path)
    d["judges"][name] = {tuple(r["task"]): r["output"] for r in recs}
    d["meta"][name] = {tuple(r["task"]): r for r in recs}
    return True


def section(d: dict, ref: str, variants: list[str]) -> dict:
    tables = cc.judge_tables(d)
    pairs = cc.one_right_pairs(d)
    out = {"one_right_pairs": len(pairs), "tables": {}, "tests": {}}
    for j in (ref, *variants):
        out["tables"][j] = {k: tables[j][k] for k in KEEP}
        m = d["meta"].get(j, {})
        if m:
            out["tables"][j]["token_cap_rate"] = float(np.mean([r.get("hit_token_cap", False) for r in m.values()]))
    base, groups = net_pos2(d, ref, pairs)
    for i, j in enumerate(variants):
        v, _ = net_pos2(d, j, pairs)
        ci, p = cc.cluster_boot_p(v - base, groups, 20261013 + i)
        tie = tables[j]["_tie"] - tables[ref]["_tie"]
        tci, tp = cc.cluster_boot_p(tie, tables[ref]["_tg"], 20261023 + i)
        out["tests"][f"{j}_minus_{ref}"] = {
            "net_position2": {"variant": float(v.mean()), "reference": float(base.mean()),
                              "difference": float((v - base).mean()), "ci": ci, "p_boot": p},
            "tie_break_difference": {"estimate": float(tie.mean()), "ci": tci, "p_boot": tp}}
    return out


def main() -> None:
    res = {"schema_version": "paper_d1_g1_v1", "post_hoc": True}
    gsm = cc.load("gsm8k")
    names = [n for n in ("d0_base", "d1_instruct", "d1_instruct_long", "g1_olmo") if add(gsm, n)]
    res["gsm8k"] = section(gsm, "d0_base", [n for n in names if n != "d0_base"])
    svamp = cc.load("svamp")
    if add(svamp, "d1_instruct_svamp"):
        res["svamp"] = section(svamp, "base", ["d1_instruct_svamp"])
    OUT.write_text(json.dumps(cc.strip(res), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(cc.strip(res), indent=1))


if __name__ == "__main__":
    main()
