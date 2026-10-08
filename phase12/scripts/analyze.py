"""Open Phase 12 holdout labels once and score both-orders judging.

Pairs: (s1, sk) for k in (2, 3) with different final answers, each judged in
order "ab" (s1 shown as Solution A) and "ba" (s1 shown as Solution B).
A one-right pair has exactly one verifier-correct solution.

Both-orders verdict: the solution whose final answer BOTH orderings' judgments
match ("consistent"); otherwise inconsistent (no verdict).

Endpoints (see docs/PREREGISTRATION.md):
  P1  DPO judge, consistent verdicts on one-right pairs: accuracy > 50%,
      one-sided exact binomial p < 0.05 after Holm (family: P1 + Secondary),
      and coverage (consistent share of one-right pairs) >= 30%.
      Baseline: the untrained judge on the same pairs (reported).
  P2  self_check_dpo (judge verdict if consistent, else maj3) vs vote@k with
      k the smallest odd k in {3, 5} >= self_check_dpo's mean model calls;
      passes if the lower 95% bootstrap bound of the difference > -2 points.
  Secondary  DPO judge single-order accuracy on one-right pairs (both
      orderings counted) > 50%, one-sided binomial, Holm with P1.
  Diagnostics  position-2 pick rate among A/B picks (target 40-60%);
      invented-answer rate (target < 5%). Tie-break score (inconsistent = 0.5)
      is reported only.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
from phase9.scripts.analyze import ci, holm, paired_p  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase11.scripts.analyze import clustered_ci  # noqa: E402
from phase12.scripts.generate import JUDGED, OUT, completed, holdout  # noqa: E402

REPORT = OUT / "analysis"
COVERAGE_MIN = 0.30
NONINFERIORITY = -0.02


def verdict(judge: dict, pid: str, k: int, first, other) -> str:
    ab, ba = parse(judge[(pid, k, "ab")]), parse(judge[(pid, k, "ba")])
    if same(ab, first) and same(ba, first):
        return "first"
    if same(ab, other) and same(ba, other):
        return "other"
    return "inconsistent"


def single_pick(text: str, shown_a, shown_b) -> str:
    value = parse(text)
    return "A" if same(value, shown_a) else "B" if same(value, shown_b) else "invented"


def one_sided(right: int, wrong: int) -> float:
    return float(binomtest(right, right + wrong, 0.5, alternative="greater").pvalue) if right + wrong else 1.0


def main() -> None:
    if REPORT.exists():
        raise RuntimeError("Refusing a second protected opening")
    samples = completed("samples")
    judges = {"base": completed("judge_base"), "dpo": completed("judge_dpo")}
    rows = holdout()
    verifier = MathVerifier()  # First read of Phase 12 holdout gold.
    strategies = ("keep", "maj3", "maj5", "agree_gated", "self_check_base", "self_check_dpo")
    acc = {n: [] for n in strategies}
    calls = Counter()
    verdicts = {j: Counter() for j in judges}
    tie_scores = {j: [] for j in judges}
    single_ok = {j: [] for j in judges}      # one entry per single-order judgment
    single_groups = []
    picks = {j: Counter() for j in judges}   # (right_position, pick)
    groups = []
    for row in rows:
        pid = row["id"]
        gold = Problem(id=pid, domain="math", question=row["question"], reference_answer=row["reference_answer"])
        ok = lambda text: bool(verifier.verify(gold, text).passed)
        s = {k: samples[(pid, k)] for k in range(1, 6)}
        p = {k: parse(s[k]) for k in s}
        agree = same(p[1], p[2])
        maj3 = [s[1], s[2], s[3]][vote([s[1], s[2], s[3]], [1, 2, 0])]
        maj5 = [s[1], s[2], s[3], s[4], s[5]][vote([s[1], s[2], s[3], s[4], s[5]], [1, 2, 3, 4, 0])]
        chosen = {"keep": s[1], "maj3": maj3, "maj5": maj5, "agree_gated": s[1] if agree else maj3}
        calls["agree_gated"] += 2 if agree else 3
        for j, judge in judges.items():
            if agree:
                chosen[f"self_check_{j}"] = s[1]
                calls[f"self_check_{j}"] += 2
            else:
                v = verdict(judge, pid, 2, p[1], p[2])
                chosen[f"self_check_{j}"] = s[1] if v == "first" else s[2] if v == "other" else maj3
                calls[f"self_check_{j}"] += 4 + (v == "inconsistent")
        for n in strategies:
            acc[n].append(ok(chosen[n]))
        for k in JUDGED:
            if (pid, k, "ab") not in judges["base"]:
                continue
            first_ok, other_ok = ok(s[1]), ok(s[k])
            if first_ok == other_ok:
                continue
            groups.append(pid)
            right = "first" if first_ok else "other"
            # In "ab" s1 is shown as A; in "ba" s1 is shown as B.
            right_pos = {"ab": "A" if right == "first" else "B", "ba": "B" if right == "first" else "A"}
            for j, judge in judges.items():
                v = verdict(judge, pid, k, p[1], p[k])
                verdicts[j]["inconsistent" if v == "inconsistent" else ("right" if v == right else "wrong")] += 1
                tie_scores[j].append(1.0 if v == right else 0.5 if v == "inconsistent" else 0.0)
                for order, (shown_a, shown_b) in (("ab", (p[1], p[k])), ("ba", (p[k], p[1]))):
                    pick = single_pick(judge[(pid, k, order)], shown_a, shown_b)
                    picks[j][(right_pos[order], pick)] += 1
                    single_ok[j].append(1.0 if pick == right_pos[order] else 0.0)
            single_groups += [pid, pid]
    a = {n: np.array(v) for n, v in acc.items()}
    keep = a["keep"]
    groups, single_groups = np.array(groups), np.array(single_groups)
    n_pairs = len(groups)

    def judge_block(j: str) -> dict:
        right, wrong, incons = verdicts[j]["right"], verdicts[j]["wrong"], verdicts[j]["inconsistent"]
        t = picks[j]
        ab_picks = sum(t[(pos, x)] for pos in "AB" for x in "AB")
        total = sum(t.values())
        sok = np.array(single_ok[j])
        return {
            "both_orders": {"right": right, "wrong": wrong, "inconsistent": incons,
                            "accuracy_consistent": right / (right + wrong) if right + wrong else None,
                            "coverage": (right + wrong) / n_pairs if n_pairs else None,
                            "p_one_sided": one_sided(right, wrong),
                            "tie_break_score": float(np.mean(tie_scores[j])) if n_pairs else None},
            "single_order": {"judgments": int(len(sok)), "accuracy": float(sok.mean()) if len(sok) else None,
                             "right": int(sok.sum()), "not_right": int(len(sok) - sok.sum()),
                             "ci_source_clustered": [0.5 + x for x in clustered_ci(sok, np.full(len(sok), 0.5),
                                                                                    single_groups, 20261010)]
                             if len(sok) else None},
            "diagnostics": {
                "position_2_rate_among_AB_picks": sum(t[(pos, "B")] for pos in "AB") / ab_picks if ab_picks else None,
                "invented_rate": sum(t[(pos, "invented")] for pos in "AB") / total if total else None,
                "picks_by_right_position": {pos: {x: t[(pos, x)] for x in ("A", "B", "invented")} for pos in "AB"}}}

    blocks = {j: judge_block(j) for j in judges}
    dpo = blocks["dpo"]
    # Secondary binomial: single-order right vs not-right (judgments are paired per pair;
    # the clustered CI above is the dependence-robust companion).
    raw = {"P1": dpo["both_orders"]["p_one_sided"],
           "Secondary": one_sided(dpo["single_order"]["right"], dpo["single_order"]["not_right"])}
    adjusted = holm(raw)
    mean_calls = calls["self_check_dpo"] / len(rows)
    k = next(k for k in (3, 5) if k >= mean_calls) if mean_calls <= 5 else 5
    vote_k = a["maj3"] if k == 3 else a["maj5"]
    p2_ci = ci(a["self_check_dpo"], vote_k, 20261011)
    p1_acc, p1_cov = dpo["both_orders"]["accuracy_consistent"], dpo["both_orders"]["coverage"]
    diag = dpo["diagnostics"]
    report = {
        "schema_version": "phase12_analysis_v2", "dataset": "SVAMP", "holdout_rows": len(rows),
        "initial_correct": int(keep.sum()), "one_right_pairs": n_pairs,
        "one_right_sources": int(len(np.unique(groups))),
        "accuracy": {n: float(v.mean()) for n, v in a.items()},
        "fixes": {n: int((v & ~keep).sum()) for n, v in a.items()},
        "harms": {n: int((~v & keep).sum()) for n, v in a.items()},
        "mean_model_calls": {"keep": 1, "maj3": 3, "maj5": 5,
                             **{n: calls[n] / len(rows) for n in ("agree_gated", "self_check_base", "self_check_dpo")}},
        "judges": blocks,
        "endpoints": {
            "P1": {"accuracy_consistent": p1_acc, "coverage": p1_cov, "p_one_sided": raw["P1"],
                   "p_holm": adjusted["P1"],
                   "baseline_untrained": blocks["base"]["both_orders"],
                   "pass": bool(p1_acc is not None and p1_acc > 0.5 and adjusted["P1"] < 0.05
                                and p1_cov is not None and p1_cov >= COVERAGE_MIN)},
            "P2": {"compute_matched_k": k, "self_check_dpo_mean_calls": mean_calls,
                   "difference": float(a["self_check_dpo"].mean() - vote_k.mean()), "ci": p2_ci,
                   "noninferiority_margin": NONINFERIORITY, "pass": bool(p2_ci[0] > NONINFERIORITY)},
            "Secondary": {"single_order_accuracy": dpo["single_order"]["accuracy"],
                          "p_one_sided": raw["Secondary"], "p_holm": adjusted["Secondary"],
                          "pass": bool(dpo["single_order"]["accuracy"] is not None
                                       and dpo["single_order"]["accuracy"] > 0.5 and adjusted["Secondary"] < 0.05)},
            "Diagnostic_order_bias": {"position_2_rate": diag["position_2_rate_among_AB_picks"],
                                      "within_40_60": bool(diag["position_2_rate_among_AB_picks"] is not None
                                                           and 0.40 <= diag["position_2_rate_among_AB_picks"] <= 0.60)},
            "Diagnostic_invented": {"rate": diag["invented_rate"],
                                    "below_5pct": bool(diag["invented_rate"] is not None and diag["invented_rate"] < 0.05)},
            "Report_tie_break_score": {j: blocks[j]["both_orders"]["tie_break_score"] for j in judges}},
        "secondary_contrasts": {}}
    report["secondary_contrasts"]["tie_break_dpo_vs_base"] = {
        "difference": float(np.mean(tie_scores["dpo"]) - np.mean(tie_scores["base"])) if n_pairs else None,
        "ci_source_clustered": clustered_ci(np.array(tie_scores["dpo"]), np.array(tie_scores["base"]),
                                           groups, 20261012) if n_pairs else None}
    for name, (left, right_name) in {"self_check_dpo_vs_agree_gated": ("self_check_dpo", "agree_gated"),
                                     "self_check_dpo_vs_keep": ("self_check_dpo", "keep"),
                                     "self_check_dpo_vs_base": ("self_check_dpo", "self_check_base"),
                                     "self_check_dpo_vs_maj5": ("self_check_dpo", "maj5"),
                                     "maj5_vs_keep": ("maj5", "keep")}.items():
        report["secondary_contrasts"][name] = {"difference": float(a[left].mean() - a[right_name].mean()),
                                               "ci": ci(a[left], a[right_name], 20261013),
                                               "p_raw": paired_p(a[left], a[right_name])}
    REPORT.mkdir(parents=True)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["endpoints"], indent=2))


if __name__ == "__main__":
    main()
