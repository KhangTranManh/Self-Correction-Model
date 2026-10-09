"""Open the Phase 13 holdout once and score directions A, B, and C.

Pairs: (s1, sk), k in (2, 3), with different final answers, judged in order
"ab" (s1 shown as Solution A) and "ba" (s1 shown as Solution B). A one-right
pair has exactly one verifier-correct solution. A judge's both-orders verdict
is consistent when both orderings pick the same solution.

A (Phase 12 judge "p12" on GSM8K test):
  A-P1  consistent accuracy > 50%, coverage >= 30%, one-sided binomial p < 0.05
        after Holm over {A-P1, A-Sec}
  A-P2  self_check_p12 vs compute-matched vote@k: lower 95% bound > -2 points
  A-Sec single-order accuracy > 50%, Holm with A-P1
B (constrained judge "p13b"):
  B1    invented rate (single-order) < 5%
  B2    consistent accuracy > 50%, one-sided binomial p < 0.05, coverage >= 30%
C (judge only on split votes, p12):
  split_judge: s1 if s1 = s2 = s3; otherwise the p12 verdict on (s1, sk) for the
  first k in (2, 3) with sk != s1, falling back to maj3 if inconsistent.
  C1    split_judge - vote@3: 95% CI above 0 and exact paired p < 0.05
  C2    split_judge - vote@5: lower 95% bound > -2 points
"""

from __future__ import annotations

from collections import Counter
import json
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
from phase13.scripts.constrained import pick  # noqa: E402
from phase13.scripts.generate import JUDGED, JUDGES, OUT, completed, holdout  # noqa: E402

REPORT = OUT / "analysis"
COVERAGE_MIN = 0.30
MARGIN = -0.02


def one_sided(right: int, wrong: int) -> float:
    return float(binomtest(right, right + wrong, 0.5, alternative="greater").pvalue) if right + wrong else 1.0


def verdict(texts: dict, pid: str, k: int, first, other, constrained: bool) -> str:
    """"first" or "other" if both orderings pick the same solution, else "inconsistent"."""
    ab = pick(texts[(pid, k, "ab")], first, other, constrained)   # s1 shown as A
    ba = pick(texts[(pid, k, "ba")], other, first, constrained)   # s1 shown as B
    ab = {"A": "first", "B": "other"}.get(ab)
    ba = {"A": "other", "B": "first"}.get(ba)
    return ab if ab is not None and ab == ba else "inconsistent"


def main() -> None:
    if REPORT.exists():
        raise RuntimeError("Refusing a second protected opening")
    samples = completed("samples")
    judges = {name: completed(f"judge_{name}") for name in JUDGES}
    rows = holdout()
    verifier = MathVerifier()  # First read of Phase 13 holdout gold.
    strategies = ["keep", "maj3", "maj5", "agree_gated", *[f"self_check_{j}" for j in JUDGES],
                  "split_judge_p12", "split_judge_p13b"]
    acc = {n: [] for n in strategies}
    calls = Counter()
    stats = {j: {"verdicts": Counter(), "tie": [], "single": [], "picks": Counter()} for j in JUDGES}
    groups, single_groups = [], []
    for row in rows:
        pid = row["id"]
        gold = Problem(id=pid, domain="math", question=row["question"], reference_answer=row["reference_answer"])
        ok = lambda text: bool(verifier.verify(gold, text).passed)
        s = {k: samples[(pid, k)] for k in range(1, 6)}
        p = {k: parse(s[k]) for k in s}
        agree12 = same(p[1], p[2])
        maj3 = [s[1], s[2], s[3]][vote([s[1], s[2], s[3]], [1, 2, 0])]
        maj5 = [s[1], s[2], s[3], s[4], s[5]][vote([s[1], s[2], s[3], s[4], s[5]], [1, 2, 3, 4, 0])]
        chosen = {"keep": s[1], "maj3": maj3, "maj5": maj5, "agree_gated": s[1] if agree12 else maj3}
        calls["agree_gated"] += 2 if agree12 else 3
        for j, (_, constrained) in JUDGES.items():
            if agree12:
                chosen[f"self_check_{j}"] = s[1]
                calls[f"self_check_{j}"] += 2
            else:
                v = verdict(judges[j], pid, 2, p[1], p[2], constrained)
                chosen[f"self_check_{j}"] = s[1] if v == "first" else s[2] if v == "other" else maj3
                calls[f"self_check_{j}"] += 4 + (v == "inconsistent")
        unanimous = agree12 and same(p[1], p[3])
        for j in ("p12", "p13b"):
            name = f"split_judge_{j}"
            if unanimous:
                chosen[name] = s[1]
                calls[name] += 3
            else:
                k = 2 if not agree12 else 3
                v = verdict(judges[j], pid, k, p[1], p[k], JUDGES[j][1])
                chosen[name] = s[1] if v == "first" else s[k] if v == "other" else maj3
                calls[name] += 5
        for n in strategies:
            acc[n].append(ok(chosen[n]))
        for k in JUDGED:
            if (pid, k, "ab") not in judges["base"]:
                continue
            first_ok, other_ok = ok(s[1]), ok(s[k])
            if first_ok == other_ok:
                continue
            groups.append(pid)
            single_groups += [pid, pid]
            right = "first" if first_ok else "other"
            right_pos = {"ab": "A" if right == "first" else "B", "ba": "B" if right == "first" else "A"}
            for j, (_, constrained) in JUDGES.items():
                st = stats[j]
                v = verdict(judges[j], pid, k, p[1], p[k], constrained)
                st["verdicts"]["inconsistent" if v == "inconsistent" else ("right" if v == right else "wrong")] += 1
                st["tie"].append(1.0 if v == right else 0.5 if v == "inconsistent" else 0.0)
                for order, (shown_a, shown_b) in (("ab", (p[1], p[k])), ("ba", (p[k], p[1]))):
                    chosen_pos = pick(judges[j][(pid, k, order)], shown_a, shown_b, constrained)
                    st["picks"][(right_pos[order], chosen_pos)] += 1
                    st["single"].append(1.0 if chosen_pos == right_pos[order] else 0.0)
    a = {n: np.array(v) for n, v in acc.items()}
    keep = a["keep"]
    groups, single_groups = np.array(groups), np.array(single_groups)
    n_pairs = len(groups)

    def block(j: str) -> dict:
        st = stats[j]
        right, wrong, incons = st["verdicts"]["right"], st["verdicts"]["wrong"], st["verdicts"]["inconsistent"]
        t = st["picks"]
        ab_picks = sum(t[(pos, x)] for pos in "AB" for x in "AB")
        total = sum(t.values())
        sok = np.array(st["single"])
        return {"consistent_accuracy": right / (right + wrong) if right + wrong else None,
                "coverage": (right + wrong) / n_pairs if n_pairs else None,
                "right": right, "wrong": wrong, "inconsistent": incons,
                "p_one_sided": one_sided(right, wrong),
                "tie_break_score": float(np.mean(st["tie"])) if n_pairs else None,
                "single_order_accuracy": float(sok.mean()) if len(sok) else None,
                "single_right": int(sok.sum()), "single_not_right": int(len(sok) - sok.sum()),
                "single_ci_source_clustered": [0.5 + x for x in clustered_ci(
                    sok, np.full(len(sok), 0.5), single_groups, 20261020)] if len(sok) else None,
                "position_2_rate": sum(t[(pos, "B")] for pos in "AB") / ab_picks if ab_picks else None,
                "invented_rate": sum(t[(pos, "invented")] for pos in "AB") / total if total else None,
                "picks_by_right_position": {pos: {x: t[(pos, x)] for x in ("A", "B", "invented")} for pos in "AB"}}

    blocks = {j: block(j) for j in JUDGES}
    p12, p13b = blocks["p12"], blocks["p13b"]
    holm_a = holm({"A-P1": p12["p_one_sided"],
                   "A-Sec": one_sided(p12["single_right"], p12["single_not_right"])})
    mean_calls = calls["self_check_p12"] / len(rows)
    k_match = 3 if mean_calls <= 3 else 5
    vote_k = a["maj3"] if k_match == 3 else a["maj5"]
    a_p2_ci = ci(a["self_check_p12"], vote_k, 20261021)
    c1_ci = ci(a["split_judge_p12"], a["maj3"], 20261022)
    c1_p = paired_p(a["split_judge_p12"], a["maj3"])
    c2_ci = ci(a["split_judge_p12"], a["maj5"], 20261023)
    passes = lambda b, p_value: bool(b["consistent_accuracy"] is not None and b["consistent_accuracy"] > 0.5
                                     and p_value < 0.05 and b["coverage"] is not None and b["coverage"] >= COVERAGE_MIN)
    report = {
        "schema_version": "phase13_analysis_v1", "dataset": "GSM8K test 750-1318",
        "holdout_rows": len(rows), "initial_correct": int(keep.sum()),
        "one_right_pairs": n_pairs, "one_right_sources": int(len(np.unique(groups))),
        "accuracy": {n: float(v.mean()) for n, v in a.items()},
        "fixes": {n: int((v & ~keep).sum()) for n, v in a.items()},
        "harms": {n: int((~v & keep).sum()) for n, v in a.items()},
        "mean_model_calls": {"keep": 1, "maj3": 3, "maj5": 5,
                             **{n: calls[n] / len(rows) for n in calls}},
        "judges": blocks,
        "A": {"A-P1": {"consistent_accuracy": p12["consistent_accuracy"], "coverage": p12["coverage"],
                       "p_holm": holm_a["A-P1"], "baseline_untrained": blocks["base"]["consistent_accuracy"],
                       "pass": passes(p12, holm_a["A-P1"])},
              "A-P2": {"compute_matched_k": k_match, "self_check_mean_calls": mean_calls,
                       "difference": float(a["self_check_p12"].mean() - vote_k.mean()), "ci": a_p2_ci,
                       "pass": bool(a_p2_ci[0] > MARGIN)},
              "A-Sec": {"single_order_accuracy": p12["single_order_accuracy"], "p_holm": holm_a["A-Sec"],
                        "pass": bool(p12["single_order_accuracy"] > 0.5 and holm_a["A-Sec"] < 0.05)},
              "diagnostics": {"position_2_rate": p12["position_2_rate"], "invented_rate": p12["invented_rate"]}},
        "B": {"B1": {"invented_rate": p13b["invented_rate"], "pass": bool(p13b["invented_rate"] < 0.05)},
              "B2": {"consistent_accuracy": p13b["consistent_accuracy"], "coverage": p13b["coverage"],
                     "p_one_sided": p13b["p_one_sided"], "pass": passes(p13b, p13b["p_one_sided"])},
              "report": {"untrained_constrained": {k: blocks["base_c"][k] for k in
                                                   ("consistent_accuracy", "coverage", "invented_rate")},
                         "p13b_vs_p12_tie_break": {
                             "difference": float(np.mean(stats["p13b"]["tie"]) - np.mean(stats["p12"]["tie"])),
                             "ci_source_clustered": clustered_ci(np.array(stats["p13b"]["tie"]),
                                                                np.array(stats["p12"]["tie"]), groups, 20261024)},
                         "position_2_rate": p13b["position_2_rate"]}},
        "C": {"mean_calls": calls["split_judge_p12"] / len(rows),
              "C1_vs_vote3": {"difference": float(a["split_judge_p12"].mean() - a["maj3"].mean()),
                              "ci": c1_ci, "p_raw": c1_p, "pass": bool(c1_ci[0] > 0 and c1_p < 0.05)},
              "C2_vs_vote5": {"difference": float(a["split_judge_p12"].mean() - a["maj5"].mean()),
                              "ci": c2_ci, "pass": bool(c2_ci[0] > MARGIN)},
              "report_split_judge_p13b": {"accuracy": float(a["split_judge_p13b"].mean()),
                                          "vs_vote5": float(a["split_judge_p13b"].mean() - a["maj5"].mean())}}}
    REPORT.mkdir(parents=True)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("A", "B", "C")}, indent=2))


if __name__ == "__main__":
    main()
