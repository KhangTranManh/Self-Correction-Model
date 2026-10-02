"""Open Phase 10 holdout labels once and score the trained judge.

A0 = sample 1 (initial prompt); B1..B4 = samples 2..5 (blind prompt).
Strategies: keep, maj3 (A0, B1, B2), agree_gated (= A0 if A0 == B1 else maj3),
maj5 (A0, B1..B4), self_check_base (A0 if agree else J_base), and
self_check_trained (A0 if agree else J_trained).
Primary (Holm over two): P1 judgment accuracy on disagreements where exactly
one of A0/B1 is right, trained vs base; P2 self_check_trained vs agree_gated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase9.scripts.analyze import ci, holm, paired_p  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase10.scripts.generate import OUT, completed, rows  # noqa: E402

REPORT = OUT / "analysis"


def main() -> None:
    if REPORT.exists():
        raise RuntimeError("Refusing a second protected opening")
    samples = completed("holdout_samples")
    j_base = completed("holdout_judge_base")
    j_trained = completed("holdout_judge_trained")
    sources = rows("holdout")
    verifier = MathVerifier()  # First read of holdout gold answers.
    strategies = {name: [] for name in ("keep", "maj3", "agree_gated", "maj5",
                                        "self_check_base", "self_check_trained")}
    judged = []  # (sided_right_base, sided_right_trained) on one-right disagreements
    format_rate = {"base": 0, "trained": 0}
    disagreements = 0
    for row in sources:
        pid = row["id"]
        problem = Problem(id=pid, domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        ok = lambda text: bool(verifier.verify(problem, text).passed)
        a0 = samples[(pid, 1)]
        b = [samples[(pid, k)] for k in range(2, 6)]
        base, trained = j_base[(pid,)], j_trained[(pid,)]
        agree = same(parse(a0), parse(b[0]))
        maj3 = [a0, b[0], b[1]][vote([a0, b[0], b[1]], [1, 2, 0])]
        maj5 = [a0, *b][vote([a0, *b], [1, 2, 3, 4, 0])]
        for name, text in (("keep", a0), ("maj3", maj3), ("agree_gated", a0 if agree else maj3),
                           ("maj5", maj5), ("self_check_base", a0 if agree else base),
                           ("self_check_trained", a0 if agree else trained)):
            strategies[name].append(ok(text))
        if not agree:
            disagreements += 1
            format_rate["base"] += "Phát hiện lỗi" in base
            format_rate["trained"] += "Phát hiện lỗi" in trained
            a_ok, b_ok = ok(a0), ok(b[0])
            if a_ok != b_ok:
                right = parse(a0 if a_ok else b[0])
                judged.append((same(parse(base), right), same(parse(trained), right)))
    acc = {name: np.array(v) for name, v in strategies.items()}
    keep = acc["keep"]
    base_side = np.array([j[0] for j in judged])
    trained_side = np.array([j[1] for j in judged])
    primary_raw = {"P1_judgment_trained_vs_base": paired_p(trained_side, base_side),
                   "P2_self_check_trained_vs_agree_gated": paired_p(acc["self_check_trained"],
                                                                     acc["agree_gated"])}
    primary_holm = holm(primary_raw)
    contrasts = {
        "P1_judgment_trained_vs_base": {
            "n_pairs": len(judged), "base_rate": float(base_side.mean()),
            "trained_rate": float(trained_side.mean()),
            "difference": float(trained_side.mean() - base_side.mean()),
            "ci": ci(trained_side, base_side, 20261001)},
        "P2_self_check_trained_vs_agree_gated": {
            "difference": float(acc["self_check_trained"].mean() - acc["agree_gated"].mean()),
            "ci": ci(acc["self_check_trained"], acc["agree_gated"], 20261002)}}
    for name in contrasts:
        contrasts[name]["p_raw"] = primary_raw[name]
        contrasts[name]["p_holm"] = primary_holm[name]
    for name, (left, right) in {"self_check_trained_vs_keep": ("self_check_trained", "keep"),
                                "self_check_trained_vs_base": ("self_check_trained", "self_check_base"),
                                "self_check_trained_vs_maj5": ("self_check_trained", "maj5"),
                                "maj5_vs_keep": ("maj5", "keep")}.items():
        contrasts[name] = {"family": "secondary",
                           "difference": float(acc[left].mean() - acc[right].mean()),
                           "ci": ci(acc[left], acc[right], 20261003),
                           "p_raw": paired_p(acc[left], acc[right])}
    report = {"schema_version": "phase10_analysis_v1", "holdout_rows": len(sources),
              "initial_correct": int(keep.sum()), "disagreements": disagreements,
              "accuracy": {k: float(v.mean()) for k, v in acc.items()},
              "fixes": {k: int((v & ~keep).sum()) for k, v in acc.items()},
              "harms": {k: int((~v & keep).sum()) for k, v in acc.items()},
              "error_format_rate_on_disagreement": {k: v / max(1, disagreements)
                                                     for k, v in format_rate.items()},
              "contrasts": contrasts}
    REPORT.mkdir(parents=True)
    path = REPORT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
