"""Open Phase 11 holdout labels once and score the DPO judge.

Pairs: every (s1, sk), k = 2..5, with different final answers. One-right
pairs (exactly one of s1/sk verifier-correct) form the judgment benchmark.
Pipeline (one per source): A0 = s1, B1 = s2.
  keep, maj3 (s1, s2, s3), agree_gated, maj5 (s1..s5),
  self_check_base / self_check_dpo (A0 if A0 == B1 else the judge on (s1, s2)).
Primary (Holm over two):
  P1 judgment accuracy on one-right pairs, DPO vs base (source-clustered bootstrap CI;
     exact sign test on discordant pairs)
  P2 self_check_dpo vs agree_gated (paired by source)
"""

from __future__ import annotations

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
from phase11.scripts.generate import OUT, completed, holdout  # noqa: E402

REPORT = OUT / "analysis"


def clustered_ci(left: np.ndarray, right: np.ndarray, groups: np.ndarray, seed: int) -> list[float]:
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(10000):
        take = np.concatenate([index[g] for g in rng.choice(ids, size=len(ids))])
        values.append(left[take].mean() - right[take].mean())
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    if REPORT.exists():
        raise RuntimeError("Refusing a second protected opening")
    samples = completed("samples")
    base, dpo = completed("judge_base"), completed("judge_dpo")
    rows = holdout()
    verifier = MathVerifier()  # First read of Phase 11 holdout gold.
    strategies = {k: [] for k in ("keep", "maj3", "agree_gated", "maj5", "self_check_base", "self_check_dpo")}
    judged, groups = [], []
    pair_count = 0
    for row in rows:
        pid = row["id"]
        problem = Problem(id=pid, domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        ok = lambda text: bool(verifier.verify(problem, text).passed)
        s = {k: samples[(pid, k)] for k in range(1, 6)}
        agree = same(parse(s[1]), parse(s[2]))
        maj3 = [s[1], s[2], s[3]][vote([s[1], s[2], s[3]], [1, 2, 0])]
        maj5 = [s[1], s[2], s[3], s[4], s[5]][vote([s[1], s[2], s[3], s[4], s[5]], [1, 2, 3, 4, 0])]
        for name, text in (("keep", s[1]), ("maj3", maj3), ("agree_gated", s[1] if agree else maj3),
                           ("maj5", maj5),
                           ("self_check_base", s[1] if agree else base[(pid, 2)]),
                           ("self_check_dpo", s[1] if agree else dpo[(pid, 2)])):
            strategies[name].append(ok(text))
        for k in range(2, 6):
            if (pid, k) not in base:
                continue
            pair_count += 1
            first_ok, other_ok = ok(s[1]), ok(s[k])
            if first_ok != other_ok:
                right = parse(s[1] if first_ok else s[k])
                judged.append((same(parse(base[(pid, k)]), right), same(parse(dpo[(pid, k)]), right),
                               "A_right" if first_ok else "B_right"))
                groups.append(pid)
    acc = {k: np.array(v) for k, v in strategies.items()}
    keep = acc["keep"]
    base_side = np.array([j[0] for j in judged])
    dpo_side = np.array([j[1] for j in judged])
    kinds = np.array([j[2] for j in judged])
    groups = np.array(groups)
    better, worse = int(np.sum(dpo_side & ~base_side)), int(np.sum(~dpo_side & base_side))
    raw = {"P1_judgment_dpo_vs_base":
               float(binomtest(min(better, worse), better + worse, 0.5).pvalue) if better + worse else 1.0,
           "P2_self_check_dpo_vs_agree_gated": paired_p(acc["self_check_dpo"], acc["agree_gated"])}
    adjusted = holm(raw)
    contrasts = {
        "P1_judgment_dpo_vs_base": {
            "one_right_pairs": len(judged), "sources": int(len(np.unique(groups))),
            "base_rate": float(base_side.mean()), "dpo_rate": float(dpo_side.mean()),
            "difference": float(dpo_side.mean() - base_side.mean()),
            "ci_source_clustered": clustered_ci(dpo_side, base_side, groups, 20261002),
            "discordant_better_worse": [better, worse],
            "by_kind": {k: {"n": int((kinds == k).sum()),
                            "base_rate": float(base_side[kinds == k].mean()),
                            "dpo_rate": float(dpo_side[kinds == k].mean())}
                        for k in ("A_right", "B_right") if (kinds == k).any()}},
        "P2_self_check_dpo_vs_agree_gated": {
            "difference": float(acc["self_check_dpo"].mean() - acc["agree_gated"].mean()),
            "ci": ci(acc["self_check_dpo"], acc["agree_gated"], 20261003)}}
    for name in raw:
        contrasts[name].update(p_raw=raw[name], p_holm=adjusted[name])
    for name, (left, right) in {"self_check_dpo_vs_keep": ("self_check_dpo", "keep"),
                                "self_check_dpo_vs_base": ("self_check_dpo", "self_check_base"),
                                "self_check_dpo_vs_maj5": ("self_check_dpo", "maj5"),
                                "maj5_vs_keep": ("maj5", "keep")}.items():
        contrasts[name] = {"family": "secondary",
                           "difference": float(acc[left].mean() - acc[right].mean()),
                           "ci": ci(acc[left], acc[right], 20261004),
                           "p_raw": paired_p(acc[left], acc[right])}
    report = {"schema_version": "phase11_analysis_v1", "holdout_rows": len(rows),
              "initial_correct": int(keep.sum()), "disagreeing_pairs_judged": pair_count,
              "accuracy": {k: float(v.mean()) for k, v in acc.items()},
              "fixes": {k: int((v & ~keep).sum()) for k, v in acc.items()},
              "harms": {k: int((~v & keep).sum()) for k, v in acc.items()},
              "contrasts": contrasts}
    REPORT.mkdir(parents=True)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
