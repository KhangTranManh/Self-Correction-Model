"""Per-cell (model x dataset) metrics shared by the pilot report and the analysis.

Definitions, on one-right pairs (exactly one of s1, sk is verifier-correct):
  pick            "A", "B", or "invented" for one single-order judgment
                  (Phase 13 rule: constrained judges use the verdict line)
  single_accuracy share of single-order judgments that pick the right solution
  skill_index     P(pick A | right shown as A) - P(pick A | right shown as B).
                  A pure position bias moves both terms equally and cancels;
                  invented answers count as "not A" in both terms.
  position_2_rate share of A/B picks that are B
  asymmetry       P(right | right shown as B) - P(right | right shown as A)
  consistent      both orders pick the same solution (Phase 12-13 rule)
  tie_break       1 consistent-right, 0 consistent-wrong, 0.5 otherwise
  tie_minus_single
                  tie_break - single_accuracy. Algebraically this equals
                  0.5 x P(no order picks the right solution and at least one
                  order invents), so it measures invented answers, not bias;
                  reported with that identity checked.
  logprob debias  (constrained judge) m = logP(A) - logP(B) at the verdict
                  token. Per pair, score_first = (m_ab - m_ba) / 2 cancels an
                  additive position bias; debiased_accuracy uses its sign.
                  debias_gain = debiased_accuracy - single_order accuracy of the
                  same log-probabilities on the same pairs. bias_logodds =
                  mean (m_ab + m_ba) / 2 (> 0 favors position A).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _to_sympy  # noqa: E402
from phase9.scripts.answers import vote  # noqa: E402
from phase13.scripts.constrained import pick as pick_phase13  # noqa: E402
from phase14.scripts.answers import MODES, parse, same  # noqa: E402
from phase14.scripts.config import JUDGED  # noqa: E402

CONSTRAINED = {"base": False, "base_c": True}
BOOTSTRAP = 10000


def clustered(stat, groups: np.ndarray, seed: int) -> list[float]:
    """Source-clustered bootstrap 95% interval of ``stat(index_array)``."""
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP):
        take = np.concatenate([members[g] for g in rng.choice(ids, size=len(ids))])
        value = stat(take)
        if value is not None and np.isfinite(value):
            values.append(value)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else [None, None]


def paired_ci(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(BOOTSTRAP, len(delta)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def cell(rows: list[dict], samples: dict, judges: dict[str, dict], seed: int, mode: str = "lenient") -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    ref = {r["id"]: _to_sympy(r["reference_answer"]) for r in rows}
    ok = lambda pid, text: same(parse(text, mode), ref[pid])  # noqa: E731
    acc = {n: [] for n in ("keep", "maj3", "maj5", "agree_gated", "keep_full", "maj5_full",
                           *[f"self_check_{j}" for j in judges])}
    expected = {n: [] for n in ("random_s1_s2", *[f"judge_or_random_{j}" for j in judges])}
    cap = {"primary": 0, "full": 0, "attempts": 0}
    pairs = []  # one row per one-right pair
    for row in rows:
        pid = row["id"]
        s = {k: samples[(pid, k)]["output"] for k in range(1, 6)}
        full = {k: samples[(pid, k)]["output_full"] for k in range(1, 6)}
        for k in range(1, 6):
            if samples[(pid, k)]["hit_primary_cap"] is None:  # bridge set: Phase 13 attempts, not recorded
                continue
            cap["attempts"] += 1
            cap["primary"] += samples[(pid, k)]["hit_primary_cap"]
            cap["full"] += samples[(pid, k)]["hit_token_cap"]
        p = {k: parse(s[k]) for k in s}
        agree12 = same(p[1], p[2])
        maj3 = [s[1], s[2], s[3]][vote([s[1], s[2], s[3]], [1, 2, 0])]
        order5 = [s[1], s[2], s[3], s[4], s[5]]
        chosen = {"keep": s[1], "maj3": maj3, "maj5": order5[vote(order5, [1, 2, 3, 4, 0])],
                  "agree_gated": s[1] if agree12 else maj3}
        full5 = [full[k] for k in range(1, 6)]
        chosen["keep_full"], chosen["maj5_full"] = full[1], full5[vote(full5, [1, 2, 3, 4, 0])]
        for j, texts in judges.items():
            if agree12:
                chosen[f"self_check_{j}"] = s[1]
            else:
                v = verdict(texts, pid, 2, p[1], p[2], CONSTRAINED[j])
                chosen[f"self_check_{j}"] = s[1] if v == "first" else s[2] if v == "other" else maj3
        for name, text in chosen.items():
            acc[name].append(ok(pid, text))
        # Report section 5 baselines, as exact expectations (no random draw):
        # pick s1 or s2 at random; judge with a random pick when inconsistent.
        ok1, ok2 = ok(pid, s[1]), ok(pid, s[2])
        expected["random_s1_s2"].append((ok1 + ok2) / 2)
        for j, texts in judges.items():
            if agree12:
                expected[f"judge_or_random_{j}"].append(float(ok1))
            else:
                v = verdict(texts, pid, 2, p[1], p[2], CONSTRAINED[j])
                expected[f"judge_or_random_{j}"].append(float(ok1) if v == "first" else float(ok2)
                                                        if v == "other" else (ok1 + ok2) / 2)
        for k in JUDGED:
            if same(p[1], p[k]):
                continue  # not a judged pair (judged pairs disagree in the lenient primary view)
            first_ok, other_ok = ok(pid, s[1]), ok(pid, s[k])
            if first_ok == other_ok:
                continue
            right = "first" if first_ok else "other"
            entry = {"pid": pid, "k": k, "right": right, "judges": {}}
            for j, texts in judges.items():
                picks = {}
                for order, (shown_a, shown_b) in (("ab", (p[1], p[k])), ("ba", (p[k], p[1]))):
                    picks[order] = pick_phase13(texts[(pid, k, order)]["output"], shown_a, shown_b, CONSTRAINED[j])
                lp = {order: texts[(pid, k, order)].get("verdict_logprobs") for order in ("ab", "ba")}
                entry["judges"][j] = {"picks": picks, "verdict": verdict(texts, pid, k, p[1], p[k], CONSTRAINED[j]),
                                      "logprobs": lp}
            pairs.append(entry)
    a = {n: np.array(v, dtype=bool) for n, v in acc.items()}
    keep = a["keep"]
    disagreeing = sum(not same(parse(samples[(r["id"], 1)]["output"]), parse(samples[(r["id"], k)]["output"]))
                      for r in rows for k in JUDGED)
    result = {"mode": mode, "problems": len(rows), "initial_correct": int(keep.sum()),
              "judged_pairs": disagreeing, "one_right_pairs": len(pairs),
              "one_right_pairs_per_problem": len(pairs) / len(rows),
              "one_right_sources": len({e["pid"] for e in pairs}),
              "cap_hit_rate": {"primary_768": cap["primary"] / cap["attempts"],
                               "full": cap["full"] / cap["attempts"]} if cap["attempts"] else None,
              "accuracy": {**{n: float(v.mean()) for n, v in a.items()},
                           **{n: float(np.mean(v)) for n, v in expected.items()}},
              "fixes": {n: int((v & ~keep).sum()) for n, v in a.items()},
              "harms": {n: int((~v & keep).sum()) for n, v in a.items()},
              "maj5_minus_keep": {"difference": float(a["maj5"].mean() - keep.mean()),
                                  "ci": paired_ci(a["maj5"].astype(float) - keep.astype(float), seed)},
              "judges": {j: judge_block(pairs, j, seed + i) for i, j in enumerate(judges)}}
    return result


def verdict(texts: dict, pid: str, k: int, first, other, constrained: bool) -> str:
    """Phase 13 rule: "first"/"other" if both orderings pick the same solution, else "inconsistent"."""
    ab = pick_phase13(texts[(pid, k, "ab")]["output"], first, other, constrained)
    ba = pick_phase13(texts[(pid, k, "ba")]["output"], other, first, constrained)
    ab = {"A": "first", "B": "other"}.get(ab)
    ba = {"A": "other", "B": "first"}.get(ba)
    return ab if ab is not None and ab == ba else "inconsistent"


def judge_block(pairs: list[dict], j: str, seed: int) -> dict:
    if not pairs:
        return {"one_right_pairs": 0}
    groups = np.array([e["pid"] for e in pairs])
    # Single-order rows: (right position, picked position).
    right_pos = np.array([[("A" if e["right"] == "first" else "B"), ("B" if e["right"] == "first" else "A")]
                          for e in pairs])
    picked = np.array([[e["judges"][j]["picks"]["ab"], e["judges"][j]["picks"]["ba"]] for e in pairs])
    verdicts = np.array([e["judges"][j]["verdict"] for e in pairs])
    rights = np.array([e["right"] for e in pairs])
    tie = np.where(verdicts == rights, 1.0, np.where(verdicts == "inconsistent", 0.5, 0.0))
    single_pair = (picked == right_pos).mean(axis=1)
    artifact = 0.5 * ((~(picked == right_pos).any(axis=1)) & (picked == "invented").any(axis=1))

    def rate(idx, cond_pos, value):
        rp, pk = right_pos[idx].ravel(), picked[idx].ravel()
        mask = rp == cond_pos
        return float((pk[mask] == value).mean()) if mask.any() else None

    def skill(idx):
        x, y = rate(idx, "A", "A"), rate(idx, "B", "A")
        return None if x is None or y is None else x - y

    def asym(idx):
        x, y = rate(idx, "B", "B"), rate(idx, "A", "A")
        return None if x is None or y is None else x - y

    every = np.arange(len(pairs))
    decisive = picked[np.isin(picked, ("A", "B"))]
    consistent = verdicts != "inconsistent"
    block = {
        "one_right_pairs": len(pairs),
        "single_accuracy": float(single_pair.mean()),
        "skill_index": skill(every), "skill_index_ci": clustered(skill, groups, seed),
        "picks_by_right_position": {pos: {x: int(((right_pos == pos) & (picked == x)).sum())
                                          for x in ("A", "B", "invented")} for pos in ("A", "B")},
        "position_2_rate": float((decisive == "B").mean()) if len(decisive) else None,
        "asymmetry": asym(every), "asymmetry_ci": clustered(asym, groups, seed + 1),
        "invented_rate": float((picked == "invented").mean()),
        "coverage": float(consistent.mean()),
        "consistent_accuracy": float((verdicts[consistent] == rights[consistent]).mean()) if consistent.any() else None,
        "all_pairs_accuracy": float((verdicts == rights).mean()),   # consistent and right / all pairs (JudgeBench)
        "tie_break": float(tie.mean()),
        "tie_break_ci": clustered(lambda i: tie[i].mean(), groups, seed + 4),
        "tie_minus_single": float(tie.mean() - single_pair.mean()),
        "tie_minus_single_ci": clustered(lambda i: tie[i].mean() - single_pair[i].mean(), groups, seed + 2),
        "invented_artifact": float(artifact.mean()),
    }
    if abs(block["tie_minus_single"] - block["invented_artifact"]) > 1e-9:
        raise RuntimeError("tie_minus_single identity failed; check the pick/verdict definitions")
    block["logprob"] = logprob_block(pairs, j, groups, seed + 3)
    return block


def logprob_block(pairs: list[dict], j: str, groups: np.ndarray, seed: int) -> dict | None:
    rows = []
    for e, g in zip(pairs, groups):
        lp = e["judges"][j]["logprobs"]
        if not all(lp.get(o) and lp[o]["A"] is not None and lp[o]["B"] is not None for o in ("ab", "ba")):
            continue
        m_ab, m_ba = lp["ab"]["A"] - lp["ab"]["B"], lp["ba"]["A"] - lp["ba"]["B"]
        first_right = e["right"] == "first"
        score_first = (m_ab - m_ba) / 2
        debiased = 0.5 if score_first == 0 else float((score_first > 0) == first_right)
        single_ab = 0.5 if m_ab == 0 else float((m_ab > 0) == first_right)          # A is first in "ab"
        single_ba = 0.5 if m_ba == 0 else float((m_ba < 0) == first_right)          # B is first in "ba"
        rows.append((g, debiased, (single_ab + single_ba) / 2, (m_ab + m_ba) / 2, m_ab, m_ba, first_right))
    if not rows:
        return None
    g = np.array([r[0] for r in rows])
    deb, single, bias, m_ab, m_ba = (np.array([r[i] for r in rows], dtype=float) for i in (1, 2, 3, 4, 5))
    first_right = np.array([r[6] for r in rows])
    # Bias-corrected single order: subtract a leave-one-source-out estimate of the
    # position bias from each single-order margin. This removes the bias without
    # combining the two readings, so the debias gain splits into
    #   bias_removal = corrected_single - single     (grows with |bias|)
    #   two_readings = debiased - corrected_single   (variance reduction from two looks)
    total, count = bias.sum(), len(bias)
    loo = np.empty(count)
    for source in np.unique(g):
        mask = g == source
        loo[mask] = (total - bias[mask].sum()) / max(count - mask.sum(), 1)
    c_ab, c_ba = m_ab - loo, m_ba - loo
    score = lambda m, first_is_a: np.where(m == 0, 0.5, ((m > 0) == first_is_a).astype(float))  # noqa: E731
    corrected = (score(c_ab, first_right) + score(c_ba, ~first_right)) / 2
    diff = lambda x, y: (lambda i: x[i].mean() - y[i].mean())  # noqa: E731
    return {"pairs_with_logprobs": len(rows), "logprob_coverage": len(rows) / len(pairs),
            "debiased_accuracy": float(deb.mean()), "single_order_accuracy": float(single.mean()),
            "corrected_single_accuracy": float(corrected.mean()),
            "debias_gain": float(deb.mean() - single.mean()),
            "debias_gain_ci": clustered(diff(deb, single), g, seed),
            "bias_removal_gain": float(corrected.mean() - single.mean()),
            "bias_removal_gain_ci": clustered(diff(corrected, single), g, seed + 2),
            "two_readings_gain": float(deb.mean() - corrected.mean()),
            "two_readings_gain_ci": clustered(diff(deb, corrected), g, seed + 3),
            "bias_logodds": float(bias.mean()),
            "bias_logodds_ci": clustered(lambda i: bias[i].mean(), g, seed + 1)}
