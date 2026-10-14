"""D0-D3 and E1 analysis (CPU) of the 3090 judge-variant runs.

Inputs: outputs/paper_3090/judge_<variant>.jsonl (paper/scripts/gpu/judge_variants.py)
plus the V100 Phase 13 mirror. Picks are positional: "A" = the solution shown first.
Per variant on GSM8K one-right pairs: the B1/B2/B4 tables of cpu_checks.
Net position-2 preference per single-order judgment: +1 picks the second shown
solution, -1 the first, 0 invented. Three hypotheses, Holm-corrected, two-sided
problem-clustered bootstrap:
  D1  Qwen2.5-7B-Instruct minus Self_Correction (d0): != 0 (checklist: confirmed
      if |difference| >= 0.10 and the CI excludes 0)
  D2  reversed labels minus d0 (same model, same 3090): != 0 means the preference
      follows the label, ~0 means it follows the position
  D3  problem restated after the solutions minus d0 (checklist: confirmed if the
      preference drops by >= 0.10)
D0: agreement of the 3090 base judge with the V100 base judge (exact text, pick).
E1: four candidates, four cyclic rotations; position pick distribution, per-problem
plurality over rotations vs vote@4 / vote@5 / keep on the same problems.
Output: paper/results/d_e_analysis.json.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import cpu_checks as cc  # noqa: E402

RUNS = ROOT / "outputs/paper_3090"
OUT = ROOT / "paper/results/d_e_analysis.json"
VARIANTS = ("d0_base", "d1_instruct", "d2_relabel", "d3_restate")


def net_pos2(d: dict, judge: str, pairs) -> tuple[np.ndarray, np.ndarray]:
    vals, groups = [], []
    for pid, k in pairs:
        pk = cc.picks_for(d, judge, pid, k)
        for order in ("ab", "ba"):
            raw = pk[f"{order}_raw"]
            vals.append(1.0 if raw == "B" else -1.0 if raw == "A" else 0.0); groups.append(pid)
    return np.array(vals), np.array(groups)


def e1(d: dict) -> dict:
    recs = cc.read_jsonl(RUNS / "judge_e1_four.jsonl")
    by = {}
    for r in recs:
        pid, rot, order = r["task"]
        by.setdefault(pid, {})[rot] = ([int(c) for c in order], cc.parse(r["output"]))
    pos = Counter()
    rot0_ok, plur_ok, v4_ok, v5_ok, keep_ok, chance, single_ok = [], [], [], [], [], [], []
    for pid, rots in by.items():
        cands = {k: d["parsed"][(pid, k)] for k in range(1, 6)}
        ok = lambda v: v is not None and any(cc.same(v, cands[k]) and d["correct"][(pid, k)] for k in range(1, 6))
        picked = []
        for rot, (order, ans) in sorted(rots.items()):
            hit = next((i for i, k in enumerate(order) if cc.same(ans, cands[k])), None)
            pos[hit if hit is not None else "invented"] += 1
            if hit is not None:
                picked.append(cands[order[hit]])
            single_ok.append(ok(ans))
            if rot == 0:
                rot0_ok.append(ok(ans))
        # plurality over rotations of picked candidate answers; ties -> s1-first priority
        if picked:
            support = [sum(cc.same(a, b) for b in picked) for a in picked]
            plur_ok.append(ok(picked[int(np.argmax(support))]))
        else:
            plur_ok.append(False)
        texts4 = [d["samples"][(pid, k)] for k in range(1, 5)]
        v4_ok.append(d["correct"][(pid, 1 + cc.vote(texts4, [0, 1, 2, 3]))])
        v5_ok.append(d["correct"][(pid, 1 + cc.vote(texts4 + [d["samples"][(pid, 5)]], [1, 2, 3, 4, 0]))])
        keep_ok.append(d["correct"][(pid, 1)])
        chance.append(np.mean([d["correct"][(pid, k)] for k in range(1, 5)]))
    total = sum(pos.values())
    return {"problems": len(by), "judgments": total,
            "position_pick_share": {str(k): v / total for k, v in sorted(pos.items(), key=str)},
            "single_judgment_accuracy": float(np.mean(single_ok)),
            "uniform_random_candidate_accuracy": float(np.mean(chance)),
            "accuracy_same_problems": {"rotation0_only": float(np.mean(rot0_ok)),
                                       "plurality_over_4_rotations": float(np.mean(plur_ok)),
                                       "vote4": float(np.mean(v4_ok)), "vote5": float(np.mean(v5_ok)),
                                       "keep": float(np.mean(keep_ok))},
            "note": "four-rotation plurality costs 4 candidates + 4 judge calls = 8 calls"}


def main() -> None:
    d = cc.load("gsm8k")
    for v in VARIANTS:
        recs = cc.read_jsonl(RUNS / f"judge_{v}.jsonl")
        d["judges"][v] = {tuple(r["task"]): r["output"] for r in recs}
        d["meta"][v] = {tuple(r["task"]): r for r in recs}
    tables = cc.judge_tables(d)
    pairs = cc.one_right_pairs(d)
    keep = ("B1_position_table", "B1_skill_index", "B1_skill_index_ci", "B1_position2_rate_among_AB",
            "B2_tie_break", "B2_consistent_accuracy", "B2_coverage", "B4_single_order_accuracy")
    res = {"schema_version": "paper_d_e_v1", "post_hoc": True, "one_right_pairs": len(pairs),
           "tables": {j: {k: tables[j][k] for k in keep} for j in ("base", *VARIANTS)}}
    for j in ("base", *VARIANTS):
        m = d["meta"][j]
        res["tables"][j]["token_cap_rate"] = float(np.mean([r.get("hit_token_cap", False) for r in m.values()]))
    v100, g3090 = d["judges"]["base"], d["judges"]["d0_base"]
    common = sorted(set(v100) & set(g3090))
    same_pick = []
    for pid, k in pairs:
        a, b = cc.picks_for(d, "base", pid, k), cc.picks_for(d, "d0_base", pid, k)
        same_pick += [a["ab_raw"] == b["ab_raw"], a["ba_raw"] == b["ba_raw"]]
    res["D0_control"] = {"tasks": len(common), "exact_text_match": float(np.mean([v100[t] == g3090[t] for t in common])),
                         "same_pick_one_right": float(np.mean(same_pick))}
    nets = {j: net_pos2(d, j, pairs) for j in ("d0_base", *VARIANTS[1:])}
    groups = nets["d0_base"][1]
    raw, tests = {}, {}
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(20261012)
    boots = [np.concatenate([idx[g] for g in rng.choice(ids, len(ids))]) for _ in range(10000)]
    for name, diff in (("D1_instruct_minus_d0", nets["d1_instruct"][0] - nets["d0_base"][0]),
                       ("D2_relabel_minus_d0", nets["d2_relabel"][0] - nets["d0_base"][0]),
                       ("D3_restate_minus_d0", nets["d3_restate"][0] - nets["d0_base"][0])):
        means = np.array([diff[b].mean() for b in boots])
        p = float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))
        tests[name] = {"estimate": float(diff.mean()), "ci": [float(x) for x in np.quantile(means, [0.025, 0.975])],
                       "p_boot": p}
        raw[name] = p
    for k, p in cc.holm(raw).items():
        tests[k]["p_holm"] = p
    tests["D1_instruct_minus_d0"]["confirmed"] = bool(abs(tests["D1_instruct_minus_d0"]["estimate"]) >= 0.10 and
                                                      np.prod(tests["D1_instruct_minus_d0"]["ci"]) > 0)
    tests["D3_restate_minus_d0"]["confirmed"] = bool(tests["D3_restate_minus_d0"]["estimate"] <= -0.10 and
                                                     tests["D3_restate_minus_d0"]["ci"][1] < 0)
    means = np.array([nets["d1_instruct"][0][b].mean() for b in boots])
    res["D1_instruct_net_position2_alone"] = {"estimate": float(nets["d1_instruct"][0].mean()),
                                              "ci": [float(x) for x in np.quantile(means, [0.025, 0.975])]}
    res["net_position2_preference"] = {j: float(v[0].mean()) for j, v in nets.items()}
    res["D_tests"] = tests
    res["E1"] = e1(d) if (RUNS / "judge_e1_four.jsonl").exists() else None
    OUT.write_text(json.dumps(cc.strip(res), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(cc.strip(res), indent=1))


if __name__ == "__main__":
    main()
