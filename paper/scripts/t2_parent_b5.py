"""T2: B5 rows with the parent model (Qwen2.5-7B-Instruct) as the judge (CPU).

GSM8K Phase 13 solutions (Self_Correction_v1 s1..s5); judges: Self_Correction base
on the 3090 (d0), Instruct with the 768-token cap (d1) and with 1536 tokens (d1_long),
plus p12 (V100) for reference. Per problem, when s1 and s2 differ:
  judge_or_random  consistent verdict, else expected accuracy of a random pick
  self_check       consistent verdict, else vote@3 (one extra solve)
Compute = model calls (F5): self_check costs 2 if s1 = s2, else 4 (+1 when the
judge is inconsistent); vote@3 costs 3. Paired problem-level bootstrap CIs vs vote@3.
If T1 has run, the same rows for Instruct judging its own s1..s3 are added
(outputs/paper_3090/t1_samples.jsonl, judge_t1_parent_self.jsonl).
Output: paper/results/t2_parent_b5.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import cpu_checks as cc  # noqa: E402
from paper.scripts.d1_g1_robust import add  # noqa: E402
from paper.scripts.d_e_analysis import RUNS  # noqa: E402

OUT = ROOT / "paper/results/t2_parent_b5.json"


def paired_ci(a: np.ndarray, b: np.ndarray, seed: int) -> list[float]:
    diff = a - b
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def strategies(d: dict, judges: list[str], n_samples: int) -> dict:
    acc = {"keep": [], "random_s1_s2": [], "vote3": []}
    calls = {"vote3": []}
    for pid in d["rows"]:
        c = {k: float(d["correct"][(pid, k)]) for k in range(1, n_samples + 1)}
        p = {k: d["parsed"][(pid, k)] for k in range(1, n_samples + 1)}
        s = {k: d["samples"][(pid, k)] for k in range(1, n_samples + 1)}
        m3 = [1, 2, 3][cc.vote([s[1], s[2], s[3]], [1, 2, 0])]
        agree = cc.same(p[1], p[2])
        acc["keep"].append(c[1]); acc["vote3"].append(c[m3]); calls["vote3"].append(3)
        acc["random_s1_s2"].append(c[1] if agree else 0.5 * (c[1] + c[2]))
        if n_samples >= 5:
            acc.setdefault("vote5", []).append(c[1 + cc.vote([s[k] for k in range(1, 6)], [1, 2, 3, 4, 0])])
        for j in judges:
            jr, sc = f"judge_or_random_{j}", f"self_check_{j}"
            if agree:
                acc.setdefault(jr, []).append(c[1]); acc.setdefault(sc, []).append(c[1])
                calls.setdefault(sc, []).append(2)
                continue
            pk = cc.picks_for(d, j, pid, 2)
            v = pk["ab"] if pk["ab"] == pk["ba"] and pk["ab"] != "invented" else None
            chosen = c[1] if v == "first" else c[2] if v == "other" else None
            acc.setdefault(jr, []).append(chosen if chosen is not None else 0.5 * (c[1] + c[2]))
            acc.setdefault(sc, []).append(chosen if chosen is not None else c[m3])
            calls.setdefault(sc, []).append(4 if chosen is not None else 5)
    a = {k: np.array(v) for k, v in acc.items()}
    out = {"problems": len(d["rows"]), "accuracy": {k: float(v.mean()) for k, v in a.items()},
           "mean_calls": {k: float(np.mean(v)) for k, v in calls.items()}, "vs_vote3": {}}
    for i, k in enumerate(x for x in a if x.startswith(("self_check", "judge_or_random"))):
        out["vs_vote3"][k] = {"difference": float((a[k] - a["vote3"]).mean()),
                              "ci": paired_ci(a[k], a["vote3"], 20261060 + i)}
    return out


def t1_data() -> dict | None:
    sp, jp = RUNS / "t1_samples.jsonl", RUNS / "judge_t1_parent_self.jsonl"
    if not (sp.exists() and jp.exists()):
        return None
    base = cc.load("gsm8k")
    samples = {tuple(r["task"]): r["output"] for r in cc.read_jsonl(sp)}
    verifier = cc.MathVerifier()
    correct = {}
    for (pid, k), text in samples.items():
        r = base["rows"][pid]
        correct[(pid, k)] = bool(verifier.verify(cc.Problem(id=pid, domain="math", question=r["question"],
                                                            reference_answer=r["reference_answer"]), text).passed)
    recs = cc.read_jsonl(jp)
    return {"rows": base["rows"], "samples": samples, "correct": correct,
            "parsed": {k: cc.parse(t) for k, t in samples.items()},
            "judges": {"base": {tuple(r["task"]): r["output"] for r in recs},
                       "t1_parent_self": {tuple(r["task"]): r["output"] for r in recs}},
            "meta": {"t1_parent_self": {tuple(r["task"]): r for r in recs}}}


def main() -> None:
    d = cc.load("gsm8k")
    judges = ["p12"] + [n for n in ("d0_base", "d1_instruct", "d1_instruct_long") if add(d, n)]
    res = {"schema_version": "paper_t2_v1", "post_hoc": True,
           "gsm8k_self_correction_solutions": strategies(d, judges, 5)}
    t1 = t1_data()
    if t1:
        res["gsm8k_instruct_own_solutions"] = strategies(t1, ["t1_parent_self"], 3)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
