"""T3, T4b, T5 analysis (CPU).

Arms (table-D metrics on one-right pairs; primary metric tie-break):
  on Self_Correction's solutions: d0 (Self_Correction judge), d1 (Instruct, 768),
                                  d1_long (Instruct, 1536)
  on Instruct's own solutions:    t1 (Instruct, 768), t3 (Instruct, 1536),
                                  t5 (Self_Correction judge, 768)
Two scorings:
  strict   the project verifier and answer parser (as everywhere else)
  lenient  (T4b) a solution also counts as right, and an answer is also read, from
           the first number on its last "Đáp số" line when the parser fails to give
           a number (T4 found 29/561 Instruct s1 answers written as sentences,
           e.g. "Đáp số: Mỗi trẻ nhận được 9 trứng.", vs 4/561 for Self_Correction)
Output: paper/results/t3_t5_analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))
from paper.scripts import cpu_checks as cc  # noqa: E402
import phase13.scripts.constrained as constrained  # noqa: E402
from paper.scripts.d_e_analysis import RUNS  # noqa: E402
from paper.scripts.d1_g1_robust import KEEP  # noqa: E402
from paper.scripts.t1_analysis import boot_means  # noqa: E402

OUT = ROOT / "paper/results/t3_t5_analysis.json"
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
STRICT_PARSE = cc.parse


def final_line_number(text: str):
    i = text.rfind("Đáp số")
    if i < 0:
        return None
    m = NUM.search(text[i:i + 120])
    return m.group().replace(",", "") if m else None


def lenient_parse(text: str):
    value = STRICT_PARSE(text)
    try:
        float(value)
        return value
    except (TypeError, ValueError):
        num = final_line_number(text)
        return STRICT_PARSE(f"Đáp số: {num}") if num is not None else value


def set_scoring(lenient: bool) -> None:
    fn = lenient_parse if lenient else STRICT_PARSE
    cc.parse = fn
    constrained.parse = fn


def dataset(solutions: str, lenient: bool) -> dict:
    base = cc.load("gsm8k")
    if solutions == "instruct":
        samples = {tuple(r["task"]): r["output"] for r in cc.read_jsonl(RUNS / "t1_samples.jsonl")}
        verifier = cc.MathVerifier()
        correct = {}
        for (pid, k), text in samples.items():
            r = base["rows"][pid]
            correct[(pid, k)] = bool(verifier.verify(cc.Problem(id=pid, domain="math", question=r["question"],
                                                                reference_answer=r["reference_answer"]), text).passed)
        d = {"rows": base["rows"], "samples": samples, "correct": correct, "judges": {}, "meta": {}}
        files = {"t1": "judge_t1_parent_self", "t3": "judge_t3_parent_self_long", "t5": "judge_t5_sc_on_parent"}
    else:
        d = base
        d["judges"], d["meta"] = {}, {}
        files = {"d0": "judge_d0_base", "d1": "judge_d1_instruct", "d1_long": "judge_d1_instruct_long"}
    if lenient:
        for (pid, k), text in d["samples"].items():
            if d["correct"][(pid, k)]:
                continue
            num = final_line_number(text)
            try:
                gold = float(str(d["rows"][pid]["reference_answer"]).replace(",", ""))
                d["correct"][(pid, k)] = num is not None and abs(float(num) - gold) < 1e-6
            except ValueError:
                pass
    d["parsed"] = {key: cc.parse(t) for key, t in d["samples"].items()}
    for arm, f in files.items():
        if (RUNS / f"{f}.jsonl").exists():
            recs = cc.read_jsonl(RUNS / f"{f}.jsonl")
            d["judges"][arm] = {tuple(r["task"]): r["output"] for r in recs}
            d["meta"][arm] = {tuple(r["task"]): r for r in recs}
    d["judges"]["base"] = next(iter(d["judges"].values()))  # one_right_pairs keys off "base"
    return d


def main() -> None:
    res = {"schema_version": "paper_t3_t5_v1", "post_hoc": True}
    for scoring in ("strict", "lenient"):
        set_scoring(scoring == "lenient")
        arms, out = {}, {}
        for solutions in ("self_correction", "instruct"):
            d = dataset(solutions, scoring == "lenient")
            t = cc.judge_tables(d)
            out[f"{solutions}_solutions"] = {"one_right_pairs": t["_pairs"],
                                             "s1_accuracy": float(np.mean([d["correct"][(p, 1)] for p in d["rows"]]))}
            for arm in d["judges"]:
                if arm == "base":
                    continue
                arms[arm] = t[arm]
                inv = t[arm]["B1_position_table"]
                out[arm] = {**{k: t[arm][k] for k in KEEP},
                            "invented_rate": (inv["right_is_A"]["picks_invented"] + inv["right_is_B"]["picks_invented"]) / 2,
                            "token_cap_rate": float(np.mean([r["hit_token_cap"] for r in d["meta"][arm].values()]))}
        boots = {k: boot_means(v["_tie"], v["_tg"], 20261080 + i) for i, (k, v) in enumerate(arms.items())}
        diffs = {}
        for a, b in (("t3", "d1_long"), ("t1", "d1"), ("t1", "d0"), ("t3", "d0"), ("t5", "d0"), ("t5", "t1"), ("t5", "t3")):
            if a in boots and b in boots:
                diffs[f"{a}_minus_{b}"] = {"tie_break_difference": float(arms[a]["B2_tie_break"] - arms[b]["B2_tie_break"]),
                                           "ci": [float(x) for x in np.quantile(boots[a] - boots[b], [0.025, 0.975])]}
        out["differences"] = diffs
        res[scoring] = out
    set_scoring(False)
    OUT.write_text(json.dumps(cc.strip(res), indent=2) + "\n", encoding="utf-8", newline="\n")
    for scoring in ("strict", "lenient"):
        print("==", scoring)
        for k, v in res[scoring].items():
            if k == "differences":
                for dk, dv in v.items():
                    print("  ", dk, round(dv["tie_break_difference"], 3), [round(x, 3) for x in dv["ci"]])
            else:
                print(" ", k, {m: (round(x, 3) if isinstance(x, float) else x) for m, x in v.items() if m != "B1_skill_index_ci"})


if __name__ == "__main__":
    main()
