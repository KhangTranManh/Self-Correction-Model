"""Paper tables for Phase 14 (CPU, exploratory; rules in phase14/RUN_STATUS.md, amendment 2026-10-15).

Computes, from the saved outputs only (paper/scripts/phase14_data.py):
  1a  seven-way classification of every one-right single-order judgment (base, base_c)
  1b  stratified sample of class-2 (format error) judgments for the rescued audit (CSV)
  1c  four-factor Shapley split of the single-order gap: bias / format / cut / no choice
  2a  bounds for the loss due to invalid choices; 2b probit link; 2c model-free identity + pi_A - pi_B
  3   practical table: accuracy, calls, tokens per strategy, paired CIs vs vote@3 / vote@5, oracle,
      share of disagreements where both are wrong
  5   counts per cell
Writes paper/results/phase14_paper_tables.json and paper/results/rescued_audit_sample.csv.
The Markdown tables, LaTeX tables and figures are rendered from the JSON by phase14_render_tables.py.
"""
from __future__ import annotations

from collections import Counter
import csv
from itertools import permutations
import json
from math import erf, sqrt
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper/scripts"))
from paper.scripts import phase14_data as P  # noqa: E402
from phase13.scripts.constrained import pick  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.answers import vote  # noqa: E402
import phase14_truncation as T  # noqa: E402

OUT = ROOT / "paper/results"
MODELS = ("llama31_8b", "mistral_7b", "original_solver", "qwen25_instruct")
GRID = ("asdiv", "math13", "gsm8k13", "svamp12")
EXTRA = ("bridge13", "gsm8k13_report", "svamp12_report")
NAMES = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "original_solver": "Self_Correction",
         "qwen25_instruct": "Qwen2.5-7B-Instruct"}
DS = {"asdiv": "ASDiv", "math13": "MATH 1–3", "gsm8k13": "GSM8K", "svamp12": "SVAMP", "bridge13": "bridge",
      "gsm8k13_report": "GSM8K báo cáo", "svamp12_report": "SVAMP báo cáo"}
CLASSES = {1: "chọn hợp lệ", 2: "lỗi định dạng", 3: "bị cắt khi lặp vòng", 4: "bị cắt, khác",
           5: "tự giải lại", 6: "đáp số thứ ba", 7: "không có đáp số"}
REPS = 2000
SEED = 20261015


def cells():
    for m in MODELS:
        for d in (*GRID, *EXTRA):
            if (m, d) in set(P.cells()):
                yield m, d


def classify(row: dict, shown_a, shown_b, constrained: bool) -> tuple[int, str]:
    text = row["output"]
    p = pick(text, shown_a, shown_b, constrained)
    if p in ("A", "B"):
        return 1, p
    r = parse(text, "rescued")
    ma, mb = same(r, shown_a), same(r, shown_b)
    if ma != mb:
        return 2, "A" if ma else "B"
    capped = row.get("hit_token_cap")
    if capped:
        return (3 if T.label(text, True) == "b" else 4), "x"
    if capped is None and "generated_tokens" not in row:   # report cells: no cap information
        pass
    if T.label(text, True) == "d":
        return 5, "x"
    if parse(text) is not None or r is not None:
        return 6, "x"
    return 7, "x"


def judgments(model: str, ds: str, judge: str):
    """[(pid, right_pos, class, pick)] for one-right pairs, both orders."""
    s, j = P.samples(model, ds), P.judgments(model, ds, judge)
    out = []
    for pid, k, first_ok in P.one_right_pairs(model, ds):
        p1, pk = parse(s[(pid, 1)]["output"]), parse(s[(pid, k)]["output"])
        for order, (a, b) in (("ab", (p1, pk)), ("ba", (pk, p1))):
            right = ("A" if first_ok else "B") if order == "ab" else ("B" if first_ok else "A")
            c, pk_ = classify(j[(pid, k, order)], a, b, judge == "base_c")
            out.append((pid, right, c, pk_))
    return out


def boot_groups(recs):
    by = {}
    for r in recs:
        by.setdefault(r[0], []).append(r)
    return list(by.values())


def boot(recs, stat, reps=REPS, seed=SEED):
    groups = boot_groups(recs)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        sample = [r for i in rng.integers(0, len(groups), len(groups)) for r in groups[i]]
        v = stat(sample)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


# ---------------------------------------------------------------- decomposition (1c, 2a-2c)
def sig(x):
    return 1 / (1 + np.exp(-x))


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def phi_inv(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


FACTORS = ("bias", "format", "cut", "nochoice")
GROUP = {2: "format", 3: "cut", 4: "cut", 5: "nochoice", 6: "nochoice", 7: "nochoice"}


def decompose(recs, link: str = "logit") -> dict:
    n = Counter((r[1], r[2] if r[2] != 1 else r[3]) for r in recs)   # (right_pos, "A"/"B"/class)
    tot = {p: sum(v for (q, _), v in n.items() if q == p) for p in "AB"}
    a = (n[("A", "A")] + 0.5) / (n[("A", "A")] + n[("A", "B")] + 1)
    b = (n[("B", "A")] + 0.5) / (n[("B", "A")] + n[("B", "B")] + 1)
    f, finv = (sig, lambda p: np.log(p / (1 - p))) if link == "logit" else (phi, phi_inv)
    beta, delta = (finv(a) + finv(b)) / 2, (finv(a) - finv(b)) / 2
    pi = {(p, g): sum(v for (q, c), v in n.items() if q == p and GROUP.get(c) == g) / tot[p]
          for p in "AB" for g in FACTORS[1:]}

    def value(removed):
        bt = 0.0 if "bias" in removed else beta
        pp = dict(pi)
        if "bias" in removed:
            pp = {(p, g): (pi[("A", g)] + pi[("B", g)]) / 2 for p in "AB" for g in FACTORS[1:]}
        for g in FACTORS[1:]:
            if g in removed:
                for p in "AB":
                    pp[(p, g)] = 0.0
        inval = {p: sum(pp[(p, g)] for g in FACTORS[1:]) for p in "AB"}
        return 0.5 * ((1 - inval["A"]) * f(bt + delta) + (1 - inval["B"]) * f(delta - bt))

    phi_v = {g: 0.0 for g in FACTORS}
    orders = list(permutations(FACTORS))
    for order in orders:
        done = frozenset()
        for g in order:
            phi_v[g] += (value(done | {g}) - value(done)) / len(orders)
            done = done | {g}
    actual, ref = value(frozenset()), value(frozenset(FACTORS))
    assert abs(sum(phi_v.values()) - (ref - actual)) < 1e-9
    valid = [r for r in recs if r[2] == 1]
    pi_tot = 1 - len(valid) / len(recs)
    acc_ab = sum(r[3] == r[1] for r in valid) / max(1, len(valid))
    single = sum(r[2] == 1 and r[3] == r[1] for r in recs) / len(recs)
    pa = sum(r[2] != 1 for r in recs if r[1] == "A") / max(1, sum(r[1] == "A" for r in recs))
    pb = sum(r[2] != 1 for r in recs if r[1] == "B") / max(1, sum(r[1] == "B" for r in recs))
    inv_part = phi_v["format"] + phi_v["cut"] + phi_v["nochoice"]
    return {"R": float(ref), "gap": float(ref - actual), **{f"phi_{g}": float(v) for g, v in phi_v.items()},
            "beta": float(beta), "delta": float(delta), "pi_invalid": pi_tot, "acc_valid": acc_ab,
            "single_order": single, "identity_gap": single - (1 - pi_tot) * acc_ab,
            "lower_invalid": pi_tot * 0.5, "upper_invalid": pi_tot * acc_ab, "upper_invalid_R": pi_tot * float(ref),
            "invalid_part": inv_part, "pi_A_minus_pi_B": pa - pb}


# ---------------------------------------------------------------- practical (3)
def practical(model: str, ds: str):
    rows = P.problems(ds)
    s, ok = P.samples(model, ds), P.correct(model, ds)
    jb = P.judgments(model, ds, "base")
    per = []
    for r in rows:
        pid = r["id"]
        texts = [s[(pid, k)]["output"] for k in range(1, 6)]
        tok = [min(s[(pid, k)].get("generated_tokens", 0), 768) for k in range(1, 6)]
        vals = [parse(t) for t in texts]
        c = [ok[(pid, k)] for k in range(1, 6)]
        maj3 = c[vote(texts[:3], [0, 1, 2])]
        maj5 = c[vote(texts, [0, 1, 2, 3, 4])]
        agree = same(vals[0], vals[1])
        row = {"pid": pid, "keep": c[0], "random": (c[0] + c[1]) / 2 if not agree else c[0], "maj3": maj3,
               "maj5": maj5, "agree_gated": c[0] if agree else maj3, "oracle": c[0] if agree else max(c[0], c[1]),
               "disagree": not agree, "both_wrong": (not agree) and not c[0] and not c[1]}
        calls = {"keep": 1, "random": 2, "maj3": 3, "maj5": 5, "agree_gated": 2 if agree else 3, "oracle": 2}
        tokens = {"keep": tok[0], "random": tok[0] + tok[1], "maj3": sum(tok[:3]), "maj5": sum(tok),
                  "agree_gated": tok[0] + tok[1] + (0 if agree else tok[2]), "oracle": tok[0] + tok[1]}
        for mode in ("lenient", "rescued"):
            if agree:
                row[f"judge_{mode}"], row[f"selfcheck_{mode}"] = c[0], c[0]
                calls[f"judge_{mode}"] = calls[f"selfcheck_{mode}"] = 2
                tokens[f"judge_{mode}"] = tokens[f"selfcheck_{mode}"] = tok[0] + tok[1]
                continue
            picks = {}
            jt = 0
            for order, (a, b) in (("ab", (vals[0], vals[1])), ("ba", (vals[1], vals[0]))):
                jr = jb.get((pid, 2, order))
                if jr is None:
                    picks[order] = None
                    continue
                jt += jr.get("generated_tokens", 0) or 0
                text = jr["output"]
                p = pick(text, a, b, False)
                if p not in ("A", "B") and mode == "rescued":
                    rv = parse(text, "rescued")
                    ma, mb = same(rv, a), same(rv, b)
                    p = ("A" if ma else "B") if ma != mb else "x"
                first = (p == "A") if order == "ab" else (p == "B")
                picks[order] = None if p not in ("A", "B") else ("first" if first else "other")
            consistent = picks.get("ab") is not None and picks.get("ab") == picks.get("ba")
            if consistent:
                choice = c[0] if picks["ab"] == "first" else c[1]
                row[f"judge_{mode}"] = row[f"selfcheck_{mode}"] = choice
                calls[f"judge_{mode}"] = calls[f"selfcheck_{mode}"] = 4
                tokens[f"judge_{mode}"] = tokens[f"selfcheck_{mode}"] = tok[0] + tok[1] + jt
            else:
                row[f"judge_{mode}"] = (c[0] + c[1]) / 2
                row[f"selfcheck_{mode}"] = maj3
                calls[f"judge_{mode}"], calls[f"selfcheck_{mode}"] = 4, 5
                tokens[f"judge_{mode}"] = tok[0] + tok[1] + jt
                tokens[f"selfcheck_{mode}"] = tok[0] + tok[1] + jt + tok[2]
        row["calls"], row["tokens"] = calls, tokens
        per.append(row)
    return per


STRATS = ("keep", "random", "judge_lenient", "judge_rescued", "selfcheck_lenient", "selfcheck_rescued",
          "agree_gated", "maj3", "maj5", "oracle")


def paired_ci(per, a, b, seed):
    d = np.array([float(r[a]) - float(r[b]) for r in per])
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(REPS, len(d)))].mean(1)
    return float(d.mean()), [float(x) for x in np.quantile(m, [0.025, 0.975])]


# ---------------------------------------------------------------- main
def main() -> None:
    res = {"schema_version": "paper_phase14_tables_v1", "exploratory": True, "cells": {}}
    audit_pool = {m: [] for m in MODELS}
    for m, d in cells():
        key = f"{m}/{d}"
        entry = {}
        for judge in ("base", "base_c"):
            recs = judgments(m, d, judge)
            share = Counter(r[2] for r in recs)
            n = len(recs)
            cls = {c: {"share": share[c] / n,
                       "ci": boot(recs, lambda x, c=c: sum(r[2] == c for r in x) / len(x))} for c in CLASSES}
            entry[judge] = {"judgments": n, "classes": cls, "decomp": decompose(recs),
                            "decomp_probit": decompose(recs, "probit")}
            dc = entry[judge]["decomp"]
            entry[judge]["decomp_ci"] = {g: boot(recs, lambda x, g=g: decompose(x)[g], reps=500)
                                         for g in ("gap", "phi_bias", "phi_format", "phi_cut", "phi_nochoice",
                                                   "pi_A_minus_pi_B")}
            if judge == "base" and d in GRID:
                s, j = P.samples(m, d), P.judgments(m, d, "base")
                for pid, k, first_ok in P.one_right_pairs(m, d):
                    p1, pk = parse(s[(pid, 1)]["output"]), parse(s[(pid, k)]["output"])
                    for order, (a, b) in (("ab", (p1, pk)), ("ba", (pk, p1))):
                        row = j[(pid, k, order)]
                        c, _ = classify(row, a, b, False)
                        if c == 2:
                            audit_pool[m].append({"model": m, "dataset": d, "task": f"{pid}|{k}|{order}",
                                                  "solution_A_answer": str(a), "solution_B_answer": str(b),
                                                  "rescued_number": str(parse(row["output"], "rescued")),
                                                  "judge_tail": row["output"][-300:].replace("\n", " / ")})
            print(key, judge, n, {c: round(v["share"], 3) for c, v in cls.items()}, flush=True)
        if d in GRID or d == "bridge13":
            per = practical(m, d)
            summ = {}
            for st in STRATS:
                summ[st] = {"accuracy": float(np.mean([float(r[st]) for r in per])),
                            "calls": float(np.mean([r["calls"][st] for r in per])),
                            "tokens": float(np.mean([r["tokens"][st] for r in per]))}
            for st in ("judge_lenient", "judge_rescued", "selfcheck_lenient", "selfcheck_rescued", "agree_gated"):
                summ[st]["vs_maj3"] = paired_ci(per, st, "maj3", 11)
                summ[st]["vs_maj5"] = paired_ci(per, st, "maj5", 12)
            dis = [r for r in per if r["disagree"]]
            entry["practical"] = {"strategies": summ, "problems": len(per),
                                  "disagree_share": len(dis) / len(per),
                                  "both_wrong_share_of_disagreements": sum(r["both_wrong"] for r in dis) / max(1, len(dis))}
        # counts (5)
        s = P.samples(m, d)
        rows = P.problems(d)
        jb = P.judgments(m, d, "base")
        jc = P.judgments(m, d, "base_c")
        orp = P.one_right_pairs(m, d)
        ok = P.correct(m, d)
        judged = {(t[0], t[1]) for t in jb}
        entry["counts"] = {
            "problems": len(rows),
            "s1_ne_s2": sum(not same(parse(s[(r['id'], 1)]["output"]), parse(s[(r['id'], 2)]["output"])) for r in rows),
            "judged_pairs": len(judged), "one_right_pairs": len(orp),
            "both_wrong_pairs": sum(1 for pid, k in judged if not ok[(pid, 1)] and not ok[(pid, k)]),
            "solution_cap_768": (float(np.mean([bool(v.get("hit_primary_cap")) for v in s.values()]))
                                 if any("hit_primary_cap" in v for v in s.values()) else None),
            "judge_cap_base": (float(np.mean([bool(v.get("hit_token_cap")) for v in jb.values()]))
                               if any("hit_token_cap" in v for v in jb.values()) else None),
            "judge_cap_base_c": float(np.mean([bool(v.get("hit_token_cap")) for v in jc.values()]))}
        res["cells"][key] = entry
        (OUT / "phase14_paper_tables.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    # 1b audit sample
    rng = random.Random(SEED)
    with open(OUT / "rescued_audit_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "dataset", "task", "solution_A_answer", "solution_B_answer",
                                            "rescued_number", "judge_tail", "verdict_ok"])
        w.writeheader()
        for m in MODELS:
            pool = audit_pool[m]
            by_ds = {}
            for x in pool:
                by_ds.setdefault(x["dataset"], []).append(x)
            pick_rows = []
            while len(pick_rows) < min(50, len(pool)):
                for ds_ in sorted(by_ds):
                    if by_ds[ds_] and len(pick_rows) < 50:
                        pick_rows.append(by_ds[ds_].pop(rng.randrange(len(by_ds[ds_]))))
            for x in pick_rows:
                w.writerow({**x, "verdict_ok": ""})
    print("done")


if __name__ == "__main__":
    main()
