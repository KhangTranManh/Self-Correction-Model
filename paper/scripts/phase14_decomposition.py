"""Phase 14: where a judge's single-order accuracy goes (position lock, invented answers,
consistent mistakes) and the log-prob position bias, per model x dataset cell (CPU, descriptive).

For every one-right judged pair (s1, sk) and each judge (base, base_c), the two single-order
judgments ("ab": s1 shown as A, "ba": s1 shown as B) are classified:
  consistent right   both orders pick the right solution
  consistent wrong   both orders pick the wrong solution
  locked to A / B    both orders pick the same POSITION (one right, one wrong): position bias
  invented, + right  one order invents, the other picks the right solution
  invented, - right  one order invents, the other picks the wrong one, or both invent
Identities checked in code:
  single-order accuracy = CR + 0.5 (lock A + lock B) + 0.5 (invented, + right)
  tie-break             = CR + 0.5 (lock A + lock B + invented, + right + invented, - right)
  tie-break - single    = 0.5 (invented, - right)            (the invented-answer term)
Log-probs (base_c): m = log P(A) - log P(B) at the verdict token; coverage, missing reasons
(no "Verdict: Solution" line: with / without the 768-token cap), bias log-odds, single-order
and order-averaged accuracy of the log-probs. Uses the same answer reading as Phase 14
(lenient = project parser) and the same pair selection.

Usage: python paper/scripts/phase14_decomposition.py [--root outputs/phase14_remote/outputs/phase14_v1]
Writes paper/results/phase14_decomposition.json and .md (every cell whose judges are complete).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _to_sympy  # noqa: E402
from phase13.scripts.constrained import _VERDICT, pick  # noqa: E402
from phase14.scripts import generate  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase14.scripts.config import COMPARISON, DATASETS, JUDGED, MODELS, PRIMARY_CAP  # noqa: E402

CATS = ("consistent_right", "consistent_wrong", "locked_A", "locked_B", "invented_with_right",
        "invented_without_right")
NAMES = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "original_solver": "Self_Correction",
         "qwen25_instruct": "Qwen2.5-7B-Instruct"}
DS = {"asdiv": "ASDiv", "math13": "MATH 1–3", "gsm8k13": "GSM8K", "svamp12": "SVAMP"}


def classify(p_ab: str, p_ba: str, first_right: bool) -> str:
    """p_ab, p_ba in {"A", "B", "invented"}; ab shows s1 as A, ba shows s1 as B."""
    right_ab = "A" if first_right else "B"
    right_ba = "B" if first_right else "A"
    hit_ab, hit_ba = p_ab == right_ab, p_ba == right_ba
    if "invented" in (p_ab, p_ba):
        return "invented_with_right" if (hit_ab or hit_ba) else "invented_without_right"
    if hit_ab and hit_ba:
        return "consistent_right"
    if not hit_ab and not hit_ba:
        return "consistent_wrong"
    return "locked_A" if p_ab == p_ba == "A" else "locked_B"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def shapley(recs: list[tuple[str, str, str]]) -> dict:
    """Exploratory (post-hoc) attribution of the single-order accuracy gap.

    recs: (problem id, right position "A"/"B", pick "A"/"B"/"invented") per single-order judgment.
    Model per cell: an invented answer with probability pi_A / pi_B (by right position); otherwise
    pick A with probability sigmoid(beta + delta) when A is right and sigmoid(beta - delta) when B
    is right (beta = position bias toward A, delta = skill). Fitted exactly from the 2 x 3 table
    (0.5 smoothing). Reference R = sigmoid(delta) = accuracy of the same judge with no position bias
    (beta = 0) and no invented answers (pi = 0). Gap R - actual is split between bias and invention
    by Shapley values: each factor's effect averaged over both removal orders; the two parts sum
    exactly to the gap. Removing bias sets beta = 0 and replaces pi_A, pi_B by their mean.
    """
    n = Counter((right, p) for _, right, p in recs)
    tot_a = sum(n[("A", x)] for x in ("A", "B", "invented"))
    tot_b = sum(n[("B", x)] for x in ("A", "B", "invented"))
    a = (n[("A", "A")] + 0.5) / (n[("A", "A")] + n[("A", "B")] + 1)
    b = (n[("B", "A")] + 0.5) / (n[("B", "A")] + n[("B", "B")] + 1)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    beta, delta = (logit(a) + logit(b)) / 2, (logit(a) - logit(b)) / 2
    pi_a, pi_b = n[("A", "invented")] / tot_a, n[("B", "invented")] / tot_b
    pi_mean = (pi_a + pi_b) / 2
    acc = lambda bt, pa, pb: 0.5 * ((1 - pa) * sigmoid(bt + delta) + (1 - pb) * sigmoid(delta - bt))  # noqa: E731
    actual = acc(beta, pi_a, pi_b)
    no_bias, no_inv, reference = acc(0.0, pi_mean, pi_mean), acc(beta, 0.0, 0.0), sigmoid(delta)
    bias_part = 0.5 * ((no_bias - actual) + (reference - no_inv))
    inv_part = 0.5 * ((no_inv - actual) + (reference - no_bias))
    observed = (n[("A", "A")] + n[("B", "B")]) / (tot_a + tot_b)
    return {"reference_no_bias_no_invention": float(reference), "model_single_order": float(actual),
            "observed_single_order": float(observed), "gap": float(reference - actual),
            "shapley_position_bias": float(bias_part), "shapley_invented": float(inv_part),
            "beta_bias_toward_A": float(beta), "delta_skill": float(delta), "pi_A": pi_a, "pi_B": pi_b}


def shapley3(recs: list[tuple]) -> dict:
    """Three-factor version: the third outcome split into "cut" (output hit the token cap) and
    "invented" (finished output, no valid choice). recs: (pid, right "A"/"B", outcome) with outcome
    in "A", "B", "cut", "invented". Model as in shapley() with pi_cut and pi_inv per right position.
    Factors: bias (beta -> 0, both pi's averaged over positions), cut (pi_cut -> 0), invented
    (pi_inv -> 0). Shapley value of each factor = mean marginal gain over all 6 removal orders;
    the three parts sum exactly to R - modelled single-order accuracy."""
    from itertools import permutations
    n = Counter((right, o) for _, right, o in recs)
    tot = {p: sum(n[(p, o)] for o in ("A", "B", "cut", "invented")) for p in "AB"}
    a = (n[("A", "A")] + 0.5) / (n[("A", "A")] + n[("A", "B")] + 1)
    b = (n[("B", "A")] + 0.5) / (n[("B", "A")] + n[("B", "B")] + 1)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    beta, delta = (logit(a) + logit(b)) / 2, (logit(a) - logit(b)) / 2
    cut = {p: n[(p, "cut")] / tot[p] for p in "AB"}
    inv = {p: n[(p, "invented")] / tot[p] for p in "AB"}

    def value(removed: frozenset) -> float:
        bt = 0.0 if "bias" in removed else beta
        c, i = dict(cut), dict(inv)
        if "bias" in removed:
            c = {p: (cut["A"] + cut["B"]) / 2 for p in "AB"}
            i = {p: (inv["A"] + inv["B"]) / 2 for p in "AB"}
        if "cut" in removed:
            c = {p: 0.0 for p in "AB"}
        if "invented" in removed:
            i = {p: 0.0 for p in "AB"}
        return 0.5 * ((1 - c["A"] - i["A"]) * sigmoid(bt + delta) + (1 - c["B"] - i["B"]) * sigmoid(delta - bt))

    factors = ("bias", "cut", "invented")
    phi = {f: 0.0 for f in factors}
    orders = list(permutations(factors))
    for order in orders:
        done = frozenset()
        for f in order:
            phi[f] += (value(done | {f}) - value(done)) / len(orders)
            done = done | {f}
    actual, reference = value(frozenset()), value(frozenset(factors))
    assert abs(sum(phi.values()) - (reference - actual)) < 1e-9
    return {"reference": float(reference), "model_single_order": float(actual), "gap": float(reference - actual),
            "shapley_position_bias": float(phi["bias"]), "shapley_cut_at_cap": float(phi["cut"]),
            "shapley_invented_uncut": float(phi["invented"]),
            "share_cut": (n[("A", "cut")] + n[("B", "cut")]) / (tot["A"] + tot["B"]),
            "share_invented_uncut": (n[("A", "invented")] + n[("B", "invented")]) / (tot["A"] + tot["B"])}


def shapley3_ci(recs: list[tuple], reps: int = 2000, seed: int = 20261014) -> dict:
    by = {}
    for r in recs:
        by.setdefault(r[0], []).append(r)
    ids = sorted(by)
    rng = np.random.default_rng(seed)
    keys = ("gap", "shapley_position_bias", "shapley_cut_at_cap", "shapley_invented_uncut")
    draws = {k: [] for k in keys}
    for _ in range(reps):
        s = shapley3([r for i in rng.choice(len(ids), len(ids)) for r in by[ids[i]]])
        for k in keys:
            draws[k].append(s[k])
    return {**shapley3(recs), "ci": {k: [float(x) for x in np.quantile(draws[k], [0.025, 0.975])] for k in keys}}


def shapley_ci(recs: list[tuple[str, str, str]], reps: int = 2000, seed: int = 20261013) -> dict:
    by = {}
    for r in recs:
        by.setdefault(r[0], []).append(r)
    ids = sorted(by)
    rng = np.random.default_rng(seed)
    keys = ("reference_no_bias_no_invention", "observed_single_order", "gap", "shapley_position_bias",
            "shapley_invented")
    draws = {k: [] for k in keys}
    for _ in range(reps):
        sample = [r for i in rng.choice(len(ids), len(ids)) for r in by[ids[i]]]
        s = shapley(sample)
        for k in keys:
            draws[k].append(s[k])
    point = shapley(recs)
    return {**point, "ci": {k: [float(x) for x in np.quantile(draws[k], [0.025, 0.975])] for k in keys},
            "bootstrap": f"{reps} resamples of problems"}


def cell(model: str, dataset: str) -> dict | None:
    base = generate.folder(model, dataset, "main")
    if not all((base / f"judge_{j}" / "summary.json").exists() for j in ("base", "base_c")):
        return None
    rows = generate.problems(dataset, "main")
    samples = generate.load_samples(model, dataset, "main")
    judges = {j: generate.completed(base, f"judge_{j}") for j in ("base", "base_c")}
    for j in ("base_1536", "base_c_1536"):  # token-cap sensitivity re-judging (ops/judge_long.py), if present
        if (base / f"judge_{j}" / "summary.json").exists():
            judges[j] = generate.completed(base, f"judge_{j}")
    ref = {r["id"]: _to_sympy(str(r["reference_answer"])) for r in rows}
    out = {}
    for judge, texts in judges.items():
        constrained = judge.startswith("base_c")
        cap = 1536 if judge.endswith("_1536") else PRIMARY_CAP
        counts, judgments, recs, source, recs3 = Counter(), Counter(), [], Counter(), []
        lp = {"pairs": 0, "covered": 0, "missing_no_verdict_cap": 0, "missing_no_verdict_other": 0,
              "bias": [], "single": [], "averaged": []}
        for row in rows:
            pid = row["id"]
            p1 = parse(samples[(pid, 1)]["output"])
            for k in JUDGED:
                pk = parse(samples[(pid, k)]["output"])
                if same(p1, pk):
                    continue
                first_ok, other_ok = same(p1, ref[pid]), same(pk, ref[pid])
                if first_ok == other_ok:
                    continue
                ab, ba = texts[(pid, k, "ab")], texts[(pid, k, "ba")]
                p_ab = pick(ab["output"], p1, pk, constrained)
                p_ba = pick(ba["output"], pk, p1, constrained)
                counts[classify(p_ab, p_ba, first_ok)] += 1
                for row_j, p in ((ab, p_ab), (ba, p_ba)):  # where each single-order choice came from
                    capped = bool(row_j.get("hit_token_cap")) or row_j.get("generated_tokens", 0) >= cap
                    if constrained and _VERDICT.findall(row_j["output"]):
                        source["verdict_line"] += 1
                    elif p in ("A", "B"):
                        source["final_answer_match" + ("" if not constrained else "_no_verdict_line")] += 1
                    else:
                        source[("no_valid_choice" if constrained else "invented_answer")
                               + ("_cap" if capped else "_no_cap")] += 1
                for row_j, p, right in ((ab, p_ab, "A" if first_ok else "B"), (ba, p_ba, "B" if first_ok else "A")):
                    judgments["right" if p == right else "invented" if p == "invented" else "wrong"] += 1
                    recs.append((pid, right, p if p in ("A", "B") else "invented"))
                    capped = bool(row_j.get("hit_token_cap")) or row_j.get("generated_tokens", 0) >= cap
                    recs3.append((pid, right, p if p in ("A", "B") else "cut" if capped else "invented"))
                if constrained:
                    lp["pairs"] += 1
                    m = {}
                    for order, row_j in (("ab", ab), ("ba", ba)):
                        v = row_j.get("verdict_logprobs")
                        if v and v.get("A") is not None and v.get("B") is not None:
                            m[order] = v["A"] - v["B"]
                        else:
                            capped = row_j.get("hit_token_cap") or row_j.get("generated_tokens", 0) >= cap
                            lp["missing_no_verdict_cap" if capped else "missing_no_verdict_other"] += 1
                    if len(m) == 2:
                        lp["covered"] += 1
                        lp["bias"].append((m["ab"] + m["ba"]) / 2)
                        # m > 0 picks A; first right means A right in ab and B right in ba
                        lp["single"].append(((m["ab"] > 0) == first_ok) * 0.5 + ((m["ba"] < 0) == first_ok) * 0.5)
                        lp["averaged"].append(float(((m["ab"] - m["ba"]) > 0) == first_ok))
        n = sum(counts.values())
        if n == 0:
            continue
        share = {c: counts[c] / n for c in CATS}
        single = judgments["right"] / sum(judgments.values())
        tie = share["consistent_right"] + 0.5 * (1 - share["consistent_right"] - share["consistent_wrong"])
        assert abs(single - (share["consistent_right"] + 0.5 * (share["locked_A"] + share["locked_B"])
                             + 0.5 * share["invented_with_right"])) < 1e-9
        assert abs((tie - single) - 0.5 * share["invented_without_right"]) < 1e-9
        entry = {"one_right_pairs": n, "pair_share": share, "single_order_accuracy": single, "tie_break": tie,
                 "tie_minus_single": tie - single,
                 "judgment_share": {k: v / sum(judgments.values()) for k, v in judgments.items()},
                 "exploratory_shapley": shapley_ci(recs),
                 "exploratory_shapley3": shapley3_ci(recs3),
                 "token_cap": cap,
                 "choice_source": {k: v / sum(source.values()) for k, v in sorted(source.items())},
                 "third_outcome_name": "no valid choice (no Verdict line and the final answer matches "
                                       "neither solution)" if constrained else "invented answer (final "
                                       "answer matches neither solution)"}
        if constrained and lp["pairs"]:
            entry["logprob"] = {"coverage": lp["covered"] / lp["pairs"],
                                "missing_judgments_no_verdict_cap": lp["missing_no_verdict_cap"],
                                "missing_judgments_no_verdict_other": lp["missing_no_verdict_other"],
                                "bias_logodds": float(np.mean(lp["bias"])) if lp["bias"] else None,
                                "single_order_accuracy": float(np.mean(lp["single"])) if lp["single"] else None,
                                "order_averaged_accuracy": float(np.mean(lp["averaged"])) if lp["averaged"] else None}
        out[judge] = entry
    return out


def markdown(report: dict) -> str:
    lines = ["# Phase 14 — phân rã bịa và bias theo ô", "",
             "**Khám phá, post-hoc — không phải endpoint.** Claim đã khóa dựa trên E1, E2 (`phase14/RUN_STATUS.md`).",
             "", "## 1. Phân rã Shapley khoảng cách một thứ tự (KTC bootstrap theo bài, 2000 lần)", "",
             "Mô hình mỗi ô: bịa với xác suất π (theo vị trí lời giải đúng); nếu không, chọn A với xác suất "
             "σ(β + δ) khi A đúng, σ(β − δ) khi B đúng (β = thiên lệch về A, δ = kỹ năng), khớp đúng bảng 2 × 3. "
             "**Mốc tham chiếu R = σ(δ): accuracy của chính judge đó nếu không thiên lệch vị trí (β = 0) và "
             "không bịa (π = 0).** Khoảng cách R − accuracy một thứ tự được chia cho bias và bịa bằng giá trị "
             "Shapley (trung bình hai thứ tự gỡ); hai phần cộng đúng bằng khoảng cách. Gỡ bias: β = 0 và π_A, π_B "
             "thay bằng trung bình.", "",
             "**Kết cục thứ ba (ngoài chọn A, chọn B) khác nhau theo judge, không gộp:** với `base` là **bịa đáp "
             "số** (đáp số cuối không khớp lời giải nào); với `base_c` là **không đưa ra lựa chọn hợp lệ** (không "
             "có dòng \"Verdict: Solution A/B\" và đáp số cuối cũng không khớp lời giải nào — phần lớn do bị cắt ở "
             "trần 768 token, xem mục 3).", "",
             "| Mô hình | Dataset | Judge | Một thứ tự (quan sát) [KTC] | R: không bias, không có kết cục thứ ba [KTC] | "
             "Khoảng cách [KTC] | Phần do bias [KTC] | Phần do kết cục thứ ba [KTC] | β | δ |",
             "|---|---|---|---|---|---|---|---|---:|---:|"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    ci = lambda e, k: f"[{pct(e['ci'][k][0])}; {pct(e['ci'][k][1])}]"  # noqa: E731
    for key, cells in report["cells"].items():
        model, dataset = key.split("/")
        for judge, e in cells.items():
            s = e["exploratory_shapley"]
            lines.append(f"| {NAMES.get(model, model)} | {DS.get(dataset, dataset)} | {judge} | "
                         f"{pct(s['observed_single_order'])} {ci(s, 'observed_single_order')} | "
                         f"{pct(s['reference_no_bias_no_invention'])} {ci(s, 'reference_no_bias_no_invention')} | "
                         f"{pct(s['gap'])} {ci(s, 'gap')} | {pct(s['shapley_position_bias'])} {ci(s, 'shapley_position_bias')} | "
                         f"{pct(s['shapley_invented'])} {ci(s, 'shapley_invented')} | {s['beta_bias_toward_A']:+.2f} | "
                         f"{s['delta_skill']:.2f} |")
    lines += ["", "Chú thích: mọi số là điểm phần trăm accuracy một thứ tự; số đứng trước là ước lượng điểm, "
              "[a; b] là KTC 95% bootstrap theo bài. Phần do bias + phần do kết cục thứ ba = khoảng cách (ước lượng "
              "điểm). \"Kết cục thứ ba\" = bịa đáp số (`base`) hoặc không có lựa chọn hợp lệ (`base_c`). Phần do bias "
              "nhỏ khi β nhỏ so với δ vì thiên lệch đối xứng chỉ làm giảm accuracy ở bậc hai "
              "(σ(δ) − ½[σ(δ+β) + σ(δ−β)]).", "",
              "## 1b. Ba thành phần: bias, bị cắt ở trần, bịa thật (Shapley trung bình qua 6 thứ tự gỡ)", "",
              "Kết cục thứ ba tách theo việc output có chạm trần token hay không: **bị cắt** (chạm trần, chưa kịp đưa ra "
              "lựa chọn) và **bịa thật / không lựa chọn hợp lệ dù đã viết xong** (không chạm trần). Ba phần cộng đúng "
              "bằng khoảng cách R − accuracy một thứ tự của mô hình. Số đứng trước là ước lượng điểm, [a; b] là KTC 95% "
              "bootstrap theo bài. Hai cột tỷ lệ là tỷ lệ lần chấm.", "",
              "| Mô hình | Dataset | Judge (trần) | Khoảng cách [KTC] | Phần do bias [KTC] | Phần do bị cắt [KTC] | "
              "Phần do bịa thật [KTC] | Tỷ lệ bị cắt | Tỷ lệ bịa thật |",
              "|---|---|---|---|---|---|---|---:|---:|"]
    for key, cells in report["cells"].items():
        model, dataset = key.split("/")
        for judge, e in cells.items():
            s = e["exploratory_shapley3"]
            c = lambda k: f"{pct(s[k])} [{pct(s['ci'][k][0])}; {pct(s['ci'][k][1])}]"  # noqa: E731
            lines.append(f"| {NAMES.get(model, model)} | {DS.get(dataset, dataset)} | {judge} ({e['token_cap']}) | "
                         f"{c('gap')} | {c('shapley_position_bias')} | {c('shapley_cut_at_cap')} | "
                         f"{c('shapley_invented_uncut')} | {pct(s['share_cut'])}% | {pct(s['share_invented_uncut'])}% |")
    lines += ["", "## 2. Loại cặp (mô tả, không phải phép gán nguyên nhân)", "",
              "| Mô hình | Dataset | Judge | Cặp | Đúng cả hai | Sai cả hai | Khóa A | Khóa B | Bịa (+đúng) | "
              "Bịa (không đúng) | Một thứ tự | Tie-break |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, cells in report["cells"].items():
        model, dataset = key.split("/")
        for judge, e in cells.items():
            s = e["pair_share"]
            lines.append(f"| {NAMES.get(model, model)} | {DS.get(dataset, dataset)} | {judge} | {e['one_right_pairs']} | "
                         f"{pct(s['consistent_right'])}% | {pct(s['consistent_wrong'])}% | {pct(s['locked_A'])}% | "
                         f"{pct(s['locked_B'])}% | {pct(s['invented_with_right'])}% | {pct(s['invented_without_right'])}% | "
                         f"{pct(e['single_order_accuracy'])}% | {e['tie_break']:.3f} |")
    lines += ["", "Một thứ tự = đúng cả hai + ½·khóa + ½·bịa(+đúng); tie-break − một thứ tự = ½·bịa(không đúng) "
              "(kiểm tra trong code). Với `base_c`, \"bịa\" ở bảng này đọc là \"không có lựa chọn hợp lệ\". Một cặp bị "
              "khóa vị trí có thể do bias hoặc do kỹ năng yếu cộng nhiễu, nên bảng này không dùng để gán phần mất; "
              "phép gán là mục 1.", "",
              "## 3. Nguồn của mỗi lựa chọn một thứ tự (tỷ lệ lần chấm, cặp đúng-một)", "",
              "| Mô hình | Dataset | Judge | Dòng Verdict | Khớp đáp số (không có dòng Verdict) | Khớp đáp số | "
              "Không lựa chọn hợp lệ: chạm trần 768 / không chạm | Bịa đáp số: chạm trần 768 / không chạm |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for key, cells in report["cells"].items():
        model, dataset = key.split("/")
        for judge, e in cells.items():
            s = e.get("choice_source", {})
            g = lambda k: pct(s.get(k, 0.0)) + "%"  # noqa: E731
            lines.append(f"| {NAMES.get(model, model)} | {DS.get(dataset, dataset)} | {judge} | "
                         f"{g('verdict_line') if judge == 'base_c' else '—'} | "
                         f"{g('final_answer_match_no_verdict_line') if judge == 'base_c' else '—'} | "
                         f"{g('final_answer_match') if judge == 'base' else '—'} | "
                         f"{(g('no_valid_choice_cap') + ' / ' + g('no_valid_choice_no_cap')) if judge == 'base_c' else '—'} | "
                         f"{(g('invented_answer_cap') + ' / ' + g('invented_answer_no_cap')) if judge == 'base' else '—'} |")
    lines += ["", "## Log-prob của judge base_c (token verdict)", "",
              "| Mô hình | Dataset | Coverage cặp | Lần chấm thiếu: chạm trần 768 / không ghi verdict | "
              "Bias log-odds (> 0 = A) | Một thứ tự | Trung bình hai thứ tự |",
              "|---|---|---:|---|---:|---:|---:|"]
    for key, cells in report["cells"].items():
        model, dataset = key.split("/")
        lp = cells.get("base_c", {}).get("logprob")
        if not lp:
            continue
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"| {NAMES.get(model, model)} | {DS.get(dataset, dataset)} | {pct(lp['coverage'])}% | "
                     f"{lp['missing_judgments_no_verdict_cap']} / {lp['missing_judgments_no_verdict_other']} | "
                     f"{f(lp['bias_logodds'])} | {f(lp['single_order_accuracy'])} | {f(lp['order_averaged_accuracy'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/phase14_remote/outputs/phase14_v1")
    args = parser.parse_args()
    generate.OUT = args.root.resolve()  # generate.folder() reads this module-level path
    report = {"schema_version": "paper_phase14_decomposition_v1", "descriptive": True,
              "root": str(args.root), "cells": {}}
    for model in MODELS:
        for dataset in (*DATASETS, *COMPARISON):
            try:
                result = cell(model, dataset)
            except (FileNotFoundError, KeyError, RuntimeError):
                result = None
            if result:
                report["cells"][f"{model}/{dataset}"] = result
                print(model, dataset, {j: round(e["single_order_accuracy"], 3) for j, e in result.items()}, flush=True)
    out = ROOT / "paper/results/phase14_decomposition"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    out.with_suffix(".md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
