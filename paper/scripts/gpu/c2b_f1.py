"""C2b and F1 on the judge-prompt hidden states (CPU; run after extract_hidden).

C2b (judges base, p12; train SVAMP, test GSM8K, layer chosen by SVAMP CV as in C4):
  hidden-only probe, surface-only model (C2 features), hidden + surface in one
  model, and the hidden probe's accuracy on GSM8K rows the surface model gets wrong.
F1 (judge p12): pair-level p_first = mean over both orders of P(s1 right);
  confidence = |p_first - 0.5|.
  (a) does low confidence flag both-wrong pairs? AUC of (0.5 - confidence) for
      both-wrong vs one-right among judged disagreements;
  (b) abstention rule on (s1, s2) disagreements: judge consistent -> its verdict;
      inconsistent -> probe pick if confidence >= tau, else vote@3; and a probe-only
      rule (probe pick if confident, else vote@3). tau is chosen on SVAMP from a fixed
      grid and applied once to GSM8K. Compared with self_check_p12, vote@3, vote@5.
Output: outputs/paper/c2b_f1.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from phase9.scripts.analyze import ci  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase12.scripts.analyze import verdict as verdict12  # noqa: E402
from phase13.layer.probe_analysis import LAYER_DIR, cv_curve, labelled, load, probe  # noqa: E402
from phase13.scripts.analyze import verdict as verdict13  # noqa: E402

OUT = ROOT / "outputs/paper"


def surface(text: str) -> list[float]:
    """Same features as paper/scripts/cpu_checks.py (C2)."""
    value = parse(text)
    try:
        num = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        num = 0.0
    finite = math.isfinite(num)
    return [len(text), len(text.split()), text.count("\n"), len(re.findall(r"\d+(?:\.\d+)?", text)),
            math.log10(abs(num) + 1) if finite else 0.0, float(finite and num == int(num)) if finite else 0.0,
            float(value is not None), float(num < 0) if finite else 0.0]


TAUS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5)


def surface_rows(keys, samples):
    x = []
    for pid, k, order in keys:
        a, b = (samples[(pid, 1)], samples[(pid, k)]) if order == "ab" else (samples[(pid, k)], samples[(pid, 1)])
        fa, fb = surface(a), surface(b)
        x.append(fa + fb + [vb - va for va, vb in zip(fa, fb)])
    return np.array(x, dtype=np.float32)


def p_first(dataset: str, judge: str, model, layer: int) -> dict:
    data = np.load(LAYER_DIR / f"{dataset}_{judge}.npz")
    pb = model.predict_proba(data["hidden"][:, layer].astype(np.float32))[:, 1]
    out = {}
    for t, p in zip(data["tasks"], pb):
        pid, k, order = t.split("|")
        out.setdefault((pid, int(k)), {})[order] = 1 - p if order == "ab" else p
    return {key: float(np.mean(list(v.values()))) for key, v in out.items() if len(v) == 2}


def strategy(samples, correct, rows, pf, judge_verdict, tau):
    acc = {"keep": [], "vote3": [], "vote5": [], "self_check": [], "abstain": [], "probe_only": []}
    for pid in rows:
        s = {k: samples[(pid, k)] for k in range(1, 6)}
        p = {k: parse(s[k]) for k in s}
        m3 = [1, 2, 3][vote([s[1], s[2], s[3]], [1, 2, 0])]
        m5 = 1 + vote([s[k] for k in range(1, 6)], [1, 2, 3, 4, 0])
        if same(p[1], p[2]):
            sc = ab = po = 1
        else:
            v = judge_verdict(pid, p[1], p[2])
            sc = 1 if v == "first" else 2 if v == "other" else m3
            q = pf.get((pid, 2))
            confident = q is not None and abs(q - 0.5) >= tau
            probe_pick = (1 if q > 0.5 else 2) if confident else m3
            ab = sc if v != "inconsistent" else probe_pick
            po = probe_pick
        for name, k in (("keep", 1), ("vote3", m3), ("vote5", m5), ("self_check", sc),
                        ("abstain", ab), ("probe_only", po)):
            acc[name].append(correct[(pid, k)])
    return {k: np.array(v, dtype=bool) for k, v in acc.items()}


def main() -> None:
    g12, s12, r12, c12 = load("phase12")
    g13, s13, r13, c13 = load("phase13")
    report = {"schema_version": "paper_c2b_f1_v1", "post_hoc": True, "C2b": {}}
    models = {}
    for judge in ("base", "p12"):
        xs, ys, gs, ks = labelled("phase12", judge, c12)
        xg, yg, gg, kg = labelled("phase13", judge, c13)
        layer = int(np.argmax(cv_curve(xs, ys, gs)))
        hid = probe().fit(xs[:, layer], ys)
        models[judge] = (hid, layer)
        fs, fg = surface_rows(ks, s12), surface_rows(kg, s13)
        sur = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(fs, ys)
        both = probe().fit(np.hstack([xs[:, layer], fs]), ys)
        ph, ps = hid.predict_proba(xg[:, layer])[:, 1], sur.predict_proba(fg)[:, 1]
        pc = both.predict_proba(np.hstack([xg[:, layer], fg]))[:, 1]
        sur_wrong = (ps > 0.5) != yg
        within = []
        for tr, te in GroupKFold(5).split(xg, yg, gg):
            m = probe().fit(np.hstack([xg[tr, layer], fg[tr]]), yg[tr])
            within.append(float((m.predict(np.hstack([xg[te, layer], fg[te]])) == yg[te]).mean()))
        report["C2b"][judge] = {
            "layer": layer, "rows": int(len(yg)),
            **{name: {"accuracy": float(((pr > 0.5) == yg).mean()), "auc": float(roc_auc_score(yg, pr))}
               for name, pr in (("hidden_only", ph), ("surface_only", ps), ("hidden_plus_surface", pc))},
            "hidden_plus_surface_within_gsm8k_cv": float(np.mean(within)),
            "hidden_accuracy_where_surface_wrong": {"n": int(sur_wrong.sum()),
                                                    "accuracy": float(((ph > 0.5) == yg)[sur_wrong].mean())},
            "hidden_accuracy_where_surface_right": float(((ph > 0.5) == yg)[~sur_wrong].mean())}
    # F1
    model, layer = models["p12"]
    pf13 = p_first("phase13", "p12", model, layer)
    pf12 = p_first("phase12", "p12", model, layer)  # in-sample on SVAMP (used only to pick tau)
    conf, both_wrong = [], []
    for (pid, k), q in pf13.items():
        if c13[(pid, 1)] and c13[(pid, k)]:
            continue
        conf.append(abs(q - 0.5)); both_wrong.append(not (c13[(pid, 1)] or c13[(pid, k)]))
    conf, both_wrong = np.array(conf), np.array(both_wrong)
    j12, j13 = g12.completed("judge_dpo"), g13.completed("judge_p12")
    v12 = lambda pid, a, b: verdict12(j12, pid, 2, a, b)
    v13 = lambda pid, a, b: verdict13(j13, pid, 2, a, b, False)
    svamp_rows = {r: None for r in r12}
    tau_scores = {tau: float(strategy(s12, c12, svamp_rows, pf12, v12, tau)["abstain"].mean()) for tau in TAUS}
    tau = max(TAUS, key=lambda t: (tau_scores[t], -t))
    tau_probe_only = max(TAUS, key=lambda t: (float(strategy(s12, c12, svamp_rows, pf12, v12, t)["probe_only"].mean()), -t))
    a = strategy(s13, c13, {r: None for r in r13}, pf13, v13, tau)
    a_po = strategy(s13, c13, {r: None for r in r13}, pf13, v13, tau_probe_only)["probe_only"]
    report["F1"] = {
        "layer": layer,
        "confidence_mean": {"one_right": float(conf[~both_wrong].mean()), "both_wrong": float(conf[both_wrong].mean())},
        "auc_low_confidence_flags_both_wrong": float(roc_auc_score(both_wrong, 0.5 - conf)),
        "pairs": {"one_right": int((~both_wrong).sum()), "both_wrong": int(both_wrong.sum())},
        "tau_grid_svamp_abstain_accuracy": tau_scores, "tau_abstain": tau, "tau_probe_only": tau_probe_only,
        "gsm8k_accuracy": {**{k: float(v.mean()) for k, v in a.items() if k != "probe_only"},
                           "probe_only": float(a_po.mean())},
        "abstain_vs_self_check": {"difference": float(a["abstain"].mean() - a["self_check"].mean()),
                                  "ci": ci(a["abstain"], a["self_check"], 20261050)},
        "abstain_vs_vote5": {"difference": float(a["abstain"].mean() - a["vote5"].mean()),
                             "ci": ci(a["abstain"], a["vote5"], 20261051)},
        "probe_only_vs_vote3": {"difference": float(a_po.mean() - a["vote3"].mean()),
                                "ci": ci(a_po, a["vote3"], 20261052)}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c2b_f1.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
