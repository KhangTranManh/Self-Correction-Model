"""Exploratory layer study: which solution is right, read from judge-prompt hidden states.

Rows: judged tasks (source, k, order) on one-right pairs. Label y = 1 when the
correct solution is the one shown as Solution B, else 0 (each pair contributes
both orders, so labels are balanced). Probe: StandardScaler + L2 logistic
regression (C = 0.01, liblinear), as in Phase 5. Fixed before running.

1. Cross-dataset (judges base, p12): train on Phase 12 SVAMP, test on Phase 13
   GSM8K, at every layer. The reported layer is chosen ONLY by 5-fold
   source-grouped CV accuracy on SVAMP.
2. Within GSM8K (judges base_c, p13b, which have no SVAMP data): 5-fold
   source-grouped CV per layer (descriptive).
3. Pair level (chosen layer, base and p12): p_first = mean(P(first right | ab),
   P(first right | ba)); accuracy on all one-right GSM8K pairs, and separately
   on pairs where the judge's both-orders verdict was consistent vs
   inconsistent (can the probe resolve what the judge leaves open?).
4. Exploratory strategy: self-check with p12, using the probe instead of
   vote@3 when p12 is inconsistent; compared with keep, vote@3, vote@5.
Holdout labels were already opened by the Phase 13 analysis; this study is
exploratory and changes no Phase 13 endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
from phase9.scripts.analyze import ci  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase13.scripts.analyze import verdict  # noqa: E402

LAYER_DIR = ROOT / "outputs/phase13_v1/layer"
C = 0.01
LAYERS = range(29)


def probe() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", solver="liblinear",
                                                              max_iter=5000, random_state=20261009))


def load(dataset: str):
    if dataset == "phase12":
        from phase12.scripts import generate as g
    else:
        from phase13.scripts import generate as g
    samples = g.completed("samples")
    rows = {row["id"]: row for row in g.holdout()}
    verifier = MathVerifier()
    correct = {}
    for (pid, k), text in samples.items():
        gold = Problem(id=pid, domain="math", question=rows[pid]["question"],
                       reference_answer=rows[pid]["reference_answer"])
        correct[(pid, k)] = bool(verifier.verify(gold, text).passed)
    return g, samples, rows, correct


def labelled(dataset: str, judge: str, correct: dict):
    data = np.load(LAYER_DIR / f"{dataset}_{judge}.npz")
    tasks = [t.split("|") for t in data["tasks"]]
    keep, y, groups, keys = [], [], [], []
    for i, (pid, k, order) in enumerate(tasks):
        k = int(k)
        first_ok, other_ok = correct[(pid, 1)], correct[(pid, k)]
        if first_ok == other_ok:
            continue
        right_shown_b = (order == "ab" and other_ok) or (order == "ba" and first_ok)
        keep.append(i); y.append(int(right_shown_b)); groups.append(pid); keys.append((pid, k, order))
    return data["hidden"][keep].astype(np.float32), np.array(y), np.array(groups), keys


def cv_curve(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[float]:
    folds = GroupKFold(n_splits=5)
    curve = []
    for layer in LAYERS:
        hits = []
        for tr, te in folds.split(x, y, groups):
            model = probe().fit(x[tr, layer], y[tr])
            hits.append(float((model.predict(x[te, layer]) == y[te]).mean()))
        curve.append(float(np.mean(hits)))
    return curve


def main() -> None:
    report = {"schema_version": "phase13_layer_probe_v1", "probe": "StandardScaler + L2 logistic, C=0.01",
              "feature": "last prompt token hidden state, layers 0-28", "exploratory": True}
    _, _, _, correct12 = load("phase12")
    g13, samples13, rows13, correct13 = load("phase13")
    models = {}
    for judge in ("base", "p12"):
        xs, ys, gs, _ = labelled("phase12", judge, correct12)
        xg, yg, gg, kg = labelled("phase13", judge, correct13)
        svamp_cv = cv_curve(xs, ys, gs)
        layer = int(np.argmax(svamp_cv))
        transfer, aucs = [], []
        for l in LAYERS:
            model = probe().fit(xs[:, l], ys)
            transfer.append(float((model.predict(xg[:, l]) == yg).mean()))
            aucs.append(float(roc_auc_score(yg, model.predict_proba(xg[:, l])[:, 1])))
        models[judge] = (probe().fit(xs[:, layer], ys), layer, xg, yg, kg)
        report[f"cross_dataset_{judge}"] = {
            "svamp_rows": int(len(ys)), "gsm8k_rows": int(len(yg)),
            "svamp_cv_accuracy_by_layer": svamp_cv, "chosen_layer_by_svamp_cv": layer,
            "gsm8k_accuracy_by_layer": transfer, "gsm8k_auc_by_layer": aucs,
            "gsm8k_accuracy_at_chosen_layer": transfer[layer], "gsm8k_auc_at_chosen_layer": aucs[layer],
            "best_gsm8k_layer_descriptive": int(np.argmax(transfer))}
    for judge in ("base_c", "p13b"):
        xg, yg, gg, _ = labelled("phase13", judge, correct13)
        curve = cv_curve(xg, yg, gg)
        report[f"within_gsm8k_{judge}"] = {"rows": int(len(yg)), "cv_accuracy_by_layer": curve,
                                           "best_layer": int(np.argmax(curve)), "best_accuracy": max(curve)}

    judges13 = {name: g13.completed(f"judge_{name}") for name in ("base", "p12")}
    for judge in ("base", "p12"):
        model, layer, xg, yg, kg = models[judge]
        prob_b = model.predict_proba(xg[:, layer])[:, 1]
        p_first = {}
        for (pid, k, order), pb in zip(kg, prob_b):
            p_first.setdefault((pid, k), []).append(1 - pb if order == "ab" else pb)
        pairs = {"all": [], "consistent": [], "inconsistent": []}
        for (pid, k), values in p_first.items():
            if len(values) != 2:
                continue
            hit = (np.mean(values) > 0.5) == correct13[(pid, 1)]
            pf, pk = parse(samples13[(pid, 1)]), parse(samples13[(pid, k)])
            v = verdict(judges13[judge], pid, k, pf, pk, False)
            pairs["all"].append(hit)
            pairs["inconsistent" if v == "inconsistent" else "consistent"].append(hit)
        report[f"pair_level_{judge}"] = {name: {"n": len(v), "accuracy": float(np.mean(v)) if v else None}
                                         for name, v in pairs.items()}

    # Exploratory strategy: p12 self-check with the p12 probe on inconsistent (s1, s2) pairs.
    model, layer, _, _, _ = models["p12"]
    data = np.load(LAYER_DIR / "phase13_p12.npz")
    index = {t: i for i, t in enumerate(data["tasks"])}
    verifier = MathVerifier()
    acc = {"keep": [], "vote3": [], "vote5": [], "self_check_p12": [], "self_check_p12_probe": []}
    for pid, row in rows13.items():
        gold = Problem(id=pid, domain="math", question=row["question"], reference_answer=row["reference_answer"])
        ok = lambda text: bool(verifier.verify(gold, text).passed)
        s = {k: samples13[(pid, k)] for k in range(1, 6)}
        p = {k: parse(s[k]) for k in s}
        maj3 = [s[1], s[2], s[3]][vote([s[1], s[2], s[3]], [1, 2, 0])]
        maj5 = [s[1], s[2], s[3], s[4], s[5]][vote([s[1], s[2], s[3], s[4], s[5]], [1, 2, 3, 4, 0])]
        if same(p[1], p[2]):
            sc = scp = s[1]
        else:
            v = verdict(judges13["p12"], pid, 2, p[1], p[2], False)
            sc = s[1] if v == "first" else s[2] if v == "other" else maj3
            if v == "inconsistent":
                rows_ = [index[f"{pid}|2|ab"], index[f"{pid}|2|ba"]]
                pb = model.predict_proba(data["hidden"][rows_][:, layer].astype(np.float32))[:, 1]
                scp = s[1] if np.mean([1 - pb[0], pb[1]]) > 0.5 else s[2]
            else:
                scp = sc
        for name, text in (("keep", s[1]), ("vote3", maj3), ("vote5", maj5),
                           ("self_check_p12", sc), ("self_check_p12_probe", scp)):
            acc[name].append(ok(text))
    a = {k: np.array(v) for k, v in acc.items()}
    report["strategy_exploratory"] = {
        "accuracy": {k: float(v.mean()) for k, v in a.items()},
        "probe_vs_vote3": {"difference": float(a["self_check_p12_probe"].mean() - a["vote3"].mean()),
                           "ci": ci(a["self_check_p12_probe"], a["vote3"], 20261030)},
        "probe_vs_vote5": {"difference": float(a["self_check_p12_probe"].mean() - a["vote5"].mean()),
                           "ci": ci(a["self_check_p12_probe"], a["vote5"], 20261031)}}
    (LAYER_DIR / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if not isinstance(v, dict) or "by_layer" not in json.dumps(v)}
    print(json.dumps({**summary, **{k: {kk: vv for kk, vv in v.items() if "by_layer" not in kk}
                                    for k, v in report.items() if isinstance(v, dict) and "by_layer" in json.dumps(v)}},
                     indent=1))


if __name__ == "__main__":
    main()
