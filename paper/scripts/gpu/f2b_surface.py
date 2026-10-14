"""F2b: pointwise surface-feature baseline for the F2 probe (CPU; run on the host
that holds outputs/paper/pointwise_<ds>.npz).

Per solution: the C2 surface features of that one solution (chars, words,
newlines, number count, log10|answer|, integer answer, answer parsed, negative
answer). Same rows, labels (verifier-correct) and split as F2: train on all
4,800 SVAMP samples, test on all 2,805 GSM8K samples. Models:
  surface only (C = 1.0 and C = 0.01), hidden only at the F2 layer (23),
  hidden + surface; for each: accuracy, AUC, best-of-5, weighted vote@5, and the
  one-right pair pick, with paired CIs vs vote@5.
Output: outputs/paper/f2b_surface.json.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from phase9.scripts.analyze import ci  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase13.layer.probe_analysis import load, probe  # noqa: E402

OUT = ROOT / "outputs/paper"
LAYER = 23  # chosen by SVAMP CV in F2


def surface(text: str) -> list[float]:
    """Same features as paper/scripts/cpu_checks.py (C2), for one solution."""
    value = parse(text)
    try:
        num = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        num = 0.0
    finite = math.isfinite(num)
    return [len(text), len(text.split()), text.count("\n"), len(re.findall(r"\d+(?:\.\d+)?", text)),
            math.log10(abs(num) + 1) if finite else 0.0, float(finite and num == int(num)) if finite else 0.0,
            float(value is not None), float(num < 0) if finite else 0.0]


def rows(name: str):
    _, samples, _, correct = load(name)
    data = np.load(OUT / f"pointwise_{name}.npz")
    keys = [(p, int(k)) for p, k in (s.split("|") for s in data["keys"])]
    x_s = np.array([surface(samples[key]) for key in keys], dtype=np.float32)
    x_h = data["hidden"][:, LAYER].astype(np.float32)
    y = np.array([int(correct[key]) for key in keys])
    return keys, x_s, x_h, y, samples, correct


def main() -> None:
    _, xs_s, xs_h, ys, _, _ = rows("phase12")
    kg, xg_s, xg_h, yg, s13, c13 = rows("phase13")
    g13 = load("phase13")[0]
    models = {
        "surface_C1": (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)), xs_s, xg_s),
        "surface_C0.01": (make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)), xs_s, xg_s),
        "hidden": (probe(), xs_h, xg_h),
        "hidden_plus_surface": (probe(), np.hstack([xs_h, xs_s]), np.hstack([xg_h, xg_s])),
    }
    report = {"schema_version": "paper_f2b_v1", "post_hoc": True, "layer": LAYER,
              "svamp_rows": int(len(ys)), "gsm8k_rows": int(len(yg)), "models": {}}
    pids = sorted({p for p, _ in kg})
    for i, (name, (model, xtr, xte)) in enumerate(models.items()):
        pr = model.fit(xtr, ys).predict_proba(xte)[:, 1]
        prob = dict(zip(kg, pr))
        acc = {"vote5": [], "best_of_5": [], "weighted_vote5": []}
        for pid in pids:
            texts = [s13[(pid, k)] for k in range(1, 6)]
            vals = [parse(t) for t in texts]
            p = [prob[(pid, k)] for k in range(1, 6)]
            acc["vote5"].append(c13[(pid, 1 + vote(texts, [1, 2, 3, 4, 0]))])
            acc["best_of_5"].append(c13[(pid, 1 + int(np.argmax(p)))])
            w = [sum(p[j] for j in range(5) if vals[a] is not None and same(vals[a], vals[j])) for a in range(5)]
            acc["weighted_vote5"].append(c13[(pid, 1 + int(np.argmax(w)))])
        a = {k: np.array(v, dtype=float) for k, v in acc.items()}
        pair = [(prob[(pid, 1)] > prob[(pid, k)]) == c13[(pid, 1)]
                for pid, k, order in g13.completed("judge_base") if order == "ab" and c13[(pid, 1)] != c13[(pid, k)]]
        report["models"][name] = {
            "accuracy": float(((pr > 0.5) == yg).mean()), "auc": float(roc_auc_score(yg, pr)),
            "best_of_5": float(a["best_of_5"].mean()), "weighted_vote5": float(a["weighted_vote5"].mean()),
            "weighted_vote5_minus_vote5": {"difference": float((a["weighted_vote5"] - a["vote5"]).mean()),
                                           "ci": ci(a["weighted_vote5"], a["vote5"], 20261100 + i)},
            "one_right_pair_pick": float(np.mean(pair))}
    report["vote5"] = float(a["vote5"].mean())
    (OUT / "f2b_surface.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
