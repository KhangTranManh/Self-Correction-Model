"""Checklist C1 and C4 on the GPU host's saved hidden states (CPU only).

C4: one protocol for all four judges — train on Phase 12 SVAMP, test on Phase 13
GSM8K, layer chosen by 5-fold source-grouped CV on SVAMP only (as in
phase13/layer/probe_analysis.py), plus within-GSM8K grouped CV per layer.
C1: at the chosen layer, the share of judged GSM8K pairs where the probe picks
the same solution in both orders, next to the judge's own consistency, and the
probe's accuracy on its consistent one-right pairs.
Output: outputs/phase13_v1/layer/c1_c4_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from phase9.scripts.answers import parse  # noqa: E402
from phase13.layer.probe_analysis import LAYER_DIR, LAYERS, cv_curve, labelled, load, probe  # noqa: E402
from phase13.scripts.analyze import verdict  # noqa: E402

JUDGES = ("base", "p12", "base_c", "p13b")
CONSTRAINED = {"base_c", "p13b"}


def main() -> None:
    _, _, _, correct12 = load("phase12")
    g13, samples13, _, correct13 = load("phase13")
    report = {"schema_version": "paper_c1_c4_v1", "post_hoc": True, "judges": {}}
    for judge in JUDGES:
        xs, ys, gs, _ = labelled("phase12", judge, correct12)
        xg, yg, gg, _ = labelled("phase13", judge, correct13)
        svamp_cv = cv_curve(xs, ys, gs)
        layer = int(np.argmax(svamp_cv))
        gsm_cv = cv_curve(xg, yg, gg)
        transfer, aucs = [], []
        for l in LAYERS:
            m = probe().fit(xs[:, l], ys)
            transfer.append(float((m.predict(xg[:, l]) == yg).mean()))
            aucs.append(float(roc_auc_score(yg, m.predict_proba(xg[:, l])[:, 1])))
        model = probe().fit(xs[:, layer], ys)
        # C1: probe consistency over all judged GSM8K pairs.
        data = np.load(LAYER_DIR / f"phase13_{judge}.npz")
        tasks = [t.split("|") for t in data["tasks"]]
        prob_b = model.predict_proba(data["hidden"][:, layer].astype(np.float32))[:, 1]
        picks = {}
        for (pid, k, order), pb in zip(tasks, prob_b):
            shown_b = pb >= 0.5
            first_pick = (order == "ab") != shown_b   # ab: s1 is A; ba: s1 is B
            picks.setdefault((pid, int(k)), {})[order] = first_pick
        judge_texts = g13.completed(f"judge_{judge}")
        stats = {"all": [0, 0], "one_right": [0, 0]}
        cons_hits = []
        judge_cons = {"all": [0, 0], "one_right": [0, 0]}
        for (pid, k), pk in picks.items():
            if len(pk) != 2:
                continue
            one_right = correct13[(pid, 1)] != correct13[(pid, k)]
            consistent = pk["ab"] == pk["ba"]
            v = verdict(judge_texts, pid, k, parse(samples13[(pid, 1)]), parse(samples13[(pid, k)]),
                        judge in CONSTRAINED)
            for key in ("all", "one_right") if one_right else ("all",):
                stats[key][0] += consistent; stats[key][1] += 1
                judge_cons[key][0] += v != "inconsistent"; judge_cons[key][1] += 1
            if one_right and consistent:
                cons_hits.append(pk["ab"] == correct13[(pid, 1)])
        report["judges"][judge] = {
            "chosen_layer_by_svamp_cv": layer,
            "svamp_cv_accuracy_by_layer": svamp_cv, "gsm8k_cross_accuracy_by_layer": transfer,
            "gsm8k_cross_auc_by_layer": aucs, "gsm8k_within_cv_accuracy_by_layer": gsm_cv,
            "gsm8k_cross_accuracy_at_chosen_layer": transfer[layer], "gsm8k_cross_auc_at_chosen_layer": aucs[layer],
            "gsm8k_within_cv_best": [int(np.argmax(gsm_cv)), max(gsm_cv)],
            "C1_probe_consistency": {k: v[0] / v[1] for k, v in stats.items()},
            "C1_judge_consistency": {k: v[0] / v[1] for k, v in judge_cons.items()},
            "C1_probe_accuracy_on_its_consistent_one_right_pairs": float(np.mean(cons_hits)),
            "C1_probe_consistent_one_right_pairs": len(cons_hits)}
    out = LAYER_DIR / "c1_c4_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for j, e in report["judges"].items():
        print(j, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if "by_layer" not in k})


if __name__ == "__main__":
    main()
