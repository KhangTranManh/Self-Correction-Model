"""Out-of-sample check of probe-weighted voting (F2), exploratory; rule in phase14/RUN_STATUS.md (2026-10-15 (iv)).

For each model and train→test pair (ASDiv→MATH, MATH→ASDiv, SVAMP→GSM8K): pointwise probe on the answer site
(last token of s1, s2, s3; label = verifier-correct, project parser); StandardScaler + L2 logistic, C = 0.01;
layer chosen by 5-fold problem-grouped CV AUC on the training set only; on the test set each problem's answer
clusters among s1–s3 are weighted by the summed P(correct); the heaviest cluster is chosen (ties → s1 first).
Compared with vote@3 (same priority) on the same problems; paired bootstrap over problems (2000).
"Confirmed out of sample for a model" if the CI of the difference is above 0 in ≥ 2 of the 3 pairs.
Output: paper/results/phase14_f2_oos.{json,md}.
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
sys.path.insert(0, str(ROOT))
from paper.scripts import phase14_data as P  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.answers import vote  # noqa: E402

MODELS = ("llama31_8b", "mistral_7b", "original_solver", "qwen25_instruct")
PAIRS = (("asdiv", "math13"), ("math13", "asdiv"), ("svamp12", "gsm8k13"))
OUT = ROOT / "paper/results/phase14_f2_oos"


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, penalty="l2", solver="liblinear",
                                                              max_iter=2000, random_state=20261015))


def load(model, ds):
    h = P.hidden(model, ds, "answer")
    tasks = [t.split("|") for t in h["tasks"]]
    keys = [(p, int(k)) for p, k in tasks]
    ok = P.correct(model, ds)
    y = np.array([int(ok[k]) for k in keys])
    return h["hidden"], keys, y, np.array([k[0] for k in keys])


def main():
    report = {"schema_version": "paper_phase14_f2_oos_v1", "exploratory": True, "models": {}}
    for m in MODELS:
        report["models"][m] = {}
        cache = {}
        for tr, te in PAIRS:
            for d in (tr, te):
                if d not in cache:
                    cache[d] = load(m, d)
            X, keys, y, g = cache[tr]
            layers = X.shape[1]
            cv = []
            for layer in range(layers):
                xs = X[:, layer].astype(np.float32)
                aucs = []
                for a, b in GroupKFold(5).split(xs, y, g):
                    if len(set(y[b])) < 2:
                        continue
                    aucs.append(roc_auc_score(y[b], probe().fit(xs[a], y[a]).predict_proba(xs[b])[:, 1]))
                cv.append(float(np.mean(aucs)))
            layer = int(np.argmax(cv))
            model = probe().fit(X[:, layer].astype(np.float32), y)
            Xt, kt, yt, _ = cache[te]
            prob = dict(zip(kt, model.predict_proba(Xt[:, layer].astype(np.float32))[:, 1]))
            s, ok = P.samples(m, te), P.correct(m, te)
            wv, v3 = [], []
            for r in P.problems(te):
                pid = r["id"]
                texts = [s[(pid, k)]["output"] for k in (1, 2, 3)]
                vals = [parse(t) for t in texts]
                pr = [prob.get((pid, k), 0.0) for k in (1, 2, 3)]
                weight = [sum(pr[j] for j in range(3) if vals[i] is not None and same(vals[i], vals[j])) if vals[i]
                          is not None else pr[i] for i in range(3)]
                best = max(range(3), key=lambda i: (weight[i], -i))
                wv.append(float(ok[(pid, best + 1)]))
                v3.append(float(ok[(pid, 1 + vote(texts, [0, 1, 2]))]))
            d = np.array(wv) - np.array(v3)
            rng = np.random.default_rng(20261015)
            ms = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
            ci = [float(x) for x in np.quantile(ms, [0.025, 0.975])]
            test_auc = float(roc_auc_score(yt, model.predict_proba(Xt[:, layer].astype(np.float32))[:, 1]))
            report["models"][m][f"{tr}->{te}"] = {"layer": layer, "train_cv_auc": cv[layer], "test_auc": test_auc,
                                                  "weighted_vote3": float(np.mean(wv)), "vote3": float(np.mean(v3)),
                                                  "difference": float(d.mean()), "ci": ci, "ci_above_0": ci[0] > 0}
            print(m, tr, te, report["models"][m][f"{tr}->{te}"], flush=True)
        hits = sum(v["ci_above_0"] for v in report["models"][m].values())
        report["models"][m]["confirmed_out_of_sample"] = hits >= 2
        OUT.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    names = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "original_solver": "Self_Correction",
             "qwen25_instruct": "Qwen2.5-7B-Instruct"}
    lines = ["# F2 ngoài mẫu: vote có trọng số theo probe (s1–s3) so với vote@3 (khám phá)", "",
             "Quy tắc ghi trước: `phase14/RUN_STATUS.md` (2026-10-15, iv). Số đứng trước là ước lượng điểm, [a; b] là "
             "KTC 95% bootstrap ghép cặp theo bài.", "",
             "| Mô hình | Train → test | Layer | AUC CV (train) | AUC test | Vote có trọng số | vote@3 | Hiệu [KTC] |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for m, e in report["models"].items():
        for k, v in e.items():
            if k == "confirmed_out_of_sample":
                continue
            lines.append(f"| {names[m]} | {k} | {v['layer']} | {v['train_cv_auc']:.3f} | {v['test_auc']:.3f} | "
                         f"{100 * v['weighted_vote3']:.1f}% | {100 * v['vote3']:.1f}% | {100 * v['difference']:+.1f} "
                         f"[{100 * v['ci'][0]:+.1f}; {100 * v['ci'][1]:+.1f}] |")
    lines += ["", "Xác nhận ngoài mẫu (KTC > 0 ở ≥ 2/3 cặp): " + ", ".join(
        f"{names[m]} {'có' if e['confirmed_out_of_sample'] else 'không'}" for m, e in report["models"].items())]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
