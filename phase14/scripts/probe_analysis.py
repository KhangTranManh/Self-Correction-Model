"""Exploratory hidden-state and forced-verdict analysis (run only after the main opening).

Not an endpoint. Mirrors the report's section 4.3 and Table 3 for every model.
Probe = StandardScaler + L2 logistic regression (C = 0.01, liblinear), fixed
before running, as in Phases 5 and 13. Training datasets: asdiv, math13,
svamp12 (the report trains on SVAMP and tests on GSM8K); each probe is tested
on every other dataset, including gsm8k13 and the bridge set. The layer is
chosen only by 5-fold source-grouped CV on the training dataset.

Selectors on one-right pairs (judge site, base prompt; label = right shown as B):
  surface            length, step count, log answer magnitude (B - A),
                     logistic regression trained on the training dataset
  likelihood         higher mean token log-probability (answer site)
  probe              judge-site hidden state at the chosen layer
  probe + surface    both feature sets together
  pointwise probe    answer-site probe trained to predict each solution's
                     correctness; the pair choice is the solution with the
                     higher predicted probability (report: "theo tung loi giai")
Pair level: the probe averaged over both orders, split by the judge's own
verdict (consistent-right, consistent-wrong, inconsistent).

Forced verdict (direct site, no written check), per dataset:
  single-order accuracy and order-averaged (Ren et al.) accuracy from the final
  layer, their difference, the log-odds position bias, and the same two
  accuracies at every layer through the logit lens.
First-answer probe (answer site, s1; Phase 5 analogue): balanced accuracy and
ROC-AUC with a mean-log-probability baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _to_sympy  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase14.scripts.config import BRIDGE, COMPARISON, DATASETS, JUDGED, OUT, RUN_MODELS  # noqa: E402
from phase14.scripts.extract_hidden import destination  # noqa: E402
from phase14.scripts.generate import completed, folder, load_samples, problems  # noqa: E402
from phase14.scripts.metrics import verdict  # noqa: E402

C = 0.01
SEED = 20261013
TRAIN = ("asdiv", "math13", "svamp12")
ALL = (*DATASETS, *COMPARISON, BRIDGE)
REPORT = OUT / "main" / "analysis" / "probe_report.json"


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", solver="liblinear",
                                                              max_iter=5000, random_state=SEED))


def load(model: str, dataset: str) -> dict | None:
    paths = {site: destination("main", model, dataset, site) for site in ("answer", "judge", "direct")}
    if not paths["answer"].exists():
        return None
    rows = problems(dataset, "main")
    samples = load_samples(model, dataset, "main")
    ref = {r["id"]: _to_sympy(r["reference_answer"]) for r in rows}
    correct = {(r["id"], k): bool(same(parse(samples[(r["id"], k)]["output"]), ref[r["id"]]))
               for r in rows for k in (1, *JUDGED)}
    answer = np.load(paths["answer"])
    out = {"correct": correct, "samples": samples, "answer": answer,
           "answer_index": {t: i for i, t in enumerate(answer["tasks"])}}
    out["judge"] = np.load(paths["judge"]) if paths["judge"].exists() else None
    out["direct"] = np.load(paths["direct"]) if paths["direct"].exists() else None
    base = folder(model, dataset, "main")
    out["judged"] = completed(base, "judge_base") if (base / "judge_base" / "summary.json").exists() else None
    return out


def surface(text: str, tokens: int) -> list[float]:
    """Report surface features: length, step count, answer magnitude."""
    steps = sum(1 for line in text.split("\n") if re.search(r"\d", line))
    value = parse(text)
    try:
        magnitude = math.log10(abs(float(value)) + 1) if value is not None else 0.0
    except (TypeError, ValueError):
        magnitude = 0.0
    return [float(tokens), float(steps), magnitude]


def pairs(data: dict) -> dict | None:
    """One-right judged tasks (both orders) with labels, features, and probe inputs."""
    if data["judge"] is None or data["judged"] is None:
        return None
    a_idx, answer, correct, samples = data["answer_index"], data["answer"], data["correct"], data["samples"]
    keep, y, groups, feats, like, tasks, pointwise = [], [], [], [], [], [], []
    for i, task in enumerate(data["judge"]["base_tasks"]):
        pid, k, order = task.split("|")
        k = int(k)
        if correct[(pid, 1)] == correct[(pid, k)]:
            continue
        a, b = (1, k) if order == "ab" else (k, 1)
        ia, ib = a_idx[f"{pid}|{a}"], a_idx[f"{pid}|{b}"]
        keep.append(i)
        y.append(int(correct[(pid, b)]))
        groups.append(pid)
        fa = surface(samples[(pid, a)]["output"], int(answer["answer_tokens"][ia]))
        fb = surface(samples[(pid, b)]["output"], int(answer["answer_tokens"][ib]))
        feats.append([fb[j] - fa[j] for j in range(3)])
        like.append(int(answer["mean_logprob"][ib] > answer["mean_logprob"][ia]))
        tasks.append((pid, k, order))
        pointwise.append((ia, ib))
    p1 = {pid: parse(samples[(pid, 1)]["output"]) for pid in {t[0] for t in tasks}}
    verdicts = {}
    for pid, k, _ in tasks:
        right = "first" if correct[(pid, 1)] else "other"
        v = verdict(data["judged"], pid, k, p1[pid], parse(samples[(pid, k)]["output"]), False)
        verdicts[(pid, k)] = "inconsistent" if v == "inconsistent" else ("right" if v == right else "wrong")
    return {"hidden": data["judge"]["base_hidden"][keep], "y": np.array(y), "groups": np.array(groups),
            "surface": np.array(feats), "likelihood": np.array(like), "tasks": tasks, "verdicts": verdicts,
            "pointwise": np.array(pointwise), "first_right": {(p, k): correct[(p, 1)] for p, k, _ in tasks}}


def pointwise_set(data: dict) -> dict:
    answer, correct = data["answer"], data["correct"]
    keys = [t.split("|") for t in answer["tasks"]]
    return {"hidden": answer["hidden"], "y": np.array([int(correct[(p, int(k))]) for p, k in keys]),
            "groups": np.array([p for p, _ in keys]), "mean_logprob": answer["mean_logprob"][:, None],
            "is_first": np.array([k == "1" for _, k in keys])}


def choose_layer(x3d: np.ndarray, y: np.ndarray, groups: np.ndarray, balanced: bool) -> tuple[int, list[float]]:
    folds = GroupKFold(n_splits=5)
    scores = []
    for layer in range(x3d.shape[1]):
        x = x3d[:, layer].astype(np.float32)
        fold = []
        for tr, te in folds.split(x, y, groups):
            pred = probe().fit(x[tr], y[tr]).predict(x[te])
            fold.append(balanced_accuracy_score(y[te], pred) if balanced else float((pred == y[te]).mean()))
        scores.append(float(np.mean(fold)))
    return int(np.argmax(scores)), scores


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else None


def selectors(train: dict, train_point: dict, tests: dict[str, tuple[dict, dict]]) -> dict:
    layer, cv = choose_layer(train["hidden"], train["y"], train["groups"], balanced=False)
    p_layer, p_cv = choose_layer(train_point["hidden"], train_point["y"], train_point["groups"], balanced=True)
    hid = probe().fit(train["hidden"][:, layer].astype(np.float32), train["y"])
    surf = probe().fit(train["surface"], train["y"])
    both = probe().fit(np.hstack([train["hidden"][:, layer].astype(np.float32), train["surface"]]), train["y"])
    point = probe().fit(train_point["hidden"][:, p_layer].astype(np.float32), train_point["y"])
    out = {"judge_site_layer": layer, "judge_site_cv_by_layer": cv,
           "pointwise_layer": p_layer, "pointwise_cv_by_layer": p_cv, "tests": {}}
    for name, (test, test_point) in tests.items():
        x = test["hidden"][:, layer].astype(np.float32)
        p_hid = hid.predict_proba(x)[:, 1]
        p_surf = surf.predict_proba(test["surface"])[:, 1]
        p_both = both.predict_proba(np.hstack([x, test["surface"]]))[:, 1]
        sol = point.predict_proba(test_point["hidden"][:, p_layer].astype(np.float32))[:, 1]
        p_point = sol[test["pointwise"][:, 1]] - sol[test["pointwise"][:, 0]]   # B minus A
        y = test["y"]
        result = {"rows": int(len(y)), "pairs": int(len(y) // 2),
                  "surface": {"accuracy": float(((p_surf > 0.5) == y).mean()), "auc": auc(y, p_surf)},
                  "likelihood": {"accuracy": float((test["likelihood"] == y).mean())},
                  "probe": {"accuracy": float(((p_hid > 0.5) == y).mean()), "auc": auc(y, p_hid)},
                  "probe_plus_surface": {"accuracy": float(((p_both > 0.5) == y).mean()), "auc": auc(y, p_both)},
                  "pointwise_probe": {"accuracy": float(((p_point > 0) == y).mean()), "auc": auc(y, p_point),
                                      "per_solution_auc": auc(test_point["y"], sol)}}
        pair, flips = {}, 0
        for (pid, k, order), prob in zip(test["tasks"], p_hid):
            pair.setdefault((pid, k), {})[order] = 1 - prob if order == "ab" else prob   # P(first right)
        by = {"right": [], "wrong": [], "inconsistent": []}
        for key, orders in pair.items():
            if len(orders) == 2:
                flips += (orders["ab"] > 0.5) != (orders["ba"] > 0.5)
            by[test["verdicts"][key]].append(float((np.mean(list(orders.values())) > 0.5) == test["first_right"][key]))
        result["pair_probe_by_judge_verdict"] = {v: {"n": len(x_), "accuracy": float(np.mean(x_)) if x_ else None}
                                                 for v, x_ in by.items()}
        result["pair_probe_order_flip_rate"] = flips / max(len(pair), 1)
        out["tests"][name] = result
    return out


def first_answer(train_point: dict, tests: dict[str, dict]) -> dict:
    tr = {k: v[train_point["is_first"]] if isinstance(v, np.ndarray) else v for k, v in train_point.items()}
    layer, cv = choose_layer(tr["hidden"], tr["y"], tr["groups"], balanced=True)
    model = probe().fit(tr["hidden"][:, layer].astype(np.float32), tr["y"])
    base = probe().fit(tr["mean_logprob"], tr["y"])
    out = {"layer": layer, "cv_by_layer": cv, "tests": {}}
    for name, test in tests.items():
        te = {k: v[test["is_first"]] if isinstance(v, np.ndarray) else v for k, v in test.items()}
        p = model.predict_proba(te["hidden"][:, layer].astype(np.float32))[:, 1]
        out["tests"][name] = {"n": int(len(te["y"])),
                              "balanced_accuracy": float(balanced_accuracy_score(te["y"], p > 0.5)),
                              "auc": auc(te["y"], p),
                              "logprob_baseline_balanced_accuracy": float(
                                  balanced_accuracy_score(te["y"], base.predict(te["mean_logprob"])))}
    return out


def forced_verdict(data: dict) -> dict | None:
    """Ren et al.-style order averaging on the forced verdict, final layer and logit lens."""
    direct, correct = data["direct"], data["correct"]
    if direct is None:
        return None
    index = {t: i for i, t in enumerate(direct["tasks"])}
    margins_final = direct["final_logprobs"][:, 0] - direct["final_logprobs"][:, 1]
    rows = []
    for task in direct["tasks"]:
        pid, k, order = task.split("|")
        if order != "ab" or correct[(pid, 1)] == correct[(pid, int(k))]:
            continue
        other = f"{pid}|{k}|ba"
        if other in index:
            rows.append((pid, index[task], index[other], correct[(pid, 1)]))
    if not rows:
        return None

    def scores(m_ab: np.ndarray, m_ba: np.ndarray, first_right: np.ndarray) -> tuple[float, float, float]:
        single = ((np.where(m_ab == 0, 0.5, (m_ab > 0) == first_right)) +
                  np.where(m_ba == 0, 0.5, (m_ba < 0) == first_right)) / 2
        s = (m_ab - m_ba) / 2
        debiased = np.where(s == 0, 0.5, (s > 0) == first_right)
        return float(single.mean()), float(debiased.mean()), float(((m_ab + m_ba) / 2).mean())

    ab, ba = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    first_right = np.array([r[3] for r in rows])
    single, debiased, bias = scores(margins_final[ab], margins_final[ba], first_right)
    lens = direct["lens_margin"]
    curve = [dict(zip(("single", "debiased", "bias"), scores(lens[ab, layer], lens[ba, layer], first_right)))
             for layer in range(lens.shape[1])]
    return {"pairs": len(rows), "single_order_accuracy": single, "order_averaged_accuracy": debiased,
            "gain": debiased - single, "bias_logodds": bias, "logit_lens_by_layer": curve}


def main() -> None:
    if not (OUT / "main" / "analysis" / "report.json").exists():
        raise RuntimeError("Probe analysis runs only after the single main opening")
    report = {"schema_version": "phase14_probe_report_v1", "exploratory": True, "C": C, "models": {}}
    for model in RUN_MODELS:
        data = {d: load(model, d) for d in ALL}
        judge = {d: pairs(v) if v else None for d, v in data.items()}
        point = {d: pointwise_set(v) if v else None for d, v in data.items()}
        entry = {"forced_verdict": {d: forced_verdict(v) for d, v in data.items() if v}}
        for train in TRAIN:
            if judge.get(train) is None:
                continue
            tests = {d: (judge[d], point[d]) for d in ALL if d != train and judge.get(d) is not None}
            entry[f"selectors_train_{train}"] = selectors(judge[train], point[train], tests)
            entry[f"first_answer_train_{train}"] = first_answer(point[train], {d: point[d] for d in ALL
                                                                               if d != train and point.get(d)})
        report["models"][model] = entry
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({m: {k: ({d: {s: v[s] for s in ("surface", "likelihood", "probe", "probe_plus_surface",
                                                     "pointwise_probe")} for d, v in e["tests"].items()}
                              if k.startswith("selectors") else None) for k, e in entry.items()
                          if k.startswith("selectors")} for m, entry in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
