"""Fit weak linear probes and controls on frozen Phase 3 activations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 161803
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def metrics(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "keep_recall": float(recall_score(y, pred, labels=[0], average=None, zero_division=0)[0]),
        "revise_recall": float(recall_score(y, pred, labels=[1], average=None, zero_division=0)[0]),
        "revise_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": {
            "KEEP": {"KEEP": int(matrix[0, 0]), "REVISE": int(matrix[0, 1])},
            "REVISE": {"KEEP": int(matrix[1, 0]), "REVISE": int(matrix[1, 1])},
        },
    }


def fit_linear(x_train: np.ndarray, y_train: np.ndarray, c: float):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=5000, random_state=SEED),
    )
    model.fit(x_train, y_train)
    return model


def tune_c(x_train: np.ndarray, y_train: np.ndarray, x_dev: np.ndarray, y_dev: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    scores = []
    for c in C_GRID:
        model = fit_linear(x_train, y_train, c)
        score = balanced_accuracy_score(y_dev, model.predict(x_dev))
        scores.append({"c": c, "dev_balanced_accuracy": float(score)})
    best = max(scores, key=lambda item: (item["dev_balanced_accuracy"], -item["c"]))
    return float(best["c"]), scores


def subgroup_metrics(rows: list[dict[str, Any]], indices: np.ndarray, y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    output = {"domain": {}, "dataset": {}}
    selected_rows = [rows[index] for index in indices]
    for field in ("domain", "dataset"):
        for value in sorted({row[field] for row in selected_rows}):
            mask = np.asarray([row[field] == value for row in selected_rows])
            output[field][value] = {"rows": int(mask.sum()), **metrics(y[mask], pred[mask], probability[mask])}
    return output


def tune_simple_feature(x: np.ndarray, y: np.ndarray, train: np.ndarray, dev: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    best_c, dev_curve = tune_c(x[train], y[train], x[dev], y[dev])
    final = fit_linear(x[np.concatenate([train, dev])], y[np.concatenate([train, dev])], best_c)
    probability = final.predict_proba(x[test])[:, 1]
    pred = (probability >= 0.5).astype(int)
    return {"selected_c": best_c, "dev_tuning": dev_curve, "test": metrics(y[test], pred, probability)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="phase3/runs/representation_probe")
    parser.add_argument("--baseline-label", default="decision_only_v1")
    parser.add_argument("--candidate-label", default="router_v3")
    parser.add_argument("--baseline-display", default="Decision-Only V1")
    parser.add_argument("--candidate-display", default="Router V3")
    args = parser.parse_args()
    model_labels = (args.baseline_label, args.candidate_label)
    run_dir = Path(args.run_dir).resolve()
    rows = read_jsonl(run_dir / "probe_dataset.jsonl")
    y = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    train = np.asarray([index for index, row in enumerate(rows) if row["split"] == "train"])
    dev = np.asarray([index for index, row in enumerate(rows) if row["split"] == "dev"])
    test = np.asarray([index for index, row in enumerate(rows) if row["split"] == "test"])
    train_dev = np.concatenate([train, dev])
    predictions = []
    all_results = {}

    for model_label in model_labels:
        archive = np.load(run_dir / "activations" / f"{model_label}_final_token.npz")
        activations = archive["activations"].astype(np.float32)
        layers = archive["layers"].tolist()
        source_ids = archive["source_ids"].tolist()
        if source_ids != [row["source_id"] for row in rows]:
            raise RuntimeError(f"Activation row order mismatch for {model_label}")
        layer_results = []
        fitted = {}
        for layer_index, layer in enumerate(layers):
            x = activations[:, layer_index, :]
            selected_c, dev_curve = tune_c(x[train], y[train], x[dev], y[dev])
            dev_model = fit_linear(x[train], y[train], selected_c)
            dev_probability = dev_model.predict_proba(x[dev])[:, 1]
            dev_pred = (dev_probability >= 0.5).astype(int)
            final_model = fit_linear(x[train_dev], y[train_dev], selected_c)
            test_probability = final_model.predict_proba(x[test])[:, 1]
            test_pred = (test_probability >= 0.5).astype(int)
            result = {
                "layer": int(layer),
                "selected_c": selected_c,
                "dev_tuning": dev_curve,
                "dev": metrics(y[dev], dev_pred, dev_probability),
                "test": metrics(y[test], test_pred, test_probability),
                "test_subgroups": subgroup_metrics(rows, test, y[test], test_pred, test_probability),
            }
            layer_results.append(result)
            fitted[int(layer)] = (selected_c, x, test_pred, test_probability)
            for local_index, row_index in enumerate(test):
                predictions.append({
                    "source_id": rows[row_index]["source_id"],
                    "dataset": rows[row_index]["dataset"],
                    "domain": rows[row_index]["domain"],
                    "expected_label": rows[row_index]["label"],
                    "expected_class_id": int(y[row_index]),
                    "model": model_label,
                    "layer": int(layer),
                    "predicted_label": "REVISE" if test_pred[local_index] else "KEEP",
                    "predicted_class_id": int(test_pred[local_index]),
                    "revise_probability": float(test_probability[local_index]),
                    "correct": bool(test_pred[local_index] == y[row_index]),
                })
        best = max(layer_results, key=lambda result: (result["dev"]["balanced_accuracy"], -result["layer"]))
        all_results[model_label] = {
            "selection_rule": "maximum dev balanced accuracy; ties choose earlier layer",
            "selected_layer": best["layer"],
            "selected_layer_dev_balanced_accuracy": best["dev"]["balanced_accuracy"],
            "selected_layer_test_balanced_accuracy": best["test"]["balanced_accuracy"],
            "layers": layer_results,
        }

    # Controls use no hidden representation other than the explicit random-label control.
    length_x = np.asarray([
        [np.log1p(row["answer_length_chars"]), np.log1p(row["answer_length_words"])] for row in rows
    ], dtype=np.float64)
    length_control = tune_simple_feature(length_x, y, train, dev, test)

    metadata_x = np.asarray([[row["dataset"], row["domain"]] for row in rows], dtype=object)
    metadata_dev_scores = []
    for c in C_GRID:
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=5000, random_state=SEED),
        )
        model.fit(metadata_x[train], y[train])
        metadata_dev_scores.append({"c": c, "dev_balanced_accuracy": float(balanced_accuracy_score(y[dev], model.predict(metadata_x[dev])))})
    metadata_c = float(max(metadata_dev_scores, key=lambda item: (item["dev_balanced_accuracy"], -item["c"]))["c"])
    metadata_model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(C=metadata_c, penalty="l2", solver="liblinear", max_iter=5000, random_state=SEED),
    )
    metadata_model.fit(metadata_x[train_dev], y[train_dev])
    metadata_probability = metadata_model.predict_proba(metadata_x[test])[:, 1]
    metadata_pred = (metadata_probability >= 0.5).astype(int)
    metadata_control = {
        "selected_c": metadata_c,
        "dev_tuning": metadata_dev_scores,
        "test": metrics(y[test], metadata_pred, metadata_probability),
    }

    random_controls = {}
    for model_label in model_labels:
        selected_layer = all_results[model_label]["selected_layer"]
        selected_result = next(item for item in all_results[model_label]["layers"] if item["layer"] == selected_layer)
        selected_c = selected_result["selected_c"]
        archive = np.load(run_dir / "activations" / f"{model_label}_final_token.npz")
        layer_index = archive["layers"].tolist().index(selected_layer)
        x = archive["activations"][:, layer_index, :].astype(np.float32)
        repeat_metrics = []
        for repeat in range(20):
            rng = np.random.default_rng(SEED + repeat)
            shuffled = rng.permutation(y[train_dev])
            model = fit_linear(x[train_dev], shuffled, selected_c)
            probability = model.predict_proba(x[test])[:, 1]
            pred = (probability >= 0.5).astype(int)
            repeat_metrics.append(metrics(y[test], pred, probability))
        bas = [item["balanced_accuracy"] for item in repeat_metrics]
        random_controls[model_label] = {
            "repeats": 20,
            "layer": selected_layer,
            "c": selected_c,
            "test_balanced_accuracy_mean": float(np.mean(bas)),
            "test_balanced_accuracy_std": float(np.std(bas)),
            "test_balanced_accuracy_min": float(np.min(bas)),
            "test_balanced_accuracy_max": float(np.max(bas)),
            "all_repeats": repeat_metrics,
        }

    controls = {
        "random_label": random_controls,
        "answer_length_only": length_control,
        "dataset_domain_only": metadata_control,
    }
    write_json(run_dir / "probe_results_by_layer.json", all_results)
    write_jsonl(run_dir / "probe_test_predictions.jsonl", predictions)
    write_json(run_dir / "probe_controls.json", controls)

    baseline = all_results[args.baseline_label]
    candidate = all_results[args.candidate_label]
    difference = candidate["selected_layer_test_balanced_accuracy"] - baseline["selected_layer_test_balanced_accuracy"]
    if difference > 0.03:
        conclusion = "candidate_representation_gain"
    elif max(baseline["selected_layer_test_balanced_accuracy"], candidate["selected_layer_test_balanced_accuracy"]) >= 0.75:
        conclusion = "strong_internal_error_signal"
    elif max(baseline["selected_layer_test_balanced_accuracy"], candidate["selected_layer_test_balanced_accuracy"]) >= 0.60:
        conclusion = "partial_internal_error_signal"
    else:
        conclusion = "weak_internal_error_signal"

    def selected_result(model_name: str) -> dict[str, Any]:
        model = all_results[model_name]
        return next(item for item in model["layers"] if item["layer"] == model["selected_layer"])

    baseline_best = selected_result(args.baseline_label)
    candidate_best = selected_result(args.candidate_label)
    summary = json.loads((run_dir / "probe_split_summary.json").read_text(encoding="utf-8"))
    lines = [
        "# Phase 3 Representation Probe",
        "",
        "## Conclusion",
        "",
        f"`{conclusion}`",
        "",
        "The language-model weights were frozen. Only regularized logistic-regression",
        "probes were fitted; layer and regularization selection used probe-dev only.",
        "",
        "## Data",
        "",
        f"- Sources: {summary['total_sources']} (maximum clean balanced size under current constraints).",
        f"- Train/dev/test: {summary['counts_by_split']['train']}/{summary['counts_by_split']['dev']}/{summary['counts_by_split']['test']}.",
        "- Every split is balanced KEEP/REVISE and math/code; no CW-as-wrong, HumanEval, frozen leakage, or source overlap.",
        "",
        "## Layer-wise test curve",
        "",
        f"| Layer | {args.baseline_display} balanced accuracy | {args.candidate_display} balanced accuracy |",
        "|---:|---:|---:|",
    ]
    candidate_by_layer = {item["layer"]: item for item in candidate["layers"]}
    for item in baseline["layers"]:
        layer = item["layer"]
        lines.append(f"| {layer} | {100*item['test']['balanced_accuracy']:.1f}% | {100*candidate_by_layer[layer]['test']['balanced_accuracy']:.1f}% |")
    lines += [
        "",
        "## Dev-selected probes",
        "",
        f"| Metric | {args.baseline_display} | {args.candidate_display} |",
        "|---|---:|---:|",
        f"| Selected layer | {baseline['selected_layer']} | {candidate['selected_layer']} |",
        f"| Test accuracy | {100*baseline_best['test']['accuracy']:.1f}% | {100*candidate_best['test']['accuracy']:.1f}% |",
        f"| Test balanced accuracy | {100*baseline_best['test']['balanced_accuracy']:.1f}% | {100*candidate_best['test']['balanced_accuracy']:.1f}% |",
        f"| KEEP recall | {100*baseline_best['test']['keep_recall']:.1f}% | {100*candidate_best['test']['keep_recall']:.1f}% |",
        f"| REVISE recall | {100*baseline_best['test']['revise_recall']:.1f}% | {100*candidate_best['test']['revise_recall']:.1f}% |",
        f"| ROC-AUC | {baseline_best['test']['roc_auc']:.3f} | {candidate_best['test']['roc_auc']:.3f} |",
        "",
        "## Controls",
        "",
        f"- Random labels (20 repeats), {args.baseline_display}: {100*random_controls[args.baseline_label]['test_balanced_accuracy_mean']:.1f}% ± {100*random_controls[args.baseline_label]['test_balanced_accuracy_std']:.1f} pp.",
        f"- Random labels (20 repeats), {args.candidate_display}: {100*random_controls[args.candidate_label]['test_balanced_accuracy_mean']:.1f}% ± {100*random_controls[args.candidate_label]['test_balanced_accuracy_std']:.1f} pp.",
        f"- Answer length only: {100*length_control['test']['balanced_accuracy']:.1f}% balanced accuracy.",
        f"- Dataset/domain only: {100*metadata_control['test']['balanced_accuracy']:.1f}% balanced accuracy.",
        "",
        "## Selected-layer subgroup results",
        "",
        f"| Group | {args.baseline_display} balanced accuracy | {args.candidate_display} balanced accuracy |",
        "|---|---:|---:|",
    ]
    for field in ("domain", "dataset"):
        for value in baseline_best["test_subgroups"][field]:
            a = baseline_best["test_subgroups"][field][value]["balanced_accuracy"]
            b = candidate_best["test_subgroups"][field][value]["balanced_accuracy"]
            lines.append(f"| {value} | {100*a:.1f}% | {100*b:.1f}% |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"The dev-selected representation probes differ by {100*difference:+.1f} pp on the 52-row test split.",
        f"ROC-AUC changes from {baseline_best['test']['roc_auc']:.3f} to {candidate_best['test']['roc_auc']:.3f}.",
        "Interpret this small probe test together with the frozen behavioral benchmark;",
        "a policy improvement without a probe gain is not evidence of a better correctness representation.",
        "The answer-length control reaching above chance also means part of the signal",
        "may be superficial.",
        "",
    ]
    (run_dir / "probe_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({
        "conclusion": conclusion,
        "baseline_label": args.baseline_label,
        "baseline_selected_layer": baseline["selected_layer"],
        "baseline_test_balanced_accuracy": baseline["selected_layer_test_balanced_accuracy"],
        "candidate_label": args.candidate_label,
        "candidate_selected_layer": candidate["selected_layer"],
        "candidate_test_balanced_accuracy": candidate["selected_layer_test_balanced_accuracy"],
        "controls": controls,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
