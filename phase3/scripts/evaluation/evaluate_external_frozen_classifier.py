"""Evaluate the preselected frozen classifier on the disjoint frozen-200 set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "keep_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "revise_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "revise_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    activation_path = Path(args.activations).resolve()
    classifier_path = Path(args.classifier).resolve()
    selection_path = Path(args.selection_report).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(dataset_path)
    archive = np.load(activation_path, allow_pickle=False)
    activations = archive["activations"].astype(np.float32)
    layers = [int(value) for value in archive["layers"]]
    source_ids = [str(value) for value in archive["source_ids"]]
    if source_ids != [str(row["source_id"]) for row in rows]:
        raise RuntimeError("Activation/dataset row order mismatch")
    if len(rows) != 200 or len(set(source_ids)) != 200:
        raise RuntimeError("External frozen evaluation requires 200 unique rows")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    layer = int(selection["layer"])
    threshold = float(selection["threshold"])
    if layer not in layers:
        raise RuntimeError(f"Preselected layer {layer} absent from {layers}")
    model = joblib.load(classifier_path)
    x = activations[:, layers.index(layer), :]
    y = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    probability = model.predict_proba(x)[:, 1]
    pred = (probability >= threshold).astype(np.int64)
    overall = measures(y, pred, probability)

    subgroups: dict[str, dict[str, Any]] = {"domain": {}, "dataset": {}}
    for field in subgroups:
        for value in sorted({str(row[field]) for row in rows}):
            mask = np.asarray([str(row[field]) == value for row in rows])
            subgroup_y = y[mask]
            subgroup_pred = pred[mask]
            subgroup_probability = probability[mask]
            item = {"rows": int(mask.sum())}
            if len(set(subgroup_y.tolist())) == 2:
                item.update(measures(subgroup_y, subgroup_pred, subgroup_probability))
            else:
                item.update({
                    "accuracy": float(np.mean(subgroup_y == subgroup_pred)),
                    "label": "REVISE" if int(subgroup_y[0]) else "KEEP",
                })
            subgroups[field][value] = item

    rng = np.random.default_rng(args.seed)
    bootstrap = defaultdict(list)
    for _ in range(args.bootstrap_repeats):
        indices = rng.integers(0, len(rows), len(rows))
        sample_y = y[indices]
        if len(set(sample_y.tolist())) < 2:
            continue
        sample = measures(sample_y, pred[indices], probability[indices])
        for key in ("balanced_accuracy", "keep_recall", "revise_recall", "roc_auc"):
            bootstrap[key].append(sample[key])
    intervals = {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in bootstrap.items()
    }

    predictions = []
    for index, row in enumerate(rows):
        predictions.append({
            "source_id": row["source_id"],
            "dataset": row["dataset"],
            "domain": row["domain"],
            "expected_label": row["label"],
            "predicted_label": "REVISE" if pred[index] else "KEEP",
            "revise_probability": float(probability[index]),
            "correct": bool(pred[index] == y[index]),
        })
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": "phase3_external_frozen_classifier_eval_v1",
        "evaluation_role": "strict external test; no fitting, layer selection, or threshold tuning performed",
        "source_overlap_with_training_probe": 0,
        "precommitted_selection": {"layer": layer, "threshold": threshold},
        "metrics": overall,
        "bootstrap_95": intervals,
        "subgroups": subgroups,
        "artifacts": {
            "dataset_sha256": sha256(dataset_path),
            "activations_sha256": sha256(activation_path),
            "classifier_sha256": sha256(classifier_path),
            "selection_report_sha256": sha256(selection_path),
        },
        "gate": {
            "balanced_accuracy_above_native_0_65": overall["balanced_accuracy"] > 0.65,
            "revise_recall_above_native_0_49": overall["revise_recall"] > 0.49,
            "keep_recall_at_least_0_70": overall["keep_recall"] >= 0.70,
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
