"""Train a correctness router on frozen hidden states (no LLM forward pass)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
REVISE_WEIGHTS = (1.0, 1.25, 1.5, 2.0, 3.0)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.30, 0.71, 0.05))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "keep_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "revise_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "revise_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": matrix.tolist(),
    }


def fit(x: np.ndarray, y: np.ndarray, c: float, revise_weight: float, seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight={0: 1.0, 1: revise_weight},
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        ),
    )
    model.fit(x, y)
    return model


def subgroup(rows: list[dict[str, Any]], indices: np.ndarray, y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    selected = [rows[int(index)] for index in indices]
    output: dict[str, Any] = {"domain": {}, "dataset": {}}
    for field in ("domain", "dataset"):
        for value in sorted({row[field] for row in selected}):
            mask = np.asarray([row[field] == value for row in selected])
            output[field][value] = {"rows": int(mask.sum()), **measures(y[mask], pred[mask], prob[mask])}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="phase3/runs/representation_probe/probe_dataset.jsonl")
    parser.add_argument("--activations", default="phase3/runs/representation_probe/activations/decision_only_v1_final_token.npz")
    parser.add_argument(
        "--output-dir",
        default="outputs/phase3_five_stage_pipeline/stages/01_frozen_classifier",
    )
    parser.add_argument("--keep-recall-floor", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    started = time.time()
    dataset_path = Path(args.dataset).resolve()
    activation_path = Path(args.activations).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(dataset_path)
    y = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    train = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"])
    dev = np.asarray([i for i, row in enumerate(rows) if row["split"] == "dev"])
    test = np.asarray([i for i, row in enumerate(rows) if row["split"] == "test"])
    archive = np.load(activation_path, allow_pickle=False)
    activations = archive["activations"].astype(np.float32)
    layers = [int(value) for value in archive["layers"]]
    source_ids = [str(value) for value in archive["source_ids"]]
    if source_ids != [row["source_id"] for row in rows]:
        raise RuntimeError("Activation/dataset row order mismatch")

    candidates = []
    for layer_index, layer in enumerate(layers):
        x = activations[:, layer_index, :]
        for c in C_GRID:
            for revise_weight in REVISE_WEIGHTS:
                model = fit(x[train], y[train], c, revise_weight, args.seed)
                probability = model.predict_proba(x[dev])[:, 1]
                for threshold in THRESHOLDS:
                    pred = (probability >= threshold).astype(int)
                    metric = measures(y[dev], pred, probability)
                    candidates.append({
                        "layer": layer,
                        "layer_index": layer_index,
                        "c": c,
                        "revise_weight": revise_weight,
                        "threshold": threshold,
                        "dev": metric,
                        "eligible": metric["keep_recall"] >= args.keep_recall_floor,
                    })
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible or candidates
    selected = max(pool, key=lambda row: (row["dev"]["balanced_accuracy"], row["dev"]["revise_recall"], row["dev"]["keep_recall"], -row["layer"]))
    layer_index = int(selected["layer_index"])
    x = activations[:, layer_index, :]
    train_dev = np.concatenate([train, dev])
    final_model = fit(x[train_dev], y[train_dev], float(selected["c"]), float(selected["revise_weight"]), args.seed)
    probability = final_model.predict_proba(x[test])[:, 1]
    pred = (probability >= float(selected["threshold"])).astype(int)
    test_metrics = measures(y[test], pred, probability)
    model_path = output / "classifier.joblib"
    joblib.dump(final_model, model_path)
    predictions = []
    for local_index, row_index in enumerate(test):
        row = rows[int(row_index)]
        predictions.append({
            "source_id": row["source_id"],
            "dataset": row["dataset"],
            "domain": row["domain"],
            "expected_label": row["label"],
            "predicted_label": "REVISE" if pred[local_index] else "KEEP",
            "revise_probability": float(probability[local_index]),
            "correct": bool(pred[local_index] == y[row_index]),
        })
    prediction_path = output / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "phase3_frozen_classifier_v1",
        "objective": "KEEP/REVISE from frozen Decision-Only V1 final-token hidden states",
        "selection_rule": f"maximize dev balanced accuracy with KEEP recall >= {args.keep_recall_floor}; tie-break REVISE recall",
        "selected": selected,
        "frozen_test": test_metrics,
        "frozen_test_subgroups": subgroup(rows, test, y[test], pred, probability),
        "data": {"rows": len(rows), "train": len(train), "dev": len(dev), "test": len(test), "dataset_sha256": sha256(dataset_path)},
        "activation": {"path": str(activation_path), "sha256": sha256(activation_path), "layers": layers, "shape": list(activations.shape)},
        "output": {"classifier": str(model_path), "classifier_sha256": sha256(model_path), "predictions": str(prediction_path)},
        "runtime": {"python": platform.python_version(), "seconds": time.time() - started, "language_model_loaded": False, "gpu_required": False},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
