"""Fit the frozen Phase 5 L2 probes on train and select only on development."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = yaml.safe_load((ROOT / "phase5/configs/review_protocol_v1.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "correct_recall": float(recall_score(y, pred, pos_label=0)),
        "wrong_recall": float(recall_score(y, pred, pos_label=1)),
        "confusion_matrix_correct_wrong": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.activation_dir / f"{args.checkpoint}_train.npz"
    dev_path = args.activation_dir / f"{args.checkpoint}_development.npz"
    train = np.load(train_path, allow_pickle=False)
    dev = np.load(dev_path, allow_pickle=False)
    train_ids, dev_ids = list(train["problem_ids"]), list(dev["problem_ids"])
    if set(train_ids) & set(dev_ids):
        raise RuntimeError("Train/development source overlap")
    layers = [int(value) for value in train["layers"]]
    if layers != [int(value) for value in dev["layers"]]:
        raise RuntimeError("Layer mismatch")
    y_train, y_dev = train["labels"].astype(int), dev["labels"].astype(int)
    if Counter(y_train) != Counter({0: 120, 1: 120}) or Counter(y_dev) != Counter({0: 40, 1: 40}):
        raise RuntimeError("Probe classes are not frozen/balanced")

    candidates: list[dict[str, Any]] = []
    fitted: dict[tuple[int, float], Any] = {}
    for layer_index, layer in enumerate(layers):
        for c in [float(value) for value in PROTOCOL["probe"]["c_grid"]]:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=5000,
                                   random_state=20260924),
            )
            model.fit(train["activations"][:, layer_index].astype(np.float32), y_train)
            probability = model.predict_proba(dev["activations"][:, layer_index].astype(np.float32))[:, 1]
            pred = (probability >= 0.5).astype(int)
            record = {"layer": layer, "layer_index": layer_index, "c": c,
                      "development": metrics(y_dev, pred, probability)}
            candidates.append(record)
            fitted[(layer, c)] = model
    selected = sorted(
        candidates,
        key=lambda row: (-row["development"]["balanced_accuracy"],
                         -row["development"]["roc_auc"], row["c"], row["layer"]),
    )[0]
    model = fitted[(selected["layer"], selected["c"])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.checkpoint}_probe.joblib"
    joblib.dump(model, model_path)
    report = {
        "schema_version": "phase5_prehint_probe_selection_v1",
        "checkpoint": args.checkpoint,
        "protected_test_opened": False,
        "selection_split": "development",
        "selection_rule": "max balanced_accuracy, then ROC-AUC, smaller C, lower layer",
        "selected": selected,
        "candidates": candidates,
        "train_rows": len(y_train),
        "development_rows": len(y_dev),
        "activation_hashes": {"train": sha256(train_path), "development": sha256(dev_path)},
        "probe_model": str(model_path),
        "probe_model_sha256": sha256(model_path),
    }
    report_path = args.output_dir / f"{args.checkpoint}_probe_selection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
