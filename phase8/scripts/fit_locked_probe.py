"""Rebuild a Phase 5-style probe with historical layer and C fixed in advance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {
    "original_solver": (14, 0.01),
    "warmstart_v2": (14, 0.01),
    "correction_sft_v3": (14, 1.0),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=SETTINGS)
    parser.add_argument("--activation-dir", type=Path,
                        default=ROOT / "outputs/phase8_probe_v1/activations")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase8_probe_v1/selection")
    args = parser.parse_args()
    layer, c = SETTINGS[args.checkpoint]
    train_path = args.activation_dir / f"{args.checkpoint}_train.npz"
    dev_path = args.activation_dir / f"{args.checkpoint}_development.npz"
    train = np.load(train_path, allow_pickle=False)
    dev = np.load(dev_path, allow_pickle=False)
    expected = {
        split: read_jsonl(ROOT / f"phase5/data/splits/v1/{split}.jsonl")
        for split in ("train", "development")
    }
    for split, archive in (("train", train), ("development", dev)):
        ids = [str(value) for value in archive["problem_ids"]]
        if ids != [row["problem_id"] for row in expected[split]]:
            raise RuntimeError(f"{split} activation IDs/order changed")
        labels = archive["labels"].astype(int).tolist()
        if labels != [0 if row["initial_correct"] else 1 for row in expected[split]]:
            raise RuntimeError(f"{split} activation labels changed")
        count = 120 if split == "train" else 40
        if Counter(labels) != Counter({0: count, 1: count}):
            raise RuntimeError(f"{split} class balance changed")
    layers = [int(value) for value in train["layers"]]
    if layers != [int(value) for value in dev["layers"]] or layer not in layers:
        raise RuntimeError("Layer 14 absent or inconsistent")
    index = layers.index(layer)
    x_train = train["activations"][:, index].astype(np.float32)
    y_train = train["labels"].astype(int)
    x_dev = dev["activations"][:, index].astype(np.float32)
    y_dev = dev["labels"].astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, penalty="l2", solver="liblinear", max_iter=5000,
        random_state=20260924))
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_dev)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.checkpoint}_probe.joblib"
    if model_path.exists():
        raise RuntimeError(f"Refusing to overwrite probe: {model_path}")
    joblib.dump(model, model_path)
    report = {
        "schema_version": "phase8_locked_rebuilt_probe_v1",
        "checkpoint": args.checkpoint,
        "selection_rule": "historical Phase 5 layer and C fixed before Phase 8 outputs",
        "selected": {"layer": layer, "layer_index": index, "c": c},
        "threshold": 0.5,
        "train_rows": len(y_train), "development_rows": len(y_dev),
        "activation_hashes": {"train": sha256(train_path), "development": sha256(dev_path)},
        "development_audit": {
            "balanced_accuracy": float(balanced_accuracy_score(y_dev, prediction)),
            "roc_auc": float(roc_auc_score(y_dev, probability)),
            "correct_recall": float(recall_score(y_dev, prediction, pos_label=0)),
            "wrong_recall": float(recall_score(y_dev, prediction, pos_label=1)),
        },
        "probe_model_sha256": sha256(model_path),
        "protected_data_used": False,
    }
    report_path = args.output_dir / f"{args.checkpoint}_probe_selection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
