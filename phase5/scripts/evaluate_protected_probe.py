"""Evaluate already-selected frozen probes once on protected activations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--opening-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.opening_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "ready_for_single_protected_opening":
        raise RuntimeError("Protected-opening lock is not ready")
    if lock.get("protected_results_read") or lock.get("protected_open_count") != 0:
        raise RuntimeError("Protected evaluation has already been opened")

    results = {}
    for checkpoint in CHECKPOINTS:
        selection_path = args.probe_dir / "selection" / f"{checkpoint}_probe_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
        frozen = lock["probe_selection"][checkpoint]
        observed = {key: selection[key] for key in ("layer", "layer_index", "c")}
        if observed != frozen:
            raise RuntimeError(f"Selection changed after lock: {checkpoint}")
        activation_path = args.probe_dir / "activations" / f"{checkpoint}_protected_test.npz"
        archive = np.load(activation_path, allow_pickle=False)
        y = archive["labels"].astype(int)
        if len(y) != 160 or int((y == 0).sum()) != 80 or int((y == 1).sum()) != 80:
            raise RuntimeError(f"Invalid protected composition: {checkpoint}")
        model_path = args.probe_dir / "selection" / f"{checkpoint}_probe.joblib"
        model = joblib.load(model_path)
        x = archive["activations"][:, int(selection["layer_index"])].astype(np.float32)
        probability = model.predict_proba(x)[:, 1]
        pred = (probability >= 0.5).astype(int)
        results[checkpoint] = {
            "selected_layer": int(selection["layer"]),
            "selected_c": float(selection["c"]),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "correct_recall": float(recall_score(y, pred, pos_label=0)),
            "wrong_recall": float(recall_score(y, pred, pos_label=1)),
            "confusion_matrix_correct_wrong": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "activation_sha256": sha256(activation_path),
            "probe_model_sha256": sha256(model_path),
        }
    report = {
        "schema_version": "phase5_prehint_probe_protected_v1",
        "protected_open_count": 1,
        "selection_changed_after_opening": False,
        "opening_lock_sha256": sha256(args.opening_lock),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
