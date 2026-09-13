"""Fit a scalar Platt calibrator without changing the frozen probe or LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))[:, None]


def metrics(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "keep_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "revise_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(set(y.tolist())) == 2 else None,
        "brier": float(brier_score_loss(y, probability)),
        "confusion_matrix": matrix.tolist(),
    }


def select_threshold(y: np.ndarray, probability: np.ndarray, keep_floor: float) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        result = metrics(y, (probability >= threshold).astype(int), probability)
        if result["keep_recall"] >= keep_floor:
            candidates.append((float(threshold), result))
    if not candidates:
        raise RuntimeError("No OOF threshold satisfies the preregistered KEEP-recall floor")
    return max(candidates, key=lambda item: (item[1]["balanced_accuracy"], item[1]["revise_recall"], -item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--base-classifier", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--keep-recall-floor", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--external-dataset")
    parser.add_argument("--external-activations")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    activation_path = Path(args.activations).resolve()
    classifier_path = Path(args.base_classifier).resolve()
    rows = read_jsonl(dataset_path)
    archive = np.load(activation_path, allow_pickle=False)
    layers = [int(value) for value in archive["layers"]]
    if args.layer not in layers:
        raise RuntimeError(f"Layer {args.layer} not present")
    if [str(x) for x in archive["source_ids"]] != [row["source_id"] for row in rows]:
        raise RuntimeError("Calibration activation order mismatch")
    x = archive["activations"][:, layers.index(args.layer), :].astype(np.float32)
    y = np.asarray([int(row["class_id"]) for row in rows])
    balanced = np.asarray([i for i, row in enumerate(rows) if row["calibration_role"] == "balanced_fit"])
    natural = np.asarray([i for i, row in enumerate(rows) if row["calibration_role"] == "natural_prevalence_audit"])
    if len(balanced) != 100 or len(natural) != 40:
        raise RuntimeError("Expected 100 balanced-fit and 40 natural-audit rows")

    base = joblib.load(classifier_path)
    base_probability = base.predict_proba(x)[:, 1]
    base_logit = logit(base_probability)
    oof = np.zeros(len(balanced), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    balanced_y = y[balanced]
    for train_local, test_local in splitter.split(base_logit[balanced], balanced_y):
        platt = LogisticRegression(C=1.0, solver="lbfgs", random_state=args.seed)
        platt.fit(base_logit[balanced][train_local], balanced_y[train_local])
        oof[test_local] = platt.predict_proba(base_logit[balanced][test_local])[:, 1]
    threshold, oof_metrics = select_threshold(balanced_y, oof, args.keep_recall_floor)
    final_platt = LogisticRegression(C=1.0, solver="lbfgs", random_state=args.seed)
    final_platt.fit(base_logit[balanced], balanced_y)

    natural_probability = final_platt.predict_proba(base_logit[natural])[:, 1]
    natural_metrics = metrics(y[natural], (natural_probability >= threshold).astype(int), natural_probability)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "platt": final_platt,
        "threshold": threshold,
        "layer": args.layer,
        "base_classifier_sha256": sha256(classifier_path),
        "fit_dataset_sha256": sha256(dataset_path),
    }
    artifact_path = output / "calibrator.joblib"
    joblib.dump(artifact, artifact_path)

    external = None
    if args.external_dataset or args.external_activations:
        if not args.external_dataset or not args.external_activations:
            raise RuntimeError("External dataset and activations must be provided together")
        external_rows = read_jsonl(Path(args.external_dataset))
        external_archive = np.load(args.external_activations, allow_pickle=False)
        if [str(x) for x in external_archive["source_ids"]] != [row["source_id"] for row in external_rows]:
            raise RuntimeError("External activation order mismatch")
        external_x = external_archive["activations"][:, layers.index(args.layer), :].astype(np.float32)
        external_y = np.asarray([int(row["class_id"]) for row in external_rows])
        external_base_probability = base.predict_proba(external_x)[:, 1]
        external_probability = final_platt.predict_proba(logit(external_base_probability))[:, 1]
        external = {
            "role": "exploratory_only_because_frozen_200_was_previously_inspected",
            "metrics": metrics(external_y, (external_probability >= threshold).astype(int), external_probability),
        }

    report = {
        "schema_version": "phase3_router_calibration_result_v1",
        "weights_changed": False,
        "selection": "5-fold out-of-fold Platt probabilities on balanced_fit only",
        "selected_threshold": threshold,
        "balanced_fit_oof": oof_metrics,
        "natural_audit": natural_metrics,
        "external_frozen_200": external,
        "artifact": {"path": str(artifact_path), "sha256": sha256(artifact_path)},
        "gate": {
            "oof_balanced_accuracy_at_least_0_67": oof_metrics["balanced_accuracy"] >= 0.67,
            "oof_keep_recall_at_least_0_70": oof_metrics["keep_recall"] >= 0.70,
            "oof_revise_recall_at_least_0_60": oof_metrics["revise_recall"] >= 0.60,
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
