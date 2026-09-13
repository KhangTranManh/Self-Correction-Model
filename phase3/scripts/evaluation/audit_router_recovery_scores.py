"""Audit Router Calibration V1 scores to target Router Recovery V2 data collection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def length_bin(value: int) -> str:
    if value <= 128:
        return "short_le_128"
    if value <= 512:
        return "medium_129_512"
    return "long_gt_512"


def token_bin(value: int) -> str:
    if value <= 256:
        return "short_le_256"
    if value <= 768:
        return "medium_257_768"
    return "long_gt_768"


def error_type(row: dict[str, Any]) -> str:
    if row["label"] == "KEEP":
        return "verified_correct"
    detail = str(row.get("verifier_detail", "")).lower()
    if row.get("domain") == "math":
        return "wrong_math_answer"
    if "timeout" in detail:
        return "timeout"
    if any(word in detail for word in ("syntax", "compile", "indentation")):
        return "compile_or_syntax"
    if any(word in detail for word in ("runtime", "exception", "nameerror", "typeerror")):
        return "runtime_error"
    return "executable_test_failure"


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["class_id"] for row in rows], dtype=np.int8)
    pred = np.asarray([row["prediction_id"] for row in rows], dtype=np.int8)
    prob = np.asarray([row["calibrated_revise_probability"] for row in rows], dtype=np.float64)
    labels = set(y.tolist())
    return {
        "rows": len(rows),
        "keep": int((y == 0).sum()),
        "revise": int((y == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(labels) == 2 else None,
        "keep_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)) if 0 in labels else None,
        "revise_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)) if 1 in labels else None,
        "roc_auc": float(roc_auc_score(y, prob)) if len(labels) == 2 else None,
        "mean_revise_probability": float(prob.mean()),
        "median_revise_probability": float(np.median(prob)),
        "errors": int((y != pred).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--base-classifier", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    activation_path = Path(args.activations).resolve()
    classifier_path = Path(args.base_classifier).resolve()
    calibrator_path = Path(args.calibrator).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(dataset_path)
    archive = np.load(activation_path, allow_pickle=False)
    artifact = joblib.load(calibrator_path)
    layer = int(artifact["layer"])
    layers = [int(value) for value in archive["layers"]]
    if layer not in layers:
        raise RuntimeError(f"Calibrator layer {layer} missing from activations")
    if [str(value) for value in archive["source_ids"]] != [str(row["source_id"]) for row in rows]:
        raise RuntimeError("Dataset and activation order differ")
    if artifact["base_classifier_sha256"] != sha256(classifier_path):
        raise RuntimeError("Base classifier hash differs from calibration artifact")
    if artifact["fit_dataset_sha256"] != sha256(dataset_path):
        raise RuntimeError("Dataset hash differs from calibration artifact")

    x = archive["activations"][:, layers.index(layer), :].astype(np.float32)
    base = joblib.load(classifier_path)
    raw_probability = base.predict_proba(x)[:, 1]
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    raw_logit = np.log(clipped / (1 - clipped))[:, None]
    calibrated = artifact["platt"].predict_proba(raw_logit)[:, 1]
    threshold = float(artifact["threshold"])

    scored = []
    for index, (row, raw, probability) in enumerate(zip(rows, raw_probability, calibrated, strict=True)):
        prediction_id = int(probability >= threshold)
        scored.append({
            "row_index": index,
            "source_id": row["source_id"],
            "dataset": row["dataset"],
            "domain": row["domain"],
            "calibration_role": row["calibration_role"],
            "label": row["label"],
            "class_id": int(row["class_id"]),
            "prediction": "REVISE" if prediction_id else "KEEP",
            "prediction_id": prediction_id,
            "correct": prediction_id == int(row["class_id"]),
            "raw_probe_revise_probability": float(raw),
            "calibrated_revise_probability": float(probability),
            "distance_to_threshold": float(probability - threshold),
            "answer_length_bin": length_bin(int(row.get("answer_length_chars", 0))),
            "input_token_bin": token_bin(int(row.get("input_token_length", 0))),
            "error_type": error_type(row),
            "verification_method": row.get("verification_method"),
            "bucket": row.get("bucket"),
        })

    grouped: dict[str, dict[str, Any]] = {}
    for field in ("calibration_role", "domain", "dataset", "error_type", "answer_length_bin", "input_token_bin"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            buckets[str(row[field])].append(row)
        grouped[field] = {key: group_metrics(value) for key, value in sorted(buckets.items())}

    misses = [row for row in scored if row["label"] == "REVISE" and not row["correct"]]
    priorities = {
        "schema_version": "phase3_router_recovery_v2_collection_priorities_v1",
        "basis": "fresh source-disjoint calibration V1 score errors",
        "total_revise_misses": len(misses),
        "by_domain": {key: len([row for row in misses if row["domain"] == key]) for key in ("code", "math")},
        "priority_order": [
            "executable code that fails substantive tests while matching the correct interface",
            "near-correct math with a genuine reasoning or final-answer error",
            "long confident answers predicted KEEP with high confidence",
        ],
        "do_not_collect": ["format-only failures", "empty answers", "easy syntax failures", "synthetic mutations"],
    }
    report = {
        "schema_version": "phase3_router_recovery_v2_score_audit_v1",
        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path), "rows": len(rows)},
        "activations_sha256": sha256(activation_path),
        "base_classifier_sha256": sha256(classifier_path),
        "calibrator_sha256": sha256(calibrator_path),
        "layer": layer,
        "threshold": threshold,
        "overall": group_metrics(scored),
        "groups": grouped,
        "revise_misses": len(misses),
        "keep_misses": len([row for row in scored if row["label"] == "KEEP" and not row["correct"]]),
        "weights_changed": False,
    }
    write_jsonl(output_dir / "score_rows.jsonl", scored)
    write_json(output_dir / "score_audit.json", report)
    write_json(output_dir / "collection_priorities.json", priorities)
    print(json.dumps({"overall": report["overall"], "revise_misses": len(misses), "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
