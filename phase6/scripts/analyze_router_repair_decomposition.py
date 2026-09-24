"""Separate detection, oracle repair, and frozen-probe harness evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.exceptions import InconsistentVersionWarning
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/router_repair_decomposition_v1.yaml"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
DISPLAY = {
    "original_solver": "Original solver",
    "warmstart_v2": "Warm-start V2",
    "correction_sft_v3": "Correction SFT V3",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def detector_metrics(y_wrong: np.ndarray, score_wrong: np.ndarray,
                     threshold: float) -> dict[str, Any]:
    routed = score_wrong >= threshold
    tp = int(np.sum(routed & (y_wrong == 1)))
    fp = int(np.sum(routed & (y_wrong == 0)))
    fn = int(np.sum(~routed & (y_wrong == 1)))
    tn = int(np.sum(~routed & (y_wrong == 0)))
    return {
        "threshold_probability_wrong": threshold,
        "rows": len(y_wrong), "routed": int(routed.sum()),
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision_wrong": safe_ratio(tp, tp + fp),
        "recall_wrong": safe_ratio(tp, tp + fn),
        "correct_preservation": safe_ratio(tn, tn + fp),
        "route_rate": float(routed.mean()),
    }


def status_repair_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [row for row in rows if not row["initial_correct"]]
    valid_revise = [row for row in wrong
                    if row.get("strict_contract_valid") and row.get("decision") == "REVISE"]
    verified = [row for row in wrong if row.get("final_correct") is True]
    return {
        "known_wrong_rows": len(wrong),
        "strict_valid_rows": sum(bool(row.get("strict_contract_valid")) for row in wrong),
        "valid_revise_rows": len(valid_revise),
        "revise_compliance_given_known_wrong": safe_ratio(len(valid_revise), len(wrong)),
        "verified_repairs_given_known_wrong": len(verified),
        "verified_repair_rate_given_known_wrong": safe_ratio(len(verified), len(wrong)),
        "verified_repairs_given_valid_revise": sum(
            row.get("final_correct") is True for row in valid_revise),
        "verified_repair_rate_given_valid_revise": safe_ratio(
            sum(row.get("final_correct") is True for row in valid_revise), len(valid_revise)
        ),
        "decisions": dict(Counter(row.get("decision") for row in wrong)),
    }


def fit_temperature(y: np.ndarray, probability: np.ndarray) -> float:
    logits = logit(np.clip(probability, 1e-6, 1 - 1e-6))
    def objective(log_temperature: float) -> float:
        p = expit(logits / np.exp(log_temperature))
        return float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]))
    result = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return {
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def selected_probe_probabilities(checkpoint: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selection_path = ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" / f"{checkpoint}_probe_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    archive_path = ROOT / "outputs/phase5_gpu_vllm/probe_v1/activations" / f"{checkpoint}_{split}.npz"
    archive = np.load(archive_path, allow_pickle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = joblib.load(ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" / f"{checkpoint}_probe.joblib")
    x = archive["activations"][:, int(selection["layer_index"])].astype(np.float32)
    return archive["problem_ids"].astype(str), archive["labels"].astype(int), model.predict_proba(x)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase6_router_repair_decomposition_v1")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    prediction_path = ROOT / config["scope"]["verbal_confidence_predictions"]
    pilot_path = ROOT / config["scope"]["verbal_confidence_manifest"]
    verbal_predictions = read_jsonl(prediction_path)
    pilot_ids = {row["problem_id"] for row in read_jsonl(pilot_path)}
    report: dict[str, Any] = {
        "schema_version": "phase6_router_repair_decomposition_report_v1",
        "protocol_sha256_lf": sha256_lf(CONFIG),
        "inputs": {
            "verbal_predictions_sha256_lf": sha256_lf(prediction_path),
            "verbal_pilot_sha256_lf": sha256_lf(pilot_path),
        },
        "scope": config["scope"],
        "checkpoints": {},
    }
    audit_rows: list[dict[str, Any]] = []
    protected_probe = json.loads((ROOT / "outputs/phase5_gpu_vllm/protected_v1"
                                  / "protected_probe_report.json").read_text(encoding="utf-8"))
    for checkpoint in CHECKPOINTS:
        confidence = [row for row in verbal_predictions if row["checkpoint"] == checkpoint]
        if len(confidence) != 16 or {row["problem_id"] for row in confidence} != pilot_ids:
            raise RuntimeError(f"Invalid verbal calibration inputs for {checkpoint}")
        y = np.asarray([int(not row["initial_correct"]) for row in confidence])
        verbal_curves = {}
        for method, key in (("temperature_leave_one_out", "temperature_loo_probability_correct"),
                            ("isotonic_leave_one_out", "isotonic_loo_probability_correct")):
            wrong_probability = np.asarray([1.0 - float(row[key]) for row in confidence])
            verbal_curves[method] = [detector_metrics(y, wrong_probability, float(threshold))
                                     for threshold in config["verbal_router"]
                                     ["thresholds_probability_wrong"]]
            for index, row in enumerate(confidence):
                audit_rows.append({
                    "schema_version": "phase6_verbal_router_audit_v1",
                    "checkpoint": checkpoint, "method": method,
                    "problem_id": row["problem_id"],
                    "initial_correct": bool(row["initial_correct"]),
                    "probability_wrong": float(wrong_probability[index]),
                })

        repair_by_split: dict[str, Any] = {}
        status_by_id_dev: dict[str, dict[str, Any]] = {}
        for split, directory in (("development", ROOT / "outputs/phase5_gpu_vllm/review_v1"),
                                 ("protected_existing", ROOT / "outputs/phase5_gpu_vllm/protected_v1/reviews")):
            path = directory / f"{checkpoint}_{'development' if split == 'development' else 'protected_test'}.jsonl"
            rows = [row for row in read_jsonl(path) if row["condition"] == "status"]
            repair_by_split[split] = status_repair_metrics(rows)
            if split == "development":
                status_by_id_dev = {row["problem_id"]: row for row in rows}

        train_ids, train_y, train_prob = selected_probe_probabilities(checkpoint, "train")
        dev_ids, dev_y, dev_prob = selected_probe_probabilities(checkpoint, "development")
        selection = json.loads((ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" /
                                f"{checkpoint}_probe_selection.json").read_text(encoding="utf-8"))
        selected_dev = selection["selected"]["development"]
        raw_at_half = detector_metrics(dev_y, dev_prob, 0.5)
        expected = selected_dev["confusion_matrix_correct_wrong"]
        observed = [[raw_at_half["true_negative"], raw_at_half["false_positive"]],
                    [raw_at_half["false_negative"], raw_at_half["true_positive"]]]
        if observed != expected:
            raise RuntimeError(f"Probe reload mismatch for {checkpoint}: {observed} != {expected}")
        temperature = fit_temperature(train_y, train_prob)
        dev_logits = logit(np.clip(dev_prob, 1e-6, 1 - 1e-6))
        dev_temp = expit(dev_logits / temperature)
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(train_prob, train_y)
        dev_iso = isotonic.predict(dev_prob)
        threshold = float(config["probe_harness"]["default_route_threshold_probability_wrong"])
        probe_options = {
            "raw": dev_prob,
            "temperature_train_calibrated": dev_temp,
            "isotonic_train_calibrated": dev_iso,
        }
        probe_harness: dict[str, Any] = {
            "calibration_fit_rows": len(train_y),
            "development_rows": len(dev_y),
            "temperature_fit_on_train": temperature,
            "raw_half_threshold_matches_frozen_selection": True,
            "probability_metrics_development": {
                name: probability_metrics(dev_y, value) for name, value in probe_options.items()
            },
            "route_threshold_probability_wrong": threshold,
            "variants": {},
            "protected_frozen_metric_reference": protected_probe["results"][checkpoint],
        }
        for name, probability in probe_options.items():
            detection = detector_metrics(dev_y, probability, threshold)
            true_positive_ids = {str(problem_id) for problem_id, label, score in
                                 zip(dev_ids, dev_y, probability)
                                 if label == 1 and score >= threshold}
            false_positive_ids = {str(problem_id) for problem_id, label, score in
                                  zip(dev_ids, dev_y, probability)
                                  if label == 0 and score >= threshold}
            cached_tp = [status_by_id_dev[problem_id] for problem_id in true_positive_ids]
            probe_harness["variants"][name] = {
                "detection_development": detection,
                "true_positive_status_repair_exact": status_repair_metrics(cached_tp),
                "false_positive_routes": len(false_positive_ids),
                "false_positive_repair_outcome": "not_observed_without_new_generation",
            }
            for problem_id, label, score in zip(dev_ids, dev_y, probability):
                audit_rows.append({
                    "schema_version": "phase6_probe_harness_audit_v1",
                    "checkpoint": checkpoint, "method": name,
                    "problem_id": str(problem_id), "initial_correct": bool(1 - label),
                    "probability_wrong": float(score),
                    "routed_at_0_5": bool(score >= threshold),
                })

        report["checkpoints"][checkpoint] = {
            "display": DISPLAY[checkpoint],
            "verbal_router_leave_one_out_curves": verbal_curves,
            "oracle_known_wrong_repair": repair_by_split,
            "probe_harness_development": probe_harness,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "audit_rows.jsonl"
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in audit_rows),
                          encoding="utf-8", newline="\n")
    report["audit_rows_sha256_lf"] = sha256_lf(audit_path)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8", newline="\n")

    lines = [
        "# Phase 6 router-repair decomposition", "",
        "CPU-only exploratory decomposition. It separates a verbal-confidence router, "
        "oracle-known-wrong repair, and a frozen-probe harness. No model, probe, prompt, "
        "or protected set was newly run; protected values below are pre-existing artifacts.", "",
        "## Oracle-known-wrong repair", "",
        "Status tells the repair model that the previous answer is wrong. It bypasses "
        "detection, so verified repair rate measures solver repair ability plus contract "
        "compliance, not autonomous self-correction.", "",
        "| Checkpoint | Split | Valid REVISE / known wrong | Verified repairs / known wrong | Verified repairs / valid REVISE |",
        "|---|---|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        for split, metric in report["checkpoints"][checkpoint]["oracle_known_wrong_repair"].items():
            lines.append(
                f"| {DISPLAY[checkpoint]} | {split} | {metric['valid_revise_rows']}/{metric['known_wrong_rows']} | "
                f"{metric['verified_repairs_given_known_wrong']}/{metric['known_wrong_rows']} | "
                f"{metric['verified_repairs_given_valid_revise']}/{metric['valid_revise_rows']} |"
            )
    lines += ["", "## Verbal-confidence router", "",
              "These curves apply LOO-calibrated verbal confidence deterministically. They "
              "measure who would be routed, not a newly generated response to a confidence prompt.", "",
              "| Checkpoint | Method | Threshold P(wrong) | Routed | Wrong recall | Precision | Correct preservation |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        for method, curve in report["checkpoints"][checkpoint]["verbal_router_leave_one_out_curves"].items():
            for metric in curve:
                precision = metric["precision_wrong"]
                preserve = metric["correct_preservation"]
                lines.append(
                    f"| {DISPLAY[checkpoint]} | {method} | {metric['threshold_probability_wrong']:.2f} | "
                    f"{metric['routed']}/16 | {metric['recall_wrong']:.3f} | "
                    f"{'NA' if precision is None else f'{precision:.3f}'} | "
                    f"{'NA' if preserve is None else f'{preserve:.3f}'} |"
                )
    lines += ["", "## Frozen-probe harness on development", "",
              "Probe temperature/isotonic calibration is fit on Phase 5 train predictions; "
              "development is held out for these calibration metrics. At route threshold 0.5, "
              "the true-positive repair column uses the exact cached status generation because "
              "those sources really were wrong. False-positive repair behavior is unobserved: "
              "their cached status prompt said no error, while a real harness would say revise.", "",
              "| Checkpoint | Probe variant | Dev routed | Wrong recall | False positives | TP repairs / TP routes |",
              "|---|---|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        variants = report["checkpoints"][checkpoint]["probe_harness_development"]["variants"]
        for name, value in variants.items():
            detection = value["detection_development"]
            repair = value["true_positive_status_repair_exact"]
            lines.append(
                f"| {DISPLAY[checkpoint]} | {name} | {detection['routed']}/80 | "
                f"{detection['recall_wrong']:.3f} | {value['false_positive_routes']} | "
                f"{repair['verified_repairs_given_known_wrong']}/{repair['known_wrong_rows']} |"
            )
    lines += ["", "## Probe-probability calibration check", "",
              "This is deliberately separate from the verbal-confidence calibration. The "
              "probe itself is a logistic classifier. Post-hoc calibrators are fit on its "
              "in-sample Phase 5 train predictions and tested on development, so they are "
              "a conservative warning about overfitting rather than deployable calibration.", "",
              "| Checkpoint | Variant | Dev NLL | Dev Brier | Dev AUC |",
              "|---|---|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        values = report["checkpoints"][checkpoint]["probe_harness_development"][
            "probability_metrics_development"]
        for name, metric in values.items():
            lines.append(
                f"| {DISPLAY[checkpoint]} | {name} | {metric['nll']:.3f} | "
                f"{metric['brier']:.3f} | {metric['roc_auc']:.3f} |"
            )
    lines += [
        "", "## Interpretation", "",
        "A calibrated verbal-confidence threshold can make routing gradual by construction, "
        "but it cannot establish that the generator uses uncertainty safely. The separate "
        "oracle repair rates show whether repair is a second bottleneck. A full probe-harness "
        "deployment claim requires a new, frozen non-protected run that sends a consistent "
        "router-triggered REVISE instruction to both true and false positives; that run is "
        "not reconstructed from mismatched cached status prompts. The raw logistic probe "
        "probabilities outperform both train-fit post-hoc calibrators on development, so no "
        "post-hoc probe calibrator or threshold is promoted.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({checkpoint: report["checkpoints"][checkpoint]["oracle_known_wrong_repair"]
        for checkpoint in CHECKPOINTS}, indent=2))


if __name__ == "__main__":
    main()
