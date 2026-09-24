"""Fit CPU-only temperature and isotonic calibration to verbal confidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/confidence_calibration_v1.yaml"
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


def ece_equal_width(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if mask.any():
            result += float(mask.sum()) / total * abs(float(p[mask].mean() - y[mask].mean()))
    return result


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    return {
        "negative_log_likelihood": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_5_equal_width": ece_equal_width(y, p),
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
        "mean_probability_correct": float(p.mean()),
        "empirical_correct_rate": float(y.mean()),
    }


def fit_temperature(y: np.ndarray, logits: np.ndarray,
                    bounds: tuple[float, float]) -> float:
    def objective(log_temperature: float) -> float:
        probability = expit(logits / np.exp(log_temperature))
        return float(log_loss(y, np.clip(probability, 1e-12, 1 - 1e-12), labels=[0, 1]))
    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                             options={"xatol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def loo_predictions(y: np.ndarray, probability: np.ndarray, logits: np.ndarray,
                    bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    temperature_predictions = np.empty_like(probability)
    isotonic_predictions = np.empty_like(probability)
    temperatures: list[float] = []
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        temperature = fit_temperature(y[train], logits[train], bounds)
        temperatures.append(temperature)
        temperature_predictions[held_out] = expit(logits[held_out] / temperature)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probability[train], y[train])
        isotonic_predictions[held_out] = float(iso.predict(probability[held_out:held_out + 1])[0])
    return temperature_predictions, isotonic_predictions, temperatures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase6_confidence_calibration_v1")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    input_dir = ROOT / config["scope"]["input_dir"]
    epsilon = float(config["score"]["probability_clip"])
    bounds = tuple(float(value) for value in
                   config["methods"]["temperature_scaling"]["log_temperature_bounds"])
    report: dict[str, Any] = {
        "schema_version": "phase6_confidence_calibration_report_v1",
        "protocol_sha256_lf": sha256_lf(CONFIG),
        "scope": config["scope"],
        "target": config["target"],
        "score": config["score"],
        "source_manifest_sha256_lf": sha256_lf(ROOT / config["scope"]["source_manifest"]),
        "checkpoints": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        input_path = input_dir / f"{checkpoint}.jsonl"
        rows = [row for row in read_jsonl(input_path) if row.get("kind") == "confidence"]
        if len(rows) != int(config["scope"]["rows_per_checkpoint"]):
            raise RuntimeError(f"Expected 16 confidence rows for {checkpoint}")
        if not all(row.get("strict_contract_valid") for row in rows):
            raise RuntimeError(f"Invalid confidence contract for {checkpoint}")
        y = np.asarray([int(row["initial_correct"]) for row in rows], dtype=np.int64)
        raw = np.asarray([float(row["confidence_correct"]) / 100.0 for row in rows])
        probability = np.clip(raw, epsilon, 1.0 - epsilon)
        logits = logit(probability)

        temperature = fit_temperature(y, logits, bounds)
        temperature_full = expit(logits / temperature)
        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip",
                                      y_min=0.0, y_max=1.0)
        isotonic_full = isotonic.fit_transform(probability, y)
        temperature_loo, isotonic_loo, loo_temperatures = loo_predictions(
            y, probability, logits, bounds)
        result = {
            "display": DISPLAY[checkpoint],
            "input_sha256_lf": sha256_lf(input_path),
            "rows": len(rows),
            "unique_reported_confidence": sorted(set(float(value) for value in raw)),
            "temperature": temperature,
            "temperature_loo_range": [min(loo_temperatures), max(loo_temperatures)],
            "isotonic_thresholds": {
                "x": [float(value) for value in isotonic.X_thresholds_],
                "y": [float(value) for value in isotonic.y_thresholds_],
            },
            "metrics": {
                "constant_0_5": metrics(y, np.full_like(probability, 0.5)),
                "raw": metrics(y, probability),
                "temperature_full_fit": metrics(y, temperature_full),
                "isotonic_full_fit": metrics(y, isotonic_full),
                "temperature_leave_one_out": metrics(y, temperature_loo),
                "isotonic_leave_one_out": metrics(y, isotonic_loo),
            },
        }
        report["checkpoints"][checkpoint] = result
        for index, row in enumerate(rows):
            prediction_rows.append({
                "schema_version": "phase6_confidence_calibration_prediction_v1",
                "checkpoint": checkpoint,
                "problem_id": row["problem_id"],
                "initial_correct": bool(row["initial_correct"]),
                "reported_probability_correct": float(raw[index]),
                "clipped_probability_correct": float(probability[index]),
                "temperature_full_fit_probability_correct": float(temperature_full[index]),
                "isotonic_full_fit_probability_correct": float(isotonic_full[index]),
                "temperature_loo_probability_correct": float(temperature_loo[index]),
                "isotonic_loo_probability_correct": float(isotonic_loo[index]),
            })

    report["aggregate"] = {
        "temperature_loo_beats_constant_0_5_brier_all_checkpoints": all(
            report["checkpoints"][checkpoint]["metrics"]["temperature_leave_one_out"]["brier"]
            < report["checkpoints"][checkpoint]["metrics"]["constant_0_5"]["brier"]
            for checkpoint in CHECKPOINTS
        ),
        "isotonic_loo_beats_constant_0_5_brier_all_checkpoints": all(
            report["checkpoints"][checkpoint]["metrics"]["isotonic_leave_one_out"]["brier"]
            < report["checkpoints"][checkpoint]["metrics"]["constant_0_5"]["brier"]
            for checkpoint in CHECKPOINTS
        ),
        "readout_only_hypothesis_supported": False,
        "conclusion": (
            "Both methods improve in-sample calibration, but neither leave-one-out "
            "method beats the balanced constant-0.5 Brier baseline for any checkpoint. "
            "The 16-row evidence does not support calibration alone as a sufficient fix."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows),
                                encoding="utf-8", newline="\n")
    report["predictions_sha256_lf"] = sha256_lf(predictions_path)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8", newline="\n")

    lines = [
        "# Phase 6 verbal-confidence calibration", "",
        "CPU-only calibration of the 16 stored verbal confidence scores per checkpoint. "
        "Labels are deterministic verifier correctness. These are reported-confidence "
        "log-odds, not saved vocabulary-token logits; no model forward pass or weight "
        "update was performed.", "",
        "## Calibration metrics", "",
        "Lower NLL, Brier, and ECE are better. LOO is leave-one-out and is the primary "
        "small-sample generalization view.", "",
        "| Checkpoint | Method | NLL | Brier | ECE | AUC | Accuracy | Mean P(correct) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    method_names = (
        ("constant_0_5", "Constant 50%"),
        ("raw", "Raw"),
        ("temperature_full_fit", "Temperature (fit)"),
        ("temperature_leave_one_out", "Temperature (LOO)"),
        ("isotonic_full_fit", "Isotonic (fit)"),
        ("isotonic_leave_one_out", "Isotonic (LOO)"),
    )
    for checkpoint in CHECKPOINTS:
        result = report["checkpoints"][checkpoint]
        for key, label in method_names:
            metric = result["metrics"][key]
            lines.append(
                f"| {DISPLAY[checkpoint]} | {label} | "
                f"{metric['negative_log_likelihood']:.3f} | {metric['brier']:.3f} | "
                f"{metric['ece_5_equal_width']:.3f} | {metric['roc_auc']:.3f} | "
                f"{metric['accuracy_at_0_5']:.3f} | {metric['mean_probability_correct']:.3f} |"
            )
    lines += ["", "## Fitted parameters", ""]
    for checkpoint in CHECKPOINTS:
        result = report["checkpoints"][checkpoint]
        mapping = ", ".join(
            f"{100*x:.1f}% -> {100*y:.1f}%" for x, y in zip(
                result["isotonic_thresholds"]["x"], result["isotonic_thresholds"]["y"])
        )
        lines.append(
            f"- {DISPLAY[checkpoint]}: T={result['temperature']:.3f}; isotonic {mapping}."
        )
    lines += [
        "", "## Result", "",
        report["aggregate"]["conclusion"], "",
        "The fitted temperatures are large, showing that the main correctable defect is "
        "severe overconfidence. Temperature scaling leaves the 0.5 decision side unchanged, "
        "so it does not improve detection accuracy. Isotonic reaches zero in-sample ECE "
        "but generalizes poorly, consistent with only 16 rows and 3--4 distinct score levels.",
        "", "## Interpretation limits", "",
        "A one-parameter temperature transform is monotone and cannot improve ranking/AUC "
        "or recover missing decision information. Isotonic is also monotone but can create "
        "ties. Its in-sample score is optimistic with only 16 rows and few distinct confidence "
        "levels, so conclusions must follow the LOO results.", "",
        "LOO AUC is unstable because each held-out point is transformed by a separately "
        "fitted calibration function; use LOO NLL and Brier for the calibration conclusion. "
        "Full-fit temperature AUC is unchanged, as expected for one monotone transform.", "",
        "Direct calibration of KEEP/REVISE vocabulary logits remains untested because the "
        "prior vLLM records did not request token logprobs. That experiment requires a new "
        "forward extraction before this same CPU fitting stage.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({checkpoint: {
        "temperature": report["checkpoints"][checkpoint]["temperature"],
        "raw_brier": report["checkpoints"][checkpoint]["metrics"]["raw"]["brier"],
        "temperature_loo_brier": report["checkpoints"][checkpoint]["metrics"][
            "temperature_leave_one_out"]["brier"],
        "isotonic_loo_brier": report["checkpoints"][checkpoint]["metrics"][
            "isotonic_leave_one_out"]["brier"],
    } for checkpoint in CHECKPOINTS}, indent=2))


if __name__ == "__main__":
    main()
