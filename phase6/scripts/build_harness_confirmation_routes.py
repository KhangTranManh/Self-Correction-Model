"""Build frozen recheck routes from fixed confidence and frozen-probe rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import expit, logit
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/harness_confirmation_v1.yaml"
HOLDOUT = ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl"
HOLDOUT_LOCK = ROOT / "phase6/data/harness_confirmation_holdout_v1_lock.json"
CALIBRATION = ROOT / "outputs/phase6_confidence_calibration_v1/report.json"
CONFIDENCE_DIR = ROOT / "outputs/phase6_harness_confirmation_v1/confidence"
ACTIVATION_DIR = ROOT / "outputs/phase6_harness_confirmation_v1/activations"
OUTPUT = ROOT / "phase6/data/harness_confirmation_routes_v1.jsonl"
LOCK = ROOT / "phase6/data/harness_confirmation_routes_v1_lock.json"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise RuntimeError("Routes already exist; refusing to overwrite")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    holdout = read_jsonl(HOLDOUT)
    by_id = {row["problem_id"]: row for row in holdout}
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    epsilon = float(config["self_confidence_router"]["probability_clip"])
    self_threshold = float(config["self_confidence_router"]["route_if_probability_wrong_at_least"])
    probe_threshold = float(config["frozen_probe_router"]["route_if_probability_wrong_at_least"])
    for checkpoint in CHECKPOINTS:
        confidence = read_jsonl(CONFIDENCE_DIR / f"confidence_{checkpoint}.jsonl")
        if len(confidence) != len(holdout) or not all(row.get("strict_contract_valid") for row in confidence):
            raise RuntimeError(f"Invalid confidence outputs: {checkpoint}")
        temperature = float(calibration["checkpoints"][checkpoint]["temperature"])
        for row in confidence:
            p_correct = np.clip(float(row["confidence_correct"]) / 100.0, epsilon, 1.0 - epsilon)
            p_wrong = float(1.0 - expit(logit(p_correct) / temperature))
            if p_wrong >= self_threshold:
                records.append({"checkpoint": checkpoint, "condition": "self_confidence",
                                "problem_id": row["problem_id"], "probability_wrong": p_wrong})
        archive = np.load(ACTIVATION_DIR / f"{checkpoint}.npz", allow_pickle=False)
        selection = json.loads((ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" /
                                f"{checkpoint}_probe_selection.json").read_text(encoding="utf-8"))["selected"]
        if int(archive["layer"][0]) != int(selection["layer"]):
            raise RuntimeError(f"Probe layer mismatch: {checkpoint}")
        model = joblib.load(ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" / f"{checkpoint}_probe.joblib")
        probabilities = model.predict_proba(archive["activations"].astype(np.float32))[:, 1]
        for problem_id, probability in zip(archive["problem_ids"], probabilities):
            if float(probability) >= probe_threshold:
                records.append({"checkpoint": checkpoint, "condition": "frozen_probe",
                                "problem_id": str(problem_id), "probability_wrong": float(probability)})
        for row in holdout:
            if not row["initial_correct"]:
                records.append({"checkpoint": checkpoint, "condition": "oracle_known_wrong",
                                "problem_id": row["problem_id"], "probability_wrong": 1.0})
    records.sort(key=lambda row: (row["checkpoint"], row["condition"], row["problem_id"]))
    unique = {(row["checkpoint"], row["condition"], row["problem_id"]) for row in records}
    if len(unique) != len(records):
        raise RuntimeError("Duplicate routes")
    if any(row["problem_id"] not in by_id for row in records):
        raise RuntimeError("Route references unknown source")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8", newline="\n")
    input_paths = [CONFIG, HOLDOUT, HOLDOUT_LOCK, CALIBRATION, OUTPUT]
    input_paths += [CONFIDENCE_DIR / f"confidence_{checkpoint}.jsonl" for checkpoint in CHECKPOINTS]
    input_paths += [ACTIVATION_DIR / f"{checkpoint}.npz" for checkpoint in CHECKPOINTS]
    input_paths += [ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" / f"{checkpoint}_probe.joblib" for checkpoint in CHECKPOINTS]
    lock = {"schema_version": "phase6_harness_confirmation_routes_lock_v1",
            "status": "frozen_before_recheck_generation",
            "files": {path.relative_to(ROOT).as_posix(): {"sha256_lf": sha256_lf(path)} for path in input_paths},
            "route_counts": {condition: sum(row["condition"] == condition for row in records)
                             for condition in ("self_confidence", "frozen_probe", "oracle_known_wrong")},
            "protected_data_used": False}
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

