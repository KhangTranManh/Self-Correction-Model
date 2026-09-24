"""Run preregistered surface-feature and shuffled-label probe controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = yaml.safe_load((ROOT / "phase5/configs/experiments.yaml").read_text(encoding="utf-8"))
PROTOCOL = yaml.safe_load((ROOT / "phase5/configs/review_protocol_v1.yaml").read_text(encoding="utf-8"))
TRAIN = ROOT / "phase5/data/splits/v1/train.jsonl"
DEV = ROOT / "phase5/data/splits/v1/development.jsonl"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
STEP_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:bước|step)\s*\d+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(x: np.ndarray, y: np.ndarray, c: float, seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=5000,
                           random_state=seed),
    )
    model.fit(x, y)
    return model


def score(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    pred = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        REGISTRY["checkpoint_hub_repos"]["original_solver"],
        revision=REGISTRY["checkpoint_hub_revisions"]["original_solver"], token=token,
    )
    train_rows, dev_rows = read_jsonl(TRAIN), read_jsonl(DEV)
    y_train = np.asarray([0 if row["initial_correct"] else 1 for row in train_rows])
    y_dev = np.asarray([0 if row["initial_correct"] else 1 for row in dev_rows])

    def surface(rows: list[dict[str, Any]]) -> np.ndarray:
        values = []
        for row in rows:
            text = row["initial_output"]
            values.append([
                len(text),
                len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                len(NUMBER_RE.findall(text)),
                len(STEP_RE.findall(text)),
                int(row["generated_tokens"]),
            ])
        return np.asarray(values, dtype=np.float32)

    x_surface_train, x_surface_dev = surface(train_rows), surface(dev_rows)
    surface_candidates = []
    for c in [float(value) for value in PROTOCOL["probe"]["c_grid"]]:
        model = fit(x_surface_train, y_train, c, 20260924)
        surface_candidates.append({"c": c, "development": score(model, x_surface_dev, y_dev)})
    selected_surface = sorted(
        surface_candidates,
        key=lambda row: (-row["development"]["balanced_accuracy"],
                         -row["development"]["roc_auc"], row["c"]),
    )[0]

    repetitions = int(PROTOCOL["probe"]["controls"]["shuffled_labels"]["repetitions"])
    seed = int(PROTOCOL["probe"]["controls"]["shuffled_labels"]["seed"])
    checkpoint_controls: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        report_path = args.selection_dir / f"{checkpoint}_probe_selection.json"
        selection = json.loads(report_path.read_text(encoding="utf-8"))["selected"]
        train_npz = np.load(args.activation_dir / f"{checkpoint}_train.npz", allow_pickle=False)
        dev_npz = np.load(args.activation_dir / f"{checkpoint}_development.npz", allow_pickle=False)
        if list(train_npz["problem_ids"]) != [row["problem_id"] for row in train_rows]:
            raise RuntimeError(f"Train row mismatch for {checkpoint}")
        if list(dev_npz["problem_ids"]) != [row["problem_id"] for row in dev_rows]:
            raise RuntimeError(f"Development row mismatch for {checkpoint}")
        layer_index = int(selection["layer_index"])
        x_train = train_npz["activations"][:, layer_index].astype(np.float32)
        x_dev = dev_npz["activations"][:, layer_index].astype(np.float32)
        rng = np.random.default_rng(seed)
        shuffled_scores = []
        for repetition in range(repetitions):
            shuffled_y = rng.permutation(y_train)
            model = fit(x_train, shuffled_y, float(selection["c"]), seed + repetition)
            shuffled_scores.append(score(model, x_dev, y_dev)["balanced_accuracy"])
        observed = float(selection["development"]["balanced_accuracy"])
        checkpoint_controls[checkpoint] = {
            "selected_layer": int(selection["layer"]),
            "selected_c": float(selection["c"]),
            "observed_development_balanced_accuracy": observed,
            "shuffle_repetitions": repetitions,
            "shuffle_balanced_accuracy": {
                "mean": float(np.mean(shuffled_scores)),
                "std": float(np.std(shuffled_scores)),
                "min": float(np.min(shuffled_scores)),
                "median": float(np.median(shuffled_scores)),
                "max": float(np.max(shuffled_scores)),
                "p_ge_observed_plus_one": float((1 + sum(v >= observed for v in shuffled_scores)) /
                                                  (repetitions + 1)),
            },
        }

    report = {
        "schema_version": "phase5_prehint_probe_controls_v1",
        "protected_test_opened": False,
        "surface_feature_definitions": {
            "initial_output_character_count": "Python len(initial_output)",
            "initial_output_token_count": "pinned base tokenizer, no special tokens",
            "digit_count": "count numeric spans matching -?digits with optional decimal/comma",
            "displayed_step_count": "case-insensitive line-start Bước/Step plus integer",
            "generated_tokens": "as recorded during frozen initial generation",
        },
        "surface_control": {
            "features": PROTOCOL["probe"]["controls"]["surface_features"],
            "candidates": surface_candidates,
            "selected": selected_surface,
        },
        "shuffled_label_controls": checkpoint_controls,
        "inputs": {
            "train_sha256": sha256(TRAIN),
            "development_sha256": sha256(DEV),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
