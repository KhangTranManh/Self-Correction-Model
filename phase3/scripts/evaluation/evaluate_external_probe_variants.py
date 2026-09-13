"""Fit probe variants on the 256-row development corpus and test on frozen-200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


C_GRID = (0.001, 0.01, 0.1, 1.0)
THRESHOLDS = tuple(float(x) for x in np.arange(0.30, 0.71, 0.05))
SEED = 20260907


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "keep_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "revise_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "revise_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "confusion_matrix": matrix.tolist(),
    }


def select_threshold(y: np.ndarray, score: np.ndarray, keep_floor: float) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in THRESHOLDS:
        item = metrics(y, (score >= threshold).astype(int), score)
        candidates.append((threshold, item))
    eligible = [item for item in candidates if item[1]["keep_recall"] >= keep_floor]
    pool = eligible or candidates
    return max(pool, key=lambda item: (item[1]["balanced_accuracy"], item[1]["revise_recall"], -item[0]))


def fit_linear(c: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="liblinear", max_iter=5000, random_state=SEED),
    )


def fit_mlp(hidden: int, alpha: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(hidden,), activation="tanh", solver="adam",
            alpha=alpha, batch_size=32, learning_rate_init=0.001,
            max_iter=200, early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=12, random_state=SEED,
        ),
    )


def select_and_evaluate(
    name: str,
    configurations: list[dict[str, Any]],
    factory: Callable[[dict[str, Any]], Any],
    train_x: np.ndarray,
    external_x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    dev: np.ndarray,
    external_y: np.ndarray,
    keep_floor: float,
) -> dict[str, Any]:
    candidates = []
    for configuration in configurations:
        print(f"[{name}] fitting {configuration}", flush=True)
        model = factory(configuration)
        model.fit(train_x[train], y[train])
        dev_score = model.predict_proba(train_x[dev])[:, 1]
        threshold, dev_metrics = select_threshold(y[dev], dev_score, keep_floor)
        candidates.append({"configuration": configuration, "threshold": threshold, "dev": dev_metrics})
    eligible = [row for row in candidates if row["dev"]["keep_recall"] >= keep_floor]
    selected = max(eligible or candidates, key=lambda row: (row["dev"]["balanced_accuracy"], row["dev"]["revise_recall"]))
    final = factory(selected["configuration"])
    train_dev = np.concatenate([train, dev])
    final.fit(train_x[train_dev], y[train_dev])
    score = final.predict_proba(external_x)[:, 1]
    fixed_pred = (score >= selected["threshold"]).astype(int)
    rank_threshold = float(np.median(score))
    rank_pred = (score >= rank_threshold).astype(int)
    return {
        "name": name,
        "selection": selected,
        "external_fixed_threshold": metrics(external_y, fixed_pred, score),
        "external_score_quantiles": {str(q): float(np.quantile(score, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "external_rank_balanced_diagnostic": {
            "assumption": "known frozen benchmark prevalence: 100 KEEP / 100 REVISE; no row label used to set threshold",
            "threshold": rank_threshold,
            **metrics(external_y, rank_pred, score),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--training-activations", required=True)
    parser.add_argument("--external-dataset", required=True)
    parser.add_argument("--external-activations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-recall-floor", type=float, default=0.70)
    args = parser.parse_args()

    training_rows = read_jsonl(Path(args.training_dataset))
    external_rows = read_jsonl(Path(args.external_dataset))
    training_archive = np.load(args.training_activations, allow_pickle=False)
    external_archive = np.load(args.external_activations, allow_pickle=False)
    layers = [int(x) for x in training_archive["layers"]]
    if layers != [int(x) for x in external_archive["layers"]]:
        raise RuntimeError("Training/external layer mismatch")
    if [str(x) for x in training_archive["source_ids"]] != [row["source_id"] for row in training_rows]:
        raise RuntimeError("Training activation order mismatch")
    if [str(x) for x in external_archive["source_ids"]] != [row["source_id"] for row in external_rows]:
        raise RuntimeError("External activation order mismatch")
    if {row["source_id"] for row in training_rows} & {row["source_id"] for row in external_rows}:
        raise RuntimeError("Training/external source leakage")

    activation = training_archive["activations"].astype(np.float32)
    external_activation = external_archive["activations"].astype(np.float32)
    y = np.asarray([int(row["class_id"]) for row in training_rows])
    external_y = np.asarray([int(row["class_id"]) for row in external_rows])
    train = np.asarray([i for i, row in enumerate(training_rows) if row["split"] == "train"])
    dev = np.asarray([i for i, row in enumerate(training_rows) if row["split"] == "dev"])

    variants = []
    linear_results = []
    for layer_index, layer in enumerate(layers):
        configs = [{"layer": layer, "c": c} for c in C_GRID]
        result = select_and_evaluate(
            f"single_layer_linear_{layer}", configs,
            lambda cfg: fit_linear(float(cfg["c"])),
            activation[:, layer_index, :], external_activation[:, layer_index, :],
            y, train, dev, external_y, args.keep_recall_floor,
        )
        linear_results.append(result)
    selected_linear = max(linear_results, key=lambda row: (row["selection"]["dev"]["balanced_accuracy"], row["selection"]["dev"]["revise_recall"]))
    selected_linear["name"] = "selected_single_layer_linear"
    selected_linear["all_layer_dev_selections"] = [
        {"name": row["name"], "selection": row["selection"]} for row in linear_results
    ]
    variants.append(selected_linear)
    concat = activation.reshape(len(training_rows), -1)
    external_concat = external_activation.reshape(len(external_rows), -1)
    variants.append(select_and_evaluate(
        "four_layer_concat_linear", [{"c": c} for c in C_GRID],
        lambda cfg: fit_linear(float(cfg["c"])),
        concat, external_concat, y, train, dev, external_y, args.keep_recall_floor,
    ))

    # Select a single layer jointly with a deliberately tiny MLP to limit capacity.
    mlp_results = []
    for layer_index, layer in enumerate(layers):
        configs = [{"layer": layer, "hidden": 8, "alpha": alpha} for alpha in (0.1, 1.0)]
        result = select_and_evaluate(
            f"small_mlp_layer_{layer}", configs,
            lambda cfg: fit_mlp(int(cfg["hidden"]), float(cfg["alpha"])),
            activation[:, layer_index, :], external_activation[:, layer_index, :],
            y, train, dev, external_y, args.keep_recall_floor,
        )
        mlp_results.append(result)
    selected_mlp = max(mlp_results, key=lambda row: (row["selection"]["dev"]["balanced_accuracy"], row["selection"]["dev"]["revise_recall"]))
    selected_mlp["name"] = "selected_small_mlp"
    selected_mlp["all_layer_dev_selections"] = [
        {"name": row["name"], "selection": row["selection"]} for row in mlp_results
    ]
    variants.append(selected_mlp)

    report = {
        "schema_version": "phase3_external_probe_variants_v1",
        "selection_policy": "all model/layer/hyperparameter/threshold selection uses original probe train/dev only",
        "external_policy": "frozen-200 labels used once for final reporting; rank diagnostic uses only known 50/50 benchmark design",
        "training_rows": len(training_rows),
        "external_rows": len(external_rows),
        "source_overlap": 0,
        "keep_recall_floor_on_dev": args.keep_recall_floor,
        "variants": variants,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
