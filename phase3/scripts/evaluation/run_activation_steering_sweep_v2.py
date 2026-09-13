"""Causal activation-steering sweep for the frozen Decision-Only V1 router.

The script never updates model weights. Probe directions are fitted only on the
original probe train+dev split, then intervened on a source-disjoint dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


CHOICES = {"KEEP": "<decision>KEEP</decision>", "REVISE": "<decision>REVISE</decision>"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(rows: list[dict[str, Any]], per_cell: int) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["domain"]), str(row["label"]))].append(row)
    required = {("code", "KEEP"), ("code", "REVISE"), ("math", "KEEP"), ("math", "REVISE")}
    if not required.issubset(cells):
        raise RuntimeError(f"Missing evaluation cells: {sorted(required - set(cells))}")
    selected = []
    for key in sorted(required):
        candidates = sorted(cells[key], key=lambda row: hashlib.sha256(str(row["source_id"]).encode()).hexdigest())
        if len(candidates) < per_cell:
            raise RuntimeError(f"Cell {key} has {len(candidates)} rows, need {per_cell}")
        selected.extend(candidates[:per_cell])
    return selected


def fit_directions(
    rows: list[dict[str, Any]], archive_path: Path, layers: list[int], c: float
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, float]]]:
    archive = np.load(archive_path, allow_pickle=False)
    archive_layers = [int(value) for value in archive["layers"]]
    source_ids = [str(value) for value in archive["source_ids"]]
    if source_ids != [str(row["source_id"]) for row in rows]:
        raise RuntimeError("Probe activation order does not match probe dataset")
    fit = np.asarray([index for index, row in enumerate(rows) if row["split"] in {"train", "dev"}])
    y = np.asarray([int(row["class_id"]) for row in rows])
    directions: dict[int, dict[str, np.ndarray]] = {}
    metadata: dict[int, dict[str, float]] = {}
    for layer in layers:
        if layer not in archive_layers:
            raise RuntimeError(f"Layer {layer} absent from {archive_path}")
        x = archive["activations"][:, archive_layers.index(layer), :].astype(np.float32)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="liblinear", max_iter=5000, random_state=20260913),
        )
        classifier.fit(x[fit], y[fit])
        scaler = classifier.named_steps["standardscaler"]
        logistic = classifier.named_steps["logisticregression"]
        probe = logistic.coef_[0].astype(np.float32) / scaler.scale_.astype(np.float32)
        caa = x[fit][y[fit] == 1].mean(axis=0) - x[fit][y[fit] == 0].mean(axis=0)
        probe_unit = probe / np.linalg.norm(probe)
        caa_unit = caa / np.linalg.norm(caa)
        typical_hidden_l2 = float(np.median(np.linalg.norm(x[fit], axis=1)))
        # One alpha unit changes the residual by 5% of a typical hidden-state
        # L2 norm. This makes probe and CAA magnitudes directly comparable.
        base_l2 = 0.05 * typical_hidden_l2
        directions[layer] = {
            "probe": (probe_unit * base_l2).astype(np.float32),
            "caa": (caa_unit * base_l2).astype(np.float32),
        }
        metadata[layer] = {
            "typical_hidden_l2": typical_hidden_l2,
            "alpha_unit_delta_l2": base_l2,
            "probe_caa_cosine": float(np.dot(probe_unit, caa_unit)),
            "fit_rows": int(len(fit)),
        }
    return directions, metadata


def metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = np.asarray([0 if row["expected_label"] == "KEEP" else 1 for row in records])
    predicted = np.asarray([0 if row["preference"] == "KEEP" else 1 for row in records])
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    return {
        "rows": len(records),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "keep_recall": float(recall_score(expected, predicted, labels=[0], average=None, zero_division=0)[0]),
        "revise_recall": float(recall_score(expected, predicted, labels=[1], average=None, zero_division=0)[0]),
        "revise_rate": float(predicted.mean()),
        "mean_log_odds_revise": float(np.mean([row["log_odds_revise"] for row in records])),
        "confusion_matrix": {
            "KEEP": {"KEEP": int(matrix[0, 0]), "REVISE": int(matrix[0, 1])},
            "REVISE": {"KEEP": int(matrix[1, 0]), "REVISE": int(matrix[1, 1])},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--probe-dataset", required=True)
    parser.add_argument("--probe-activations", required=True)
    parser.add_argument("--base-model", default="Kxck/Self_Correction_v1")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--layers", default="14,21,28")
    parser.add_argument("--directions", default="probe,caa")
    parser.add_argument("--token-scopes", default="final")
    parser.add_argument("--alphas", default="-2,0,2")
    parser.add_argument("--per-domain-label", type=int, default=5)
    parser.add_argument("--probe-c", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    started = time.time()
    dataset_path = Path(args.dataset).resolve()
    probe_dataset_path = Path(args.probe_dataset).resolve()
    probe_activation_path = Path(args.probe_activations).resolve()
    output = Path(args.output_dir).resolve()
    rows = select_rows(read_jsonl(dataset_path), args.per_domain_label)
    probe_rows = read_jsonl(probe_dataset_path)
    layers = [int(value) for value in args.layers.split(",")]
    direction_names = [value.strip() for value in args.directions.split(",")]
    scopes = [value.strip() for value in args.token_scopes.split(",")]
    alphas = [float(value) for value in args.alphas.split(",")]
    if set(direction_names) - {"probe", "caa"}:
        raise RuntimeError("Directions must be probe and/or caa")
    if set(scopes) - {"final", "last_32", "all_prompt"}:
        raise RuntimeError("Token scopes must be final, last_32, and/or all_prompt")

    direction_vectors, direction_metadata = fit_directions(
        probe_rows, probe_activation_path, layers, args.probe_c
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    causal_lm = model.get_base_model().model
    transformer_layers = causal_lm.layers if hasattr(causal_lm, "layers") else causal_lm.model.layers

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(
                rendered,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
            )["input_ids"]
            prompt_length = len(prompt_ids)
            for layer in layers:
                module = transformer_layers[layer - 1]
                for direction_name in direction_names:
                    delta = torch.tensor(direction_vectors[layer][direction_name], dtype=torch.bfloat16, device="cuda")
                    for scope in scopes:
                        start = {
                            "final": prompt_length - 1,
                            "last_32": max(0, prompt_length - 32),
                            "all_prompt": 0,
                        }[scope]
                        for alpha in alphas:
                            scores = {}
                            for label, choice_text in CHOICES.items():
                                candidate_ids = tokenizer(choice_text, add_special_tokens=False)["input_ids"]
                                all_ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device="cuda")

                                def hook(_module: Any, _inputs: Any, hook_output: Any) -> Any:
                                    hidden = hook_output[0] if isinstance(hook_output, tuple) else hook_output
                                    changed = hidden.clone()
                                    changed[:, start:prompt_length, :] += alpha * delta
                                    return (changed, *hook_output[1:]) if isinstance(hook_output, tuple) else changed

                                handle = module.register_forward_hook(hook)
                                try:
                                    logits = model(input_ids=all_ids, use_cache=False, return_dict=True).logits
                                finally:
                                    handle.remove()
                                positions = logits[0, prompt_length - 1 : prompt_length + len(candidate_ids) - 1]
                                target = torch.tensor(candidate_ids, dtype=torch.long, device="cuda")
                                token_logp = torch.log_softmax(positions.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
                                scores[label] = float(token_logp.mean())
                            log_odds = scores["REVISE"] - scores["KEEP"]
                            records.append({
                                "source_id": row["source_id"],
                                "dataset": row["dataset"],
                                "domain": row["domain"],
                                "expected_label": row["label"],
                                "layer": layer,
                                "direction": direction_name,
                                "token_scope": scope,
                                "alpha": alpha,
                                "keep_mean_logprob": scores["KEEP"],
                                "revise_mean_logprob": scores["REVISE"],
                                "log_odds_revise": log_odds,
                                "preference": "REVISE" if log_odds >= 0 else "KEEP",
                            })
            write_jsonl(output / "intervention_rows.jsonl", records)
            print(f"steering {row_index + 1}/{len(rows)}", flush=True)

    aggregates = []
    for layer in layers:
        for direction_name in direction_names:
            for scope in scopes:
                curve = []
                for alpha in alphas:
                    subset = [
                        row for row in records
                        if row["layer"] == layer and row["direction"] == direction_name
                        and row["token_scope"] == scope and row["alpha"] == alpha
                    ]
                    overall = metric_block(subset)
                    overall["subgroups"] = {
                        domain: metric_block([row for row in subset if row["domain"] == domain])
                        for domain in ("math", "code")
                    }
                    curve.append({"alpha": alpha, **overall})
                log_odds = [point["mean_log_odds_revise"] for point in curve]
                slope = float(np.polyfit(np.asarray(alphas), np.asarray(log_odds), 1)[0]) if len(alphas) > 1 else 0.0
                monotonic = all(right >= left for left, right in zip(log_odds, log_odds[1:]))
                baseline = next(point for point in curve if point["alpha"] == 0.0)
                eligible = [
                    point for point in curve
                    if point["keep_recall"] >= 0.70
                    and point["balanced_accuracy"] > baseline["balanced_accuracy"]
                    and point["revise_recall"] > baseline["revise_recall"]
                ]
                aggregates.append({
                    "layer": layer,
                    "direction": direction_name,
                    "token_scope": scope,
                    "curve": curve,
                    "log_odds_slope": slope,
                    "monotonic_in_alpha": monotonic,
                    "causal_readout_gate": bool(monotonic and slope > 0.001),
                    "behavioral_success_alphas": [point["alpha"] for point in eligible],
                })

    report = {
        "schema_version": "phase3_activation_steering_sweep_v2",
        "weights_changed": False,
        "status": "diagnostic_only",
        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path), "rows_selected": len(rows)},
        "probe": {
            "dataset": str(probe_dataset_path),
            "dataset_sha256": sha256(probe_dataset_path),
            "activations": str(probe_activation_path),
            "activations_sha256": sha256(probe_activation_path),
            "fit_splits": ["train", "dev"],
            "c": args.probe_c,
        },
        "model": {"base": args.base_model, "adapter": str(Path(args.adapter).resolve())},
        "intervention": "Residual addition at selected prompt token positions; one alpha unit is 5% of typical hidden-state L2.",
        "layers": layers,
        "directions": direction_names,
        "token_scopes": scopes,
        "alphas": alphas,
        "direction_metadata": direction_metadata,
        "aggregates": aggregates,
        "runtime_seconds": time.time() - started,
        "limitations": [
            "Small source-disjoint diagnostic, not confirmatory evaluation.",
            "Forced-choice log-probability evaluation does not measure free-form repair quality.",
            "A higher REVISE rate without balanced-accuracy and KEEP-floor gains is bias shift, not semantic improvement.",
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
