"""Prepare the next Phase 3 datasets and audits without using a GPU.

This script deliberately does not generate model answers, run verifiers, train
adapters, or load a language model. It repackages already verified examples,
creates leakage-safe splits, and measures obvious label shortcuts on CPU.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
SEED = 20260907
DEV_QUOTAS = {
    ("apps", "CW"): 2,
    ("apps", "WC"): 2,
    ("mbpp", "CW"): 1,
    ("mbpp", "WC"): 1,
    ("gsm8k", "CW"): 7,
    ("gsm8k", "WC"): 7,
}
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "self_correction_v1": "Kxck/Self_Correction_v1",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def label_of(row: dict[str, Any]) -> str:
    return "KEEP" if row["answer_state"] == "verified_correct" else "REVISE"


def split_pairs(pairs: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["dataset"], pair["bucket"])].append(pair)
    dev: set[str] = set()
    for key, quota in DEV_QUOTAS.items():
        candidates = sorted(grouped[key], key=lambda row: stable_key(row["pair_id"]))
        if len(candidates) < quota:
            raise ValueError(f"Not enough pairs for dev stratum {key}: {len(candidates)} < {quota}")
        dev.update(row["pair_id"] for row in candidates[:quota])
    all_ids = {row["pair_id"] for row in pairs}
    return all_ids - dev, dev


def surface_features(text: str) -> list[float]:
    chars = max(1, len(text))
    words = re.findall(r"\b\w+\b", text)
    digits = sum(ch.isdigit() for ch in text)
    punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    confidence = len(re.findall(r"\b(clearly|obviously|therefore|thus|certainly|correct|final)\b", text.lower()))
    return [
        np.log1p(chars),
        np.log1p(len(words)),
        np.log1p(text.count("\n") + 1),
        float("```" in text),
        digits / chars,
        punctuation / chars,
        confidence / max(1, len(words)),
        sum(ch.isupper() for ch in text) / chars,
    ]


def summarize_cv(y: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, pred in predictions.items():
        result[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "keep_recall": float(recall_score(y, pred, pos_label=0)),
            "revise_recall": float(recall_score(y, pred, pos_label=1)),
        }
    return result


def shortcut_audit(preferences: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    texts = np.asarray([row["prompt_messages"][1]["content"] for row in preferences], dtype=object)
    y = np.asarray([0 if label_of(row) == "KEEP" else 1 for row in preferences])
    groups = np.asarray([row["pair_id"] for row in preferences])
    origins = np.asarray([0 if row["answer_model_origin"] == "base" else 1 for row in preferences]).reshape(-1, 1)
    surface = np.asarray([surface_features(text) for text in texts], dtype=float)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = {name: np.full(len(y), -1, dtype=int) for name in ("char_tfidf", "surface_only", "origin_only")}
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(cv.split(texts, y, groups)):
        tfidf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        )
        numeric = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        )
        origin = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
        tfidf.fit(texts[train_idx].tolist(), y[train_idx])
        numeric.fit(surface[train_idx], y[train_idx])
        origin.fit(origins[train_idx], y[train_idx])
        predictions["char_tfidf"][test_idx] = tfidf.predict(texts[test_idx].tolist())
        predictions["surface_only"][test_idx] = numeric.predict(surface[test_idx])
        predictions["origin_only"][test_idx] = origin.predict(origins[test_idx])
        fold_ids[test_idx] = fold
    if any(np.any(value < 0) for value in predictions.values()):
        raise AssertionError("Cross-validation did not produce all predictions")
    prediction_rows = []
    for index, row in enumerate(preferences):
        prediction_rows.append({
            "pair_id": row["pair_id"],
            "preference_id": row["preference_id"],
            "gold_label": label_of(row),
            "fold": int(fold_ids[index]),
            **{f"{name}_prediction": "REVISE" if pred[index] else "KEEP" for name, pred in predictions.items()},
        })
    metrics = summarize_cv(y, predictions)
    risks = {
        "answer_text_style": "high" if metrics["char_tfidf"]["balanced_accuracy"] >= 0.65 else "moderate" if metrics["char_tfidf"]["balanced_accuracy"] >= 0.58 else "low",
        "simple_surface_features": "high" if metrics["surface_only"]["balanced_accuracy"] >= 0.65 else "moderate" if metrics["surface_only"]["balanced_accuracy"] >= 0.58 else "low",
        "model_origin": "high" if metrics["origin_only"]["balanced_accuracy"] >= 0.65 else "moderate" if metrics["origin_only"]["balanced_accuracy"] >= 0.58 else "low",
    }
    return {
        "method": "5-fold StratifiedGroupKFold grouped by pair/problem; labels are hidden from features",
        "interpretation": "This diagnoses learnable answer-style confounds inside the training pool; it is not a frozen benchmark result.",
        "metrics": metrics,
        "estimated_risks": risks,
    }, prediction_rows


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def activation_info(path: Path, expected_source_ids: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "aligned": False, "path": relative(path)}
    with np.load(path, allow_pickle=False) as archive:
        source_ids = [str(value) for value in archive["source_ids"]]
        shape = list(archive["activations"].shape)
        layers = [int(value) for value in archive["layers"]]
    return {
        "available": True,
        "aligned": source_ids == expected_source_ids,
        "path": relative(path),
        "shape": shape,
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "offline_next")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source_dir = ROOT / "data" / "semantic_model_dpo"
    pairs = read_jsonl(source_dir / "semantic_pairs.jsonl")
    preferences = read_jsonl(source_dir / "dpo_preferences.jsonl")
    probe_rows = read_jsonl(ROOT / "runs" / "representation_probe" / "probe_dataset.jsonl")
    if len(pairs) != 100 or len(preferences) != 200:
        raise ValueError("Expected the validated 100-pair / 200-row semantic dataset")

    pair_by_id = {row["pair_id"]: row for row in pairs}
    preferences_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preferences:
        preferences_by_pair[row["pair_id"]].append(row)
    if any(sorted(label_of(row) for row in rows) != ["KEEP", "REVISE"] for rows in preferences_by_pair.values()):
        raise AssertionError("Each semantic pair must contain one KEEP and one REVISE row")

    train_ids, dev_ids = split_pairs(pairs)
    def decision_row(row: dict[str, Any], split: str) -> dict[str, Any]:
        label = label_of(row)
        return {
            **row,
            "construction_id": f"decision_token_v2::{row['source_id']}::{label}",
            "messages": row["prompt_messages"] + [{"role": "assistant", "content": row["chosen"]}],
            "label": label,
            "split": split,
            "objective": "decision_token_only",
        }

    train_rows = [decision_row(row, "train") for row in preferences if row["pair_id"] in train_ids]
    dev_rows = [decision_row(row, "dev") for row in preferences if row["pair_id"] in dev_ids]
    train_rows.sort(key=lambda row: stable_key(row["preference_id"]))
    dev_rows.sort(key=lambda row: stable_key(row["preference_id"]))

    baseline_activation = ROOT / "runs" / "representation_probe" / "activations" / "decision_only_v1_final_token.npz"
    semantic_activation = PROJECT_ROOT / "outputs" / "phase3_dpo_semantic_v2" / "representation_probe" / "activations" / "dpo_semantic_v2_final_token.npz"
    expected_probe_source_ids = [row["source_id"] for row in probe_rows]
    baseline_activation_info = activation_info(baseline_activation, expected_probe_source_ids)
    semantic_activation_info = activation_info(semantic_activation, expected_probe_source_ids)
    classifier_rows = []
    for index, row in enumerate(probe_rows):
        classifier_rows.append({
            **row,
            "row_index": index,
            "group_id": row["source_id"],
            "decision_only_v1_activation_ref": f"{relative(baseline_activation)}#activations[{index},:,:]",
            "semantic_v2_activation_ref": f"{relative(semantic_activation)}#activations[{index},:,:]" if semantic_activation.exists() else None,
        })

    hard_rows = []
    future_rows = []
    for pair in pairs:
        members = {label_of(row): row for row in preferences_by_pair[pair["pair_id"]]}
        correct = members["KEEP"]
        wrong = members["REVISE"]
        matching = pair["answer_matching"]
        hardness = (
            40 * int(pair["partial_test_pass"])
            + 30 * int(pair["near_miss"])
            + 20 * float(matching["similarity_score"])
            + 5 * float(matching["char_length_ratio"])
            + 5 * float(matching["token_length_ratio"])
        )
        hard_rows.append({
            "pair_id": pair["pair_id"],
            "source_id": pair["source_id"],
            "problem_ref": pair["problem_ref"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "bucket": pair["bucket"],
            "hardness_score": round(hardness, 6),
            "wrong_type": pair["wrong_type"],
            "near_miss": pair["near_miss"],
            "partial_test_pass": pair["partial_test_pass"],
            "similarity_score": matching["similarity_score"],
            "char_length_ratio": matching["char_length_ratio"],
            "token_length_ratio": matching["token_length_ratio"],
            "correct_model_origin": correct["answer_model_origin"],
            "wrong_model_origin": wrong["answer_model_origin"],
            "problem": correct["prompt_messages"][0]["content"],
            "correct_answer": correct["prompt_messages"][1]["content"],
            "wrong_answer": wrong["prompt_messages"][1]["content"],
            "wrong_verifier_detail": pair["wrong_verifier_detail"],
        })
        origin = correct["answer_model_origin"]
        future_rows.append({
            "task_id": f"same_origin_wrong::{pair['source_id']}",
            "priority": "pilot" if pair["partial_test_pass"] or pair["near_miss"] else "standard",
            "pair_id": pair["pair_id"],
            "source_id": pair["source_id"],
            "problem_ref": pair["problem_ref"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "problem_message": correct["prompt_messages"][0],
            "existing_verified_correct_answer": correct["prompt_messages"][1]["content"],
            "existing_correct_answer_ref": correct["answer_ref"],
            "model_origin_to_sample": origin,
            "model_id_to_sample": MODEL_IDS[origin],
            "desired_new_state": "verified_wrong_but_plausible",
            "samples_requested": 12 if pair["domain"] == "code" else 6,
            "acceptance_constraints": [
                "fresh verifier fails semantically",
                "same output interface and similar length as correct answer",
                "not a formatting-only, compile-only, or empty-answer failure",
                "no reference answer text in the router prompt",
            ],
        })
    hard_rows.sort(key=lambda row: (-row["hardness_score"], stable_key(row["pair_id"])))
    future_rows.sort(key=lambda row: (0 if row["priority"] == "pilot" else 1, stable_key(row["task_id"])))

    # Forty diverse rows for manual inspection: all partial-pass code, then low
    # matching pairs, then high-ranked hard cases while limiting any dataset.
    selected: dict[str, dict[str, Any]] = {}
    candidate_lists = [
        [row for row in hard_rows if row["partial_test_pass"]],
        sorted(hard_rows, key=lambda row: (row["similarity_score"], stable_key(row["pair_id"])))[:15],
        sorted(hard_rows, key=lambda row: (row["token_length_ratio"], stable_key(row["pair_id"])))[:15],
        [row for row in hard_rows if row["near_miss"]],
        hard_rows,
    ]
    for candidates in candidate_lists:
        for row in candidates:
            if len(selected) >= 40:
                break
            selected.setdefault(row["pair_id"], row)
    manual_rows = []
    for row in selected.values():
        manual_rows.append({
            **row,
            "human_audit": {
                "semantic_plausibility": None,
                "surface_confound": None,
                "length_or_format_leak": None,
                "verifier_label_confirmed": None,
                "notes": None,
            },
        })

    shortcut, shortcut_predictions = shortcut_audit(preferences)
    frozen_path = ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"
    frozen_ids = {row["source_id"] for row in read_jsonl(frozen_path)} if frozen_path.exists() else set()
    source_ids = {row["source_id"] for row in pairs}

    outputs = {
        "decision_token_train": output / "decision_token_train.jsonl",
        "decision_token_dev": output / "decision_token_dev.jsonl",
        "classifier_manifest": output / "classifier_manifest.jsonl",
        "hard_revise_existing": output / "hard_revise_existing.jsonl",
        "manual_shortcut_audit_queue": output / "manual_shortcut_audit_queue.jsonl",
        "future_same_origin_generation_manifest": output / "future_same_origin_generation_manifest.jsonl",
        "shortcut_cv_predictions": output / "shortcut_cv_predictions.jsonl",
        "shortcut_audit": output / "shortcut_audit.json",
    }
    write_jsonl(outputs["decision_token_train"], train_rows)
    write_jsonl(outputs["decision_token_dev"], dev_rows)
    write_jsonl(outputs["classifier_manifest"], classifier_rows)
    write_jsonl(outputs["hard_revise_existing"], hard_rows)
    write_jsonl(outputs["manual_shortcut_audit_queue"], manual_rows)
    write_jsonl(outputs["future_same_origin_generation_manifest"], future_rows)
    write_jsonl(outputs["shortcut_cv_predictions"], shortcut_predictions)
    write_json(outputs["shortcut_audit"], shortcut)

    summary = {
        "schema_version": "phase3_offline_next_v1",
        "seed": SEED,
        "gpu_used": False,
        "training_started": False,
        "source_verification": "Inherited from semantic_model_dpo: correct 100/100 fresh pass; wrong 100/100 fresh fail.",
        "decision_token_data": {
            "train_rows": len(train_rows),
            "train_pairs": len(train_ids),
            "dev_rows": len(dev_rows),
            "dev_pairs": len(dev_ids),
            "train_labels": distribution([{**r, "label": label_of(r)} for r in train_rows], "label"),
            "dev_labels": distribution([{**r, "label": label_of(r)} for r in dev_rows], "label"),
            "train_datasets": distribution([pair_by_id[pair_id] for pair_id in train_ids], "dataset"),
            "dev_datasets": distribution([pair_by_id[pair_id] for pair_id in dev_ids], "dataset"),
            "train_dev_source_overlap": sorted({r["source_id"] for r in train_rows} & {r["source_id"] for r in dev_rows}),
        },
        "classifier_data": {
            "rows": len(classifier_rows),
            "splits": distribution(classifier_rows, "split"),
            "labels": distribution(classifier_rows, "label"),
            "domains": distribution(classifier_rows, "domain"),
            "decision_only_v1_activations": baseline_activation_info,
            "semantic_v2_activations": semantic_activation_info,
            "recommended_models": ["class-weighted logistic regression", "small MLP with focal loss"],
        },
        "hard_revise_pool": {
            "rows": len(hard_rows),
            "near_miss_math": sum(bool(row["near_miss"]) for row in hard_rows),
            "partial_test_pass_code": sum(bool(row["partial_test_pass"]) for row in hard_rows),
            "datasets": distribution(hard_rows, "dataset"),
            "manual_audit_rows": len(manual_rows),
        },
        "future_same_origin_generation": {
            "tasks": len(future_rows),
            "requested_generations": sum(row["samples_requested"] for row in future_rows),
            "requires_gpu_later": True,
            "reason": "The current verified pool has only one generation per model/problem, so same-origin correct/wrong pairs require new sampling.",
        },
        "shortcut_audit": shortcut,
        "validation": {
            "all_passed": True,
            "pair_groups_not_split": not ({r["pair_id"] for r in train_rows} & {r["pair_id"] for r in dev_rows}),
            "train_keep_revise_balanced": Counter(label_of(r) for r in train_rows) == Counter({"KEEP": 80, "REVISE": 80}),
            "dev_keep_revise_balanced": Counter(label_of(r) for r in dev_rows) == Counter({"KEEP": 20, "REVISE": 20}),
            "frozen_eval_source_overlap": sorted(source_ids & frozen_ids),
            "duplicate_pair_ids": len(pair_by_id) != len(pairs),
            "decision_only_v1_activation_order_aligned": baseline_activation_info["aligned"],
            "semantic_v2_activation_order_aligned": semantic_activation_info["aligned"],
            "new_synthetic_wrong_answers": False,
            "new_model_generation": False,
        },
        "outputs": {},
    }
    summary_path = output / "offline_preparation_summary.json"
    write_json(summary_path, summary)
    summary["outputs"] = {name: {"path": relative(path), "sha256": sha256(path)} for name, path in outputs.items()}
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
