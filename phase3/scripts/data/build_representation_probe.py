"""Build the frozen, balanced Phase 3 representation-probe dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine, build_prompt, problem_for_prompt
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SEED = 161803
NEUTRAL_REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."
QUOTAS = {
    ("gsm8k", "KEEP"): 64,
    ("gsm8k", "REVISE"): 64,
    ("mbpp", "KEEP"): 25,
    ("mbpp", "REVISE"): 25,
    ("apps", "KEEP"): 39,
    ("apps", "REVISE"): 39,
}
SPLITS = {
    ("gsm8k", "KEEP"): (38, 13, 13),
    ("gsm8k", "REVISE"): (38, 13, 13),
    ("mbpp", "KEEP"): (15, 5, 5),
    ("mbpp", "REVISE"): (15, 5, 5),
    ("apps", "KEEP"): (23, 8, 8),
    ("apps", "REVISE"): (23, 8, 8),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rank(context: str, source_id: str) -> str:
    return hashlib.sha256(f"{SEED}|probe|{context}|{source_id}".encode()).hexdigest()


def counts(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    if len(fields) == 1:
        return dict(sorted(Counter(str(row[fields[0]]) for row in rows).items()))
    output = {}
    for value in sorted({str(row[fields[0]]) for row in rows}):
        output[value] = counts([row for row in rows if str(row[fields[0]]) == value], *fields[1:])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(ROOT / "data" / "behavior" / "unified_source_inventory.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "representation_probe"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "outputs" / "phase3_decision_only_v1" / "final_adapter"))
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    inventory_path = Path(args.inventory).resolve()
    frozen_path = Path(args.frozen).resolve()
    output_dir = Path(args.output_dir).resolve()
    inventory = read_jsonl(inventory_path)
    frozen_ids = {str(row["source_id"]) for row in read_jsonl(frozen_path)}
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    selected = []
    excluded = []
    length_excluded = []

    for (dataset, label), requested in QUOTAS.items():
        expected_correct = label == "KEEP"
        candidates = [
            row for row in inventory
            if row["dataset"] == dataset
            and bool(row["v1_correct"]) == expected_correct
            and row["id"] not in frozen_ids
            and (
                (label == "KEEP" and row["bucket"] in {"CC", "WC"})
                or (label == "REVISE" and row["bucket"] == "WW")
            )
        ]
        candidates.sort(key=lambda row: (rank(f"select:{dataset}:{label}", row["id"]), row["id"]))
        accepted = []
        for meta in candidates:
            source_id = str(meta["id"])
            source = resolver.resolve(meta["source_ref"], source_id)
            attempt = resolver.resolve(meta["v1_attempt_ref"], source_id)
            result = verifier.verify(source, str(attempt["initial_output"]))
            if bool(result["passed"]) != expected_correct:
                excluded.append({
                    "source_id": source_id,
                    "dataset": dataset,
                    "label": label,
                    "historical_correct": bool(meta["v1_correct"]),
                    "fresh_correct": bool(result["passed"]),
                    "detail": result["detail"],
                })
                continue
            task_prompt = build_prompt(problem_for_prompt(source))
            messages = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": str(attempt["initial_output"])},
                {"role": "user", "content": NEUTRAL_REVIEW},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            token_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if token_length > args.max_length:
                length_excluded.append({
                    "source_id": source_id,
                    "dataset": dataset,
                    "label": label,
                    "token_length": token_length,
                    "max_length": args.max_length,
                })
                continue
            accepted.append((meta, source, attempt, result, task_prompt, token_length))
            if len(accepted) == requested:
                break
        if len(accepted) != requested:
            raise RuntimeError(f"Insufficient fresh rows for {dataset}/{label}: {len(accepted)}/{requested}")
        split_sizes = SPLITS[(dataset, label)]
        cursor = 0
        for split, size in zip(("train", "dev", "test"), split_sizes):
            for meta, source, attempt, result, task_prompt, token_length in accepted[cursor:cursor + size]:
                selected.append({
                    "source_id": str(meta["id"]),
                    "dataset": dataset,
                    "domain": str(meta["domain"]),
                    "label": label,
                    "class_id": 0 if label == "KEEP" else 1,
                    "split": split,
                    "bucket": str(meta["bucket"]),
                    "v1_correct": bool(meta["v1_correct"]),
                    "feedback_type": "neutral_review",
                    "verification_method": verifier.method(source),
                    "verifier_detail": str(result["detail"]),
                    "source_ref": str(meta["source_ref"]),
                    "v1_attempt_ref": str(meta["v1_attempt_ref"]),
                    "messages": [
                        {"role": "user", "content": task_prompt},
                        {"role": "assistant", "content": str(attempt["initial_output"])},
                        {"role": "user", "content": NEUTRAL_REVIEW},
                    ],
                    "answer_length_chars": len(str(attempt["initial_output"])),
                    "answer_length_words": len(str(attempt["initial_output"]).split()),
                    "input_token_length": token_length,
                })
            cursor += size

    selected.sort(key=lambda row: ({"train": 0, "dev": 1, "test": 2}[row["split"]], row["dataset"], row["label"], row["source_id"]))
    ids = [row["source_id"] for row in selected]
    errors = []
    if len(selected) != 256 or len(set(ids)) != 256:
        errors.append("Expected 256 unique sources")
    if any(row["label"] == "REVISE" and row["bucket"] != "WW" for row in selected):
        errors.append("Non-WW REVISE contamination")
    if any(row["source_id"] in frozen_ids for row in selected):
        errors.append("Frozen benchmark overlap")
    if any(row["dataset"] == "humaneval" for row in selected):
        errors.append("HumanEval contamination")
    split_ids = {split: {row["source_id"] for row in selected if row["split"] == split} for split in ("train", "dev", "test")}
    overlaps = {
        "train_dev": sorted(split_ids["train"] & split_ids["dev"]),
        "train_test": sorted(split_ids["train"] & split_ids["test"]),
        "dev_test": sorted(split_ids["dev"] & split_ids["test"]),
    }
    if any(overlaps.values()):
        errors.append("Split source overlap")
    if errors:
        raise RuntimeError("; ".join(errors))

    dataset_path = output_dir / "probe_dataset.jsonl"
    summary_path = output_dir / "probe_split_summary.json"
    write_jsonl(dataset_path, selected)
    summary = {
        "schema_version": "phase3_representation_probe_v1",
        "seed": SEED,
        "size_note": "256 is the maximum clean class/domain-balanced size after fresh verification left only 64 eligible GSM8K WW rows; preferred 300-500 is infeasible without CW or frozen leakage.",
        "total_sources": len(selected),
        "counts_by_split": counts(selected, "split"),
        "counts_by_split_label": counts(selected, "split", "label"),
        "counts_by_split_domain_label": counts(selected, "split", "domain", "label"),
        "counts_by_split_dataset_label": counts(selected, "split", "dataset", "label"),
        "overall_domain_label": counts(selected, "domain", "label"),
        "overall_dataset_label": counts(selected, "dataset", "label"),
        "bucket_by_label": counts(selected, "label", "bucket"),
        "fresh_verification_exclusions": excluded,
        "fresh_verification_exclusion_count": len(excluded),
        "overlength_exclusions": length_excluded,
        "overlength_exclusion_count": len(length_excluded),
        "max_input_tokens": max(row["input_token_length"] for row in selected),
        "source_overlap": overlaps,
        "frozen_benchmark_overlap": [],
        "input": {"path": str(inventory_path), "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest()},
        "output": {"path": str(dataset_path), "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()},
        "validation": {
            "all_existing_phase3_sources": True,
            "all_labels_freshly_verified": True,
            "balanced_correct_wrong": True,
            "balanced_math_code": True,
            "balanced_labels_within_dataset": True,
            "neutral_review_only": True,
            "no_cw_as_wrong": True,
            "no_humaneval": True,
            "no_frozen_leakage": True,
            "zero_split_overlap": True,
            "no_generation_target_in_input": True,
            "all_passed": True,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
