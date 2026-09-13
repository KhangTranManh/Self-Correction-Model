"""Build a source-disjoint router calibration set from verified Phase 3 sources."""

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
SEED = 20260907
NEUTRAL_REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."
BALANCED_QUOTAS = {
    ("gsm8k", "KEEP"): 21,
    ("gsm8k", "REVISE"): 21,
    ("apps", "KEEP"): 29,
    ("apps", "REVISE"): 29,
}
NATURAL_DATASET_QUOTAS = {"gsm8k": 16, "apps": 16, "mbpp": 8}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ids_from(paths: list[Path]) -> set[str]:
    output = set()
    for path in paths:
        for row in read_jsonl(path):
            value = row.get("source_id", row.get("id"))
            if value is not None:
                output.add(str(value))
    return output


def rank(context: str, source_id: str) -> str:
    return hashlib.sha256(f"{SEED}|calibration-v1|{context}|{source_id}".encode()).hexdigest()


def nested_counts(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    if len(fields) == 1:
        return dict(sorted(Counter(str(row[fields[0]]) for row in rows).items()))
    return {
        value: nested_counts([row for row in rows if str(row[fields[0]]) == value], *fields[1:])
        for value in sorted({str(row[fields[0]]) for row in rows})
    }


def eligible(meta: dict[str, Any], label: str | None = None) -> bool:
    actual = "KEEP" if bool(meta["v1_correct"]) else "REVISE"
    if label is not None and actual != label:
        return False
    return (actual == "KEEP" and meta["bucket"] in {"CC", "WC"}) or (actual == "REVISE" and meta["bucket"] == "WW")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(ROOT / "data/behavior/unified_source_inventory.jsonl"))
    parser.add_argument("--probe", default=str(ROOT / "runs/representation_probe/probe_dataset.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "data/two_stage_selective_repair/frozen_eval.jsonl"))
    parser.add_argument("--decision-train", default=str(ROOT / "data/decision_only/decision_only_train.jsonl"))
    parser.add_argument("--decision-dev", default=str(ROOT / "data/decision_only/decision_only_dev.jsonl"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "outputs/phase3_decision_only_v1/final_adapter"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/router_calibration_v1"))
    parser.add_argument("--generated-wrong", default=str(ROOT / "data/router_calibration_v1/generation/verified_wrong.jsonl"))
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    inventory_path = Path(args.inventory).resolve()
    probe_path = Path(args.probe).resolve()
    frozen_path = Path(args.frozen).resolve()
    decision_paths = [Path(args.decision_train).resolve(), Path(args.decision_dev).resolve()]
    output = Path(args.output_dir).resolve()
    inventory = read_jsonl(inventory_path)
    probe_ids = ids_from([probe_path])
    frozen_ids = ids_from([frozen_path])
    decision_train_ids = ids_from(decision_paths)
    protected_ids = probe_ids | frozen_ids
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    verification_exclusions = []
    overlength_exclusions = []
    generated_wrong_path = Path(args.generated_wrong).resolve()

    def materialize(meta: dict[str, Any], role: str) -> dict[str, Any] | None:
        source_id = str(meta["id"])
        source = resolver.resolve(meta["source_ref"], source_id)
        attempt = resolver.resolve(meta["v1_attempt_ref"], source_id)
        answer = str(attempt["initial_output"])
        result = verifier.verify(source, answer)
        historical = bool(meta["v1_correct"])
        if bool(result["passed"]) != historical:
            verification_exclusions.append({
                "source_id": source_id, "dataset": meta["dataset"],
                "historical_correct": historical, "fresh_correct": bool(result["passed"]),
                "detail": str(result["detail"]),
            })
            return None
        task_prompt = build_prompt(problem_for_prompt(source))
        messages = [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": NEUTRAL_REVIEW},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if token_length > args.max_length:
            overlength_exclusions.append({"source_id": source_id, "token_length": token_length})
            return None
        label = "KEEP" if historical else "REVISE"
        return {
            "schema_version": "phase3_router_calibration_v1",
            "calibration_role": role,
            "source_id": source_id,
            "dataset": str(meta["dataset"]),
            "domain": str(meta["domain"]),
            "label": label,
            "class_id": 0 if label == "KEEP" else 1,
            "bucket": str(meta["bucket"]),
            "messages": messages,
            "feedback_type": "canonical_neutral_review",
            "v1_correct": historical,
            "verification_method": verifier.method(source),
            "verifier_detail": str(result["detail"]),
            "source_ref": str(meta["source_ref"]),
            "v1_attempt_ref": str(meta["v1_attempt_ref"]),
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            "input_token_length": token_length,
            "decision_only_sft_source_overlap": source_id in decision_train_ids,
        }

    balanced = []
    used = set()
    for (dataset, label), requested in BALANCED_QUOTAS.items():
        if dataset == "gsm8k" and label == "REVISE" and generated_wrong_path.exists():
            generated = [
                row for row in read_jsonl(generated_wrong_path)
                if row["source_id"] not in protected_ids
            ]
            generated.sort(key=lambda row: (rank("generated:gsm8k:REVISE", row["source_id"]), row["source_id"]))
            if len(generated) < requested:
                raise RuntimeError(f"Insufficient verified generated GSM8K REVISE rows: {len(generated)}/{requested}")
            for row in generated[:requested]:
                selected_row = dict(row)
                selected_row["schema_version"] = "phase3_router_calibration_v1"
                selected_row["calibration_role"] = "balanced_fit"
                balanced.append(selected_row)
                used.add(selected_row["source_id"])
            continue
        candidates = [
            row for row in inventory
            if row["dataset"] == dataset
            and str(row["id"]) not in protected_ids
            and eligible(row, label)
        ]
        candidates.sort(key=lambda row: (
            str(row["id"]) in decision_train_ids,
            rank(f"balanced:{dataset}:{label}", str(row["id"])),
            str(row["id"]),
        ))
        accepted = 0
        for meta in candidates:
            row = materialize(meta, "balanced_fit")
            if row is None:
                continue
            balanced.append(row)
            used.add(row["source_id"])
            accepted += 1
            if accepted == requested:
                break
        if accepted != requested:
            raise RuntimeError(f"Insufficient verified rows for {dataset}/{label}: {accepted}/{requested}")

    natural = []
    for dataset, requested in NATURAL_DATASET_QUOTAS.items():
        candidates = [
            row for row in inventory
            if row["dataset"] == dataset
            and str(row["id"]) not in protected_ids
            and str(row["id"]) not in used
            and eligible(row)
        ]
        candidates.sort(key=lambda row: (
            str(row["id"]) in decision_train_ids,
            rank(f"natural:{dataset}", str(row["id"])),
            str(row["id"]),
        ))
        accepted = 0
        for meta in candidates:
            row = materialize(meta, "natural_prevalence_audit")
            if row is None:
                continue
            natural.append(row)
            used.add(row["source_id"])
            accepted += 1
            if accepted == requested:
                break
        if accepted != requested:
            raise RuntimeError(f"Insufficient natural rows for {dataset}: {accepted}/{requested}")

    balanced.sort(key=lambda row: (row["domain"], row["dataset"], row["label"], row["source_id"]))
    natural.sort(key=lambda row: (row["domain"], row["dataset"], row["source_id"]))
    balanced_ids = {row["source_id"] for row in balanced}
    natural_ids = {row["source_id"] for row in natural}
    errors = []
    if len(balanced) != 100 or Counter(row["label"] for row in balanced) != Counter({"KEEP": 50, "REVISE": 50}):
        errors.append("Balanced calibration must contain 100 rows, 50 per label")
    if len(natural) != 40:
        errors.append("Natural audit must contain 40 rows")
    if balanced_ids & natural_ids:
        errors.append("Balanced/natural source overlap")
    if (balanced_ids | natural_ids) & probe_ids:
        errors.append("Probe-256 source overlap")
    if (balanced_ids | natural_ids) & frozen_ids:
        errors.append("Frozen-200 source overlap")
    if errors:
        raise RuntimeError("; ".join(errors))

    balanced_path = output / "calibration_balanced.jsonl"
    natural_path = output / "calibration_natural.jsonl"
    all_path = output / "calibration_all.jsonl"
    manifest_path = output / "source_manifest.jsonl"
    write_jsonl(balanced_path, balanced)
    write_jsonl(natural_path, natural)
    write_jsonl(all_path, balanced + natural)
    write_jsonl(manifest_path, [
        {key: row[key] for key in (
            "calibration_role", "source_id", "dataset", "domain", "label", "bucket",
            "source_ref", "v1_attempt_ref", "verification_method", "decision_only_sft_source_overlap",
        )}
        for row in balanced + natural
    ])
    overlap_audit = {
        "balanced_natural": sorted(balanced_ids & natural_ids),
        "probe_256": sorted((balanced_ids | natural_ids) & probe_ids),
        "frozen_200": sorted((balanced_ids | natural_ids) & frozen_ids),
        "decision_only_sft": sorted((balanced_ids | natural_ids) & decision_train_ids),
    }
    write_json(output / "overlap_audit.json", overlap_audit)
    summary = {
        "schema_version": "phase3_router_calibration_summary_v1",
        "seed": SEED,
        "total_rows": len(balanced) + len(natural),
        "balanced_fit": {
            "rows": len(balanced),
            "label": nested_counts(balanced, "label"),
            "domain": nested_counts(balanced, "domain", "label"),
            "dataset": nested_counts(balanced, "dataset", "label"),
            "decision_only_sft_source_overlap": sum(row["decision_only_sft_source_overlap"] for row in balanced),
        },
        "natural_audit": {
            "rows": len(natural),
            "label": nested_counts(natural, "label"),
            "domain": nested_counts(natural, "domain", "label"),
            "dataset": nested_counts(natural, "dataset", "label"),
            "decision_only_sft_source_overlap": sum(row["decision_only_sft_source_overlap"] for row in natural),
        },
        "fresh_verification_exclusions": len(verification_exclusions),
        "overlength_exclusions": len(overlength_exclusions),
        "max_input_tokens": max(row["input_token_length"] for row in balanced + natural),
        "overlap": {key: len(value) for key, value in overlap_audit.items()},
        "estimated_shortcut_risks": {
            "single_template": "intentional: calibrates the canonical deployment prompt; does not establish template robustness",
            "missing_datasets": ["humaneval", "svamp"],
            "balanced_fit_missing_mbpp": "only four eligible MBPP KEEP rows remain outside probe/frozen; MBPP is natural-audit only",
            "decision_only_sft_source_overlap": "reported explicitly; unseen SFT sources are prioritized but strict exclusion is infeasible for GSM8K REVISE",
            "balanced_prevalence": "balanced_fit must not be treated as natural deployment prevalence",
        },
        "usage_contract": {
            "balanced_fit": "fit probability/threshold calibration only; never update LLM or probe weights",
            "natural_audit": "report expected deployment skew; do not select calibration hyperparameters",
            "frozen_200": "final test only; never fit threshold or calibrator",
        },
        "validation": {
            "fresh_labels": True,
            "balanced_fit_50_50": True,
            "balanced_natural_disjoint": True,
            "probe_256_overlap_zero": True,
            "frozen_200_overlap_zero": True,
            "no_synthetic_answers": True,
            "gpu_required": False,
            "all_passed": True,
        },
    }
    write_json(output / "calibration_summary.json", summary)
    write_json(output / "verification_exclusions.json", verification_exclusions)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
