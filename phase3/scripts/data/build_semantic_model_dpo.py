"""Build model-generated, same-problem semantic DPO pairs for Phase 3.

Every selected problem contributes exactly two behavioral rows.  The correct
and plausible-wrong answers are both natural historical generations from Base
or Self-Correction V1; reference solutions are used only by the verifier and
are never placed in a training prompt.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine, build_prompt, problem_for_prompt
from phase3.scripts.data.build_contrastive_pairs import (
    INTERFACE_RE,
    TRIVIAL_RE,
    code_symbols,
    expected_mbpp_symbols,
    formatting,
    parse_number,
    similarity,
)


ROOT = Path(__file__).resolve().parents[2]
SEED = 271828
QUOTAS = {
    "apps": {"CW": 10, "WC": 10},
    "mbpp": {"CW": 3, "WC": 4},
    "gsm8k": {"CW": 35, "WC": 38},
}
TEMPLATES = (
    ("neutral_review_01", "Review your previous answer carefully and decide whether it should be kept or revised."),
    ("neutral_review_02", "Re-examine your previous answer and decide whether to keep it or revise it."),
    ("neutral_review_03", "Check your previous response carefully, then decide if it should be kept or revised."),
    ("neutral_review_04", "Evaluate your previous response on its own merits and choose whether to keep or revise it."),
    ("neutral_review_05", "Inspect the answer you gave above and determine whether it should be kept or revised."),
)
KEEP = "<decision>KEEP</decision>"
REVISE = "<decision>REVISE</decision>"


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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(*values: str) -> str:
    return hashlib.sha256((str(SEED) + "|" + "|".join(values)).encode()).hexdigest()


def model_members(meta: dict[str, Any], resolver: ProvenanceResolver) -> tuple[dict[str, Any], dict[str, Any], str, str, str, str]:
    source_id = str(meta["id"])
    base = resolver.resolve(meta["base_attempt_ref"], source_id)
    v1 = resolver.resolve(meta["v1_attempt_ref"], source_id)
    if meta["bucket"] == "WC":
        return v1, base, "self_correction_v1", "base", str(meta["v1_attempt_ref"]), str(meta["base_attempt_ref"])
    if meta["bucket"] == "CW":
        return base, v1, "base", "self_correction_v1", str(meta["base_attempt_ref"]), str(meta["v1_attempt_ref"])
    raise ValueError(f"Expected WC/CW, got {meta['bucket']}")


def answer_text(attempt: dict[str, Any], domain: str) -> tuple[str, str]:
    if domain == "code":
        code = str(attempt.get("extracted_answer") or "").strip()
        return f"```python\n{code}\n```", "model_extracted_code_uniform_python_fence"
    return str(attempt["initial_output"]).strip(), "unaltered_model_generation"


def numeric_distance(source: dict[str, Any], wrong_attempt: dict[str, Any], wrong: str) -> tuple[bool, float | None, float | None]:
    predicted = parse_number(wrong_attempt.get("extracted_answer") or wrong)
    expected = parse_number(source.get("reference_answer"))
    if predicted is None or expected is None:
        return False, None, None
    absolute = abs(float(predicted - expected))
    relative = absolute / max(abs(float(expected)), 1.0)
    return absolute <= 2 or relative <= 0.15, absolute, relative


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def nested_counts(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    if len(fields) == 1:
        return dict(sorted(Counter(str(row[fields[0]]) for row in rows).items()))
    return {
        value: nested_counts([row for row in rows if str(row[fields[0]]) == value], *fields[1:])
        for value in sorted({str(row[fields[0]]) for row in rows})
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(ROOT / "data" / "behavior" / "unified_source_inventory.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"))
    parser.add_argument("--previous-dpo", default=str(ROOT / "data" / "dpo_pilot" / "dpo_preferences.jsonl"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "outputs" / "phase3_decision_only_v1" / "final_adapter"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "semantic_model_dpo"))
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    inventory_path = Path(args.inventory).resolve()
    frozen_path = Path(args.frozen).resolve()
    previous_path = Path(args.previous_dpo).resolve()
    output_dir = Path(args.output_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(Path(args.tokenizer).resolve(), use_fast=True)
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    inventory = read_jsonl(inventory_path)
    frozen_ids = {str(row["source_id"]) for row in read_jsonl(frozen_path)}
    previous_ids = {str(row["source_id"]) for row in read_jsonl(previous_path)} if previous_path.exists() else set()
    exclusions: list[dict[str, Any]] = []
    eligible: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for meta in inventory:
        dataset = str(meta["dataset"])
        bucket = str(meta["bucket"])
        source_id = str(meta["id"])
        key = (dataset, bucket)
        if key not in {(d, b) for d, targets in QUOTAS.items() for b in targets}:
            continue
        if source_id in frozen_ids:
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "frozen_eval_overlap"})
            continue
        source = resolver.resolve(meta["source_ref"], source_id)
        correct_attempt, wrong_attempt, correct_origin, wrong_origin, correct_ref, wrong_ref = model_members(meta, resolver)
        correct, correct_transform = answer_text(correct_attempt, str(meta["domain"]))
        wrong, wrong_transform = answer_text(wrong_attempt, str(meta["domain"]))
        if not correct.strip() or not wrong.strip():
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "empty_model_answer"})
            continue
        correct_result = verifier.verify(source, correct)
        wrong_result = verifier.verify(source, wrong)
        if not correct_result["passed"]:
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "model_correct_failed_fresh_verifier", "detail": correct_result["detail"]})
            continue
        if wrong_result["passed"]:
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "model_wrong_passed_fresh_verifier", "detail": wrong_result["detail"]})
            continue
        wrong_detail = str(wrong_result.get("detail", ""))
        if TRIVIAL_RE.search(wrong_detail):
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "trivial_invalid_wrong", "detail": wrong_detail})
            continue

        partial_pass = False
        near_miss = False
        absolute_error = None
        relative_error = None
        interface_preserved: bool | None = None
        if dataset == "apps":
            if "wrong_answer" not in wrong_detail:
                exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "wrong_not_semantic_execution_failure", "detail": wrong_detail})
                continue
            passed_tests = int(wrong_result.get("passed_tests", 0) or 0)
            total_tests = int(wrong_result.get("total_tests", 0) or 0)
            partial_pass = 0 < passed_tests < total_tests
            interface_preserved = True
            wrong_type = "partial_test_pass" if partial_pass else "semantic_wrong_answer"
        elif dataset == "mbpp":
            if "AssertionError" not in wrong_detail or INTERFACE_RE.search(wrong_detail):
                exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "wrong_not_semantic_assertion_failure", "detail": wrong_detail})
                continue
            expected_symbols = expected_mbpp_symbols(source, correct)
            interface_preserved = bool(expected_symbols) and expected_symbols.issubset(code_symbols(wrong))
            if not interface_preserved:
                exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "code_interface_mismatch"})
                continue
            wrong_type = "semantic_test_failure"
        else:
            if len(wrong.split()) < 20:
                exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "wrong_reasoning_not_substantive"})
                continue
            near_miss, absolute_error, relative_error = numeric_distance(source, wrong_attempt, wrong)
            if absolute_error is None:
                exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "wrong_numeric_answer_not_parseable"})
                continue
            wrong_type = "near_miss_final" if near_miss else "plausible_numeric_reasoning_error"

        correct_format = formatting(correct)
        wrong_format = formatting(wrong)
        char_ratio = min(correct_format["chars"], wrong_format["chars"]) / max(correct_format["chars"], wrong_format["chars"], 1)
        if char_ratio < 0.5:
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "answer_length_ratio_below_0.5", "ratio": char_ratio})
            continue
        if meta["domain"] == "code" and correct_format["code_fence"] != wrong_format["code_fence"]:
            raise RuntimeError(f"Internal format normalization failure: {source_id}")

        task_prompt = build_prompt(problem_for_prompt(source))
        template_index = int(stable_rank("template", source_id), 16) % len(TEMPLATES)
        template_id, template = TEMPLATES[template_index]
        prefixes = {
            "KEEP": [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": correct},
                {"role": "user", "content": template},
            ],
            "REVISE": [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": wrong},
                {"role": "user", "content": template},
            ],
        }
        prompt_tokens = {}
        answer_tokens = {
            "correct": len(tokenizer(correct, add_special_tokens=False)["input_ids"]),
            "wrong": len(tokenizer(wrong, add_special_tokens=False)["input_ids"]),
        }
        for decision, messages in prefixes.items():
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokens[decision] = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if max(prompt_tokens.values()) > args.max_length:
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "context_over_max_length", "prompt_tokens": prompt_tokens})
            continue
        token_ratio = min(answer_tokens.values()) / max(answer_tokens.values())
        if token_ratio < 0.49:
            exclusions.append({"source_id": source_id, "dataset": dataset, "reason": "answer_token_ratio_below_0.49", "ratio": token_ratio})
            continue
        score = (
            100 * int(partial_pass)
            + 60 * int(near_miss)
            + 30 * similarity(correct, wrong)
            + 20 * token_ratio
            + 10 * char_ratio
        )
        eligible[key].append({
            "source_id": source_id,
            "dataset": dataset,
            "domain": str(meta["domain"]),
            "bucket": bucket,
            "problem_ref": str(meta["source_ref"]),
            "correct_answer_ref": correct_ref,
            "wrong_answer_ref": wrong_ref,
            "correct_origin": correct_origin,
            "wrong_origin": wrong_origin,
            "same_model_origin": correct_origin == wrong_origin,
            "correct_transform": correct_transform,
            "wrong_transform": wrong_transform,
            "correct": correct,
            "wrong": wrong,
            "correct_verifier_detail": correct_result["detail"],
            "wrong_verifier_detail": wrong_result["detail"],
            "wrong_type": wrong_type,
            "partial_pass": partial_pass,
            "near_miss": near_miss,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "interface_preserved": interface_preserved,
            "similarity_score": similarity(correct, wrong),
            "char_length_ratio": char_ratio,
            "token_length_ratio": token_ratio,
            "answer_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "correct_format": correct_format,
            "wrong_format": wrong_format,
            "template_id": template_id,
            "template": template,
            "prefixes": prefixes,
            "selection_score": score,
        })

    selected: list[dict[str, Any]] = []
    availability = {}
    for dataset, bucket_targets in QUOTAS.items():
        availability[dataset] = {}
        for bucket, requested in bucket_targets.items():
            candidates = eligible[(dataset, bucket)]
            candidates.sort(key=lambda row: (-row["selection_score"], stable_rank("select", row["source_id"])))
            availability[dataset][bucket] = len(candidates)
            if len(candidates) < requested:
                raise RuntimeError(f"Only {len(candidates)}/{requested} eligible model-model pairs for {dataset}/{bucket}")
            selected.extend(candidates[:requested])

    selected.sort(key=lambda row: (row["dataset"], row["source_id"]))
    pair_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    for item in selected:
        pair_id = f"semantic_model_dpo_v2::{item['dataset']}::{item['source_id']}"
        pair_rows.append({
            "pair_id": pair_id,
            "source_id": item["source_id"],
            "dataset": item["dataset"],
            "domain": item["domain"],
            "bucket": item["bucket"],
            "problem_ref": item["problem_ref"],
            "correct_answer_ref": item["correct_answer_ref"],
            "wrong_answer_ref": item["wrong_answer_ref"],
            "source_origin": {"correct": item["correct_origin"], "wrong": item["wrong_origin"]},
            "same_model_origin": item["same_model_origin"],
            "answer_transformation": {"correct": item["correct_transform"], "wrong": item["wrong_transform"]},
            "correct_verified": True,
            "wrong_verified": False,
            "correct_verifier_detail": item["correct_verifier_detail"],
            "wrong_verifier_detail": item["wrong_verifier_detail"],
            "wrong_type": item["wrong_type"],
            "hard_plausible_wrong": True,
            "partial_test_pass": item["partial_pass"],
            "near_miss": item["near_miss"],
            "absolute_error": item["absolute_error"],
            "relative_error": item["relative_error"],
            "neutral_template_id": item["template_id"],
            "neutral_review_text": item["template"],
            "answer_matching": {
                "similarity_score": item["similarity_score"],
                "char_length_ratio": item["char_length_ratio"],
                "token_length_ratio": item["token_length_ratio"],
                "answer_tokens": item["answer_tokens"],
                "prompt_tokens": item["prompt_tokens"],
                "correct_format": item["correct_format"],
                "wrong_format": item["wrong_format"],
                "code_interface_preserved": item["interface_preserved"],
            },
        })
        for decision, answer_ref in (("KEEP", item["correct_answer_ref"]), ("REVISE", item["wrong_answer_ref"])):
            expected = KEEP if decision == "KEEP" else REVISE
            construction_id = f"{pair_id}::{decision}"
            messages = item["prefixes"][decision] + [{"role": "assistant", "content": expected}]
            behavior_rows.append({
                "construction_id": construction_id,
                "pair_id": pair_id,
                "source_id": item["source_id"],
                "dataset": item["dataset"],
                "domain": item["domain"],
                "decision": decision,
                "neutral_template_id": item["template_id"],
                "feedback_type": "neutral_review",
                "messages": messages,
                "source_ref": item["problem_ref"],
                "answer_ref": answer_ref,
                "answer_model_origin": item["correct_origin"] if decision == "KEEP" else item["wrong_origin"],
            })
            preference_rows.append({
                "preference_id": f"dpo_{construction_id}",
                "pair_id": pair_id,
                "source_id": item["source_id"],
                "dataset": item["dataset"],
                "domain": item["domain"],
                "answer_state": "verified_correct" if decision == "KEEP" else "verified_wrong",
                "neutral_template_id": item["template_id"],
                "prompt_messages": messages[:-1],
                "chosen": expected,
                "rejected": REVISE if decision == "KEEP" else KEEP,
                "source_ref": item["problem_ref"],
                "answer_ref": answer_ref,
                "answer_model_origin": item["correct_origin"] if decision == "KEEP" else item["wrong_origin"],
            })

    errors: list[str] = []
    if len(pair_rows) != 100 or len(behavior_rows) != 200 or len(preference_rows) != 200:
        errors.append("Expected exactly 100 pairs and 200 behavioral/preferences rows")
    if len({row["source_id"] for row in pair_rows}) != 100:
        errors.append("Duplicate source IDs")
    if any(row["source_id"] in frozen_ids for row in pair_rows):
        errors.append("Frozen benchmark overlap")
    if any("reference" in row["source_origin"].values() for row in pair_rows):
        errors.append("Reference answer used as a training answer")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior_rows:
        grouped[row["pair_id"]].append(row)
    for pair_id, rows in grouped.items():
        if len(rows) != 2 or {row["decision"] for row in rows} != {"KEEP", "REVISE"}:
            errors.append(f"Broken pair polarity: {pair_id}")
            continue
        if rows[0]["messages"][0] != rows[1]["messages"][0] or rows[0]["messages"][2] != rows[1]["messages"][2]:
            errors.append(f"Problem/template mismatch: {pair_id}")
    template_by_label = nested_counts(behavior_rows, "decision", "neutral_template_id")
    if template_by_label["KEEP"] != template_by_label["REVISE"]:
        errors.append("Template-label imbalance")
    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    pairs_path = output_dir / "semantic_pairs.jsonl"
    behavior_path = output_dir / "semantic_behavior_rows.jsonl"
    preferences_path = output_dir / "dpo_preferences.jsonl"
    audit_path = output_dir / "hard_semantic_audit.jsonl"
    summary_path = output_dir / "semantic_dpo_summary.json"
    write_jsonl(pairs_path, pair_rows)
    write_jsonl(behavior_path, behavior_rows)
    write_jsonl(preferences_path, preference_rows)
    write_jsonl(audit_path, pair_rows)

    char_ratios = [float(row["answer_matching"]["char_length_ratio"]) for row in pair_rows]
    token_ratios = [float(row["answer_matching"]["token_length_ratio"]) for row in pair_rows]
    similarities = [float(row["answer_matching"]["similarity_score"]) for row in pair_rows]
    selected_ids = {row["source_id"] for row in pair_rows}
    summary = {
        "schema_version": "phase3_semantic_model_dpo_v2",
        "seed": SEED,
        "pairs": len(pair_rows),
        "behavioral_rows": len(behavior_rows),
        "preference_rows": len(preference_rows),
        "label_distribution": dict(sorted(Counter(row["decision"] for row in behavior_rows).items())),
        "dataset_distribution_pairs": nested_counts(pair_rows, "dataset"),
        "domain_distribution_pairs": nested_counts(pair_rows, "domain"),
        "bucket_distribution_pairs": nested_counts(pair_rows, "bucket"),
        "origin_direction_pairs": dict(sorted(Counter(f"{row['source_origin']['correct']}_correct__{row['source_origin']['wrong']}_wrong" for row in pair_rows).items())),
        "same_model_origin_pairs": sum(row["same_model_origin"] for row in pair_rows),
        "same_model_origin_limitation": "The verified source pool stores one generation per model/problem. Same-origin correct/wrong pairs cannot be formed without collecting new generations; selected pairs use Base and V1 outputs only, never references.",
        "hard_plausible_wrong_count": sum(row["hard_plausible_wrong"] for row in pair_rows),
        "wrong_type_distribution": nested_counts(pair_rows, "wrong_type"),
        "partial_code_pass_count": sum(row["partial_test_pass"] for row in pair_rows),
        "near_miss_math_count": sum(row["near_miss"] for row in pair_rows),
        "matching": {
            "char_length_ratio_min_over_max": describe(char_ratios),
            "token_length_ratio_min_over_max": describe(token_ratios),
            "similarity_score": describe(similarities),
            "code_interface_preserved": sum(row["answer_matching"]["code_interface_preserved"] is True for row in pair_rows if row["domain"] == "code"),
            "code_pairs": sum(row["domain"] == "code" for row in pair_rows),
            "uniform_code_format": all(row["answer_matching"]["correct_format"]["code_fence"] == row["answer_matching"]["wrong_format"]["code_fence"] for row in pair_rows if row["domain"] == "code"),
        },
        "neutral_template_distribution_by_label": template_by_label,
        "availability_after_validation": availability,
        "excluded_reason_distribution": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "overlap": {
            "frozen_eval": sorted(selected_ids & frozen_ids),
            "previous_dpo_source_count": len(selected_ids & previous_ids),
            "previous_dpo_source_ids": sorted(selected_ids & previous_ids),
        },
        "outputs": {},
        "validation": {
            "all_passed": True,
            "same_problem_within_pair": True,
            "fresh_correct_verifier_pass": 100,
            "fresh_wrong_verifier_fail": 100,
            "correct_and_wrong_are_model_generated": True,
            "reference_answers_in_training_prompts": False,
            "synthetic_wrong_answers": False,
            "balanced_keep_revise": True,
            "same_template_distribution_by_label": True,
            "char_length_ratio_at_least_0.5": True,
            "token_length_ratio_at_least_0.49": True,
            "code_interface_and_format_matched": True,
            "frozen_eval_overlap": False,
            "training_started": False,
        },
    }
    for name, path in (("pairs", pairs_path), ("behavior", behavior_path), ("preferences", preferences_path), ("audit", audit_path)):
        summary["outputs"][name] = {"path": str(path), "sha256": file_hash(path)}
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
