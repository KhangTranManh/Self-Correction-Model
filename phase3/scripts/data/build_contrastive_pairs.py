"""Build same-problem correct-vs-plausible-wrong Phase 3 contrastive pairs."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine, build_prompt, problem_for_prompt


ROOT = Path(__file__).resolve().parents[2]
SEED = 141421
QUOTAS = {"apps": 70, "mbpp": 20, "gsm8k": 30}
TEMPLATES = (
    ("neutral_review_01", "Review your previous answer carefully and decide whether it should be kept or revised."),
    ("neutral_review_02", "Re-examine your previous answer and decide whether to keep it or revise it."),
    ("neutral_review_03", "Check your previous response carefully, then decide if it should be kept or revised."),
    ("neutral_review_04", "Evaluate your previous response on its own merits and choose whether to keep or revise it."),
    ("neutral_review_05", "Inspect the answer you gave above and determine whether it should be kept or revised."),
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?")
DEF_RE = re.compile(r"(?:^|\n)\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")
CLASS_RE = re.compile(r"(?:^|\n)\s*class\s+([A-Za-z_]\w*)")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
TRIVIAL_RE = re.compile(r"syntaxerror|indentationerror|no.code|empty|timeout|modulenotfounderror|importerror", re.I)
INTERFACE_RE = re.compile(r"nameerror|not defined|attributeerror", re.I)


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


def rank(context: str, value: str) -> str:
    return hashlib.sha256(f"{SEED}|contrastive|{context}|{value}".encode()).hexdigest()


def parse_number(value: Any) -> Fraction | None:
    matches = NUMBER_RE.findall(str(value or ""))
    if not matches:
        return None
    try:
        return Fraction(matches[-1].replace(",", "").replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def code_symbols(text: str) -> set[str]:
    return set(DEF_RE.findall(text)) | set(CLASS_RE.findall(text))


def expected_mbpp_symbols(source: dict[str, Any], correct: str) -> set[str]:
    calls = set()
    ignored = {"assert", "len", "str", "int", "float", "list", "set", "tuple", "dict", "sorted", "sum", "min", "max", "range", "abs", "round"}
    for test in source.get("tests", []):
        calls.update(name for name in CALL_RE.findall(str(test)) if name not in ignored)
    return calls & code_symbols(correct)


def formatting(text: str) -> dict[str, Any]:
    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()) or 1,
        "code_fence": "```" in text,
    }


def similarity(correct: str, wrong: str) -> float:
    a, b = formatting(correct), formatting(wrong)
    length_ratio = min(a["chars"], b["chars"]) / max(a["chars"], b["chars"], 1)
    line_ratio = min(a["lines"], b["lines"]) / max(a["lines"], b["lines"], 1)
    fence = 1.0 if a["code_fence"] == b["code_fence"] else 0.0
    return 0.6 * length_ratio + 0.25 * line_ratio + 0.15 * fence


def correct_member(meta: dict[str, Any], source: dict[str, Any], resolver: ProvenanceResolver) -> tuple[str, str, str]:
    source_id = str(meta["id"])
    if meta["bucket"] == "WC":
        attempt = resolver.resolve(meta["v1_attempt_ref"], source_id)
        return str(attempt["initial_output"]), str(meta["v1_attempt_ref"]), "self_correction_v1_natural"
    if source["dataset"] == "gsm8k":
        return str(source["reference_solution"]), f"{meta['source_ref']}::reference_solution", "verified_reference"
    return str(source["ground_truth"]), f"{meta['source_ref']}::ground_truth", "verified_reference"


def wrong_options(meta: dict[str, Any], resolver: ProvenanceResolver) -> list[tuple[str, str, str, dict[str, Any]]]:
    source_id = str(meta["id"])
    names = ("base",) if meta["bucket"] == "WC" else ("base", "v1")
    output = []
    for name in names:
        reference = meta["base_attempt_ref"] if name == "base" else meta["v1_attempt_ref"]
        attempt = resolver.resolve(reference, source_id)
        output.append((str(attempt["initial_output"]), str(reference), name, attempt))
    return output


def option_metadata(dataset: str, source: dict[str, Any], correct: str, option: tuple[str, str, str, dict[str, Any]]) -> dict[str, Any] | None:
    wrong, reference, origin, attempt = option
    detail = str(attempt.get("verifier_detail", ""))
    sim = similarity(correct, wrong)
    if dataset == "gsm8k":
        predicted = parse_number(attempt.get("extracted_answer") or wrong)
        expected = parse_number(source.get("reference_answer"))
        if predicted is None or expected is None or len(wrong.split()) < 20:
            return None
        absolute = abs(float(predicted - expected))
        relative = absolute / max(abs(float(expected)), 1.0)
        near = absolute <= 2 or relative <= 0.15
        wrong_type = "near_miss_final" if near else "plausible_numeric_reasoning_error"
        tier = 3 if near else 2
        return {
            "answer": wrong,
            "ref": reference,
            "origin": origin,
            "attempt": attempt,
            "score": 100 * tier + 20 * sim + (5 if origin == "base" else 0),
            "wrong_type": wrong_type,
            "hardness_tier": tier,
            "hardness_reason": "parseable verifier-wrong numeric conclusion with substantive natural reasoning" + (" and near-correct final value" if near else ""),
            "similarity": sim,
            "interface_preserved": None,
        }

    if TRIVIAL_RE.search(detail):
        return None
    extracted = str(attempt.get("extracted_answer") or wrong)
    if not extracted.strip():
        return None
    if dataset == "mbpp":
        if "AssertionError" not in detail or INTERFACE_RE.search(detail):
            return None
        expected_symbols = expected_mbpp_symbols(source, correct)
        preserved = bool(expected_symbols) and expected_symbols.issubset(code_symbols(extracted))
        if not preserved:
            return None
        wrong_type = "semantic_test_failure"
        tier = 2
        partial = False
    else:
        passed = int(attempt.get("passed_tests", 0) or 0)
        total = int(attempt.get("total_tests", 0) or 0)
        partial = 0 < passed < total
        if "wrong_answer" not in detail:
            return None
        preserved = True  # A verifier wrong-answer result implies the APPS I/O/call interface executed.
        wrong_type = "partial_test_pass" if partial else "semantic_test_failure"
        tier = 3 if partial else 2
    return {
        "answer": wrong,
        "ref": reference,
        "origin": origin,
        "attempt": attempt,
        "score": 100 * tier + 25 * int(preserved) + 20 * sim + (5 if origin == "base" else 0),
        "wrong_type": wrong_type,
        "hardness_tier": tier,
        "hardness_reason": "natural code executes the required interface but fails deterministic semantic tests" + (" after passing at least one test" if partial else ""),
        "similarity": sim,
        "interface_preserved": preserved,
    }


def nested_counts(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    if len(fields) == 1:
        return dict(sorted(Counter(str(row[fields[0]]) for row in rows).items()))
    output = {}
    for value in sorted({str(row[fields[0]]) for row in rows}):
        output[value] = nested_counts([row for row in rows if str(row[fields[0]]) == value], *fields[1:])
    return output


def describe(values: list[float]) -> dict[str, float | int]:
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "mean": statistics.mean(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(ROOT / "data" / "behavior" / "unified_source_inventory.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "outputs" / "phase3_decision_only_v1" / "final_adapter"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "contrastive_pairs"))
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    inventory_path = Path(args.inventory).resolve()
    frozen_path = Path(args.frozen).resolve()
    output_dir = Path(args.output_dir).resolve()
    inventory = read_jsonl(inventory_path)
    frozen_ids = {str(row["source_id"]) for row in read_jsonl(frozen_path)}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    pairs = []
    excluded = []

    for dataset, requested in QUOTAS.items():
        candidates = []
        for meta in inventory:
            if meta["dataset"] != dataset or meta["bucket"] not in {"WC", "WW"} or meta["id"] in frozen_ids:
                continue
            source_id = str(meta["id"])
            source = resolver.resolve(meta["source_ref"], source_id)
            correct, correct_ref, correct_origin = correct_member(meta, source, resolver)
            options = [
                value for value in (
                    option_metadata(dataset, source, correct, option)
                    for option in wrong_options(meta, resolver)
                ) if value is not None
            ]
            if not options:
                excluded.append({"source_id": source_id, "dataset": dataset, "reason": "no_natural_plausible_wrong_candidate"})
                continue
            options.sort(key=lambda value: (-value["score"], rank("wrong-option", value["ref"])))
            best = options[0]
            candidates.append((meta, source, correct, correct_ref, correct_origin, best))
        candidates.sort(key=lambda item: (-item[5]["score"], rank(f"source:{dataset}", item[0]["id"])))

        accepted = 0
        for meta, source, correct, correct_ref, correct_origin, wrong_meta in candidates:
            source_id = str(meta["id"])
            correct_result = verifier.verify(source, correct)
            if not correct_result["passed"]:
                excluded.append({"source_id": source_id, "dataset": dataset, "reason": "correct_member_failed_fresh_verifier", "detail": correct_result["detail"]})
                continue
            wrong_result = verifier.verify(source, wrong_meta["answer"])
            if wrong_result["passed"]:
                excluded.append({"source_id": source_id, "dataset": dataset, "reason": "wrong_member_passed_fresh_verifier", "detail": wrong_result["detail"]})
                continue
            fresh_detail = str(wrong_result["detail"])
            if dataset == "mbpp" and ("AssertionError" not in fresh_detail or INTERFACE_RE.search(fresh_detail)):
                excluded.append({"source_id": source_id, "dataset": dataset, "reason": "fresh_failure_not_semantic", "detail": fresh_detail})
                continue
            if dataset == "apps" and "wrong_answer" not in fresh_detail:
                excluded.append({"source_id": source_id, "dataset": dataset, "reason": "fresh_failure_not_wrong_answer", "detail": fresh_detail})
                continue
            task_prompt = build_prompt(problem_for_prompt(source))
            template_id, template = TEMPLATES[accepted % len(TEMPLATES)]
            keep_prefix = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": correct},
                {"role": "user", "content": template},
            ]
            revise_prefix = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": wrong_meta["answer"]},
                {"role": "user", "content": template},
            ]
            lengths = []
            for messages in (keep_prefix, revise_prefix):
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
            if max(lengths) > args.max_length:
                excluded.append({"source_id": source_id, "dataset": dataset, "reason": "context_over_max_length", "keep_tokens": lengths[0], "revise_tokens": lengths[1]})
                continue
            correct_format = formatting(correct)
            wrong_format = formatting(wrong_meta["answer"])
            correct_format["tokens"] = len(tokenizer(correct, add_special_tokens=False)["input_ids"])
            wrong_format["tokens"] = len(tokenizer(wrong_meta["answer"], add_special_tokens=False)["input_ids"])
            fresh_passed = int(wrong_result.get("passed_tests", 0) or 0)
            fresh_total = int(wrong_result.get("total_tests", 0) or 0)
            wrong_type = wrong_meta["wrong_type"]
            if dataset == "apps" and 0 < fresh_passed < fresh_total:
                wrong_type = "partial_test_pass"
            pair_id = f"contrastive_v1::{dataset}::{source_id}"
            pairs.append({
                "pair_id": pair_id,
                "source_id": source_id,
                "dataset": dataset,
                "domain": meta["domain"],
                "problem_ref": meta["source_ref"],
                "correct_answer_ref": correct_ref,
                "wrong_answer_ref": wrong_meta["ref"],
                "correct_verified": True,
                "wrong_verified": False,
                "correct_verifier_detail": correct_result["detail"],
                "wrong_verifier_detail": wrong_result["detail"],
                "wrong_type": wrong_type,
                "hardness_tier": wrong_meta["hardness_tier"],
                "hardness_reason": wrong_meta["hardness_reason"],
                "hardness_is_heuristic_not_human_rated": True,
                "neutral_template_id": template_id,
                "neutral_review_text": template,
                "keep_example": {"decision": "KEEP", "behavior_id": f"{pair_id}::KEEP"},
                "revise_example": {"decision": "REVISE", "behavior_id": f"{pair_id}::REVISE"},
                "source_origin": {"correct": correct_origin, "wrong": wrong_meta["origin"]},
                "answer_matching": {
                    "similarity_score": wrong_meta["similarity"],
                    "correct": correct_format,
                    "wrong": wrong_format,
                    "char_length_ratio_wrong_over_correct": wrong_format["chars"] / max(correct_format["chars"], 1),
                    "token_lengths_full_prompt": {"KEEP": lengths[0], "REVISE": lengths[1]},
                    "function_signature_or_interface_preserved": wrong_meta["interface_preserved"],
                },
                "code_execution": {
                    "wrong_executed_to_semantic_check": dataset in {"apps", "mbpp"},
                    "fresh_passed_tests": fresh_passed if dataset == "apps" else None,
                    "fresh_total_tests": fresh_total if dataset == "apps" else None,
                    "partial_pass": dataset == "apps" and 0 < fresh_passed < fresh_total,
                } if dataset in {"apps", "mbpp"} else None,
                "_keep_prefix": keep_prefix,
                "_revise_prefix": revise_prefix,
            })
            accepted += 1
            if accepted == requested:
                break
        if accepted != requested:
            raise RuntimeError(f"Only {accepted}/{requested} clean pairs available for {dataset}")

    pairs.sort(key=lambda row: (row["dataset"], row["source_id"]))
    behavioral_rows = []
    pair_file_rows = []
    for pair in pairs:
        keep_prefix = pair.pop("_keep_prefix")
        revise_prefix = pair.pop("_revise_prefix")
        pair_file_rows.append(pair)
        for decision, prefix in (("KEEP", keep_prefix), ("REVISE", revise_prefix)):
            behavioral_rows.append({
                "construction_id": pair[f"{decision.lower()}_example"]["behavior_id"],
                "pair_id": pair["pair_id"],
                "source_id": pair["source_id"],
                "dataset": pair["dataset"],
                "domain": pair["domain"],
                "decision": decision,
                "neutral_template_id": pair["neutral_template_id"],
                "feedback_type": "neutral_review",
                "messages": prefix + [{"role": "assistant", "content": f"<decision>{decision}</decision>"}],
                "source_ref": pair["problem_ref"],
                "answer_ref": pair["correct_answer_ref"] if decision == "KEEP" else pair["wrong_answer_ref"],
            })

    pair_ids = [row["pair_id"] for row in pair_file_rows]
    source_ids = [row["source_id"] for row in pair_file_rows]
    problem_text_by_pair = {
        row["pair_id"]: row["messages"][0]["content"]
        for row in behavioral_rows if row["decision"] == "KEEP"
    }
    problem_hashes = [
        hashlib.sha256(" ".join(problem_text_by_pair[row["pair_id"]].casefold().split()).encode()).hexdigest()
        for row in pair_file_rows
    ]
    errors = []
    if len(pair_file_rows) != 120 or len(behavioral_rows) != 240:
        errors.append("Expected 120 pairs and 240 behavioral rows")
    if len(set(pair_ids)) != len(pair_ids) or len(set(source_ids)) != len(source_ids) or len(set(problem_hashes)) != len(problem_hashes):
        errors.append("Duplicate pair/source/problem detected")
    if any(row["source_id"] in frozen_ids for row in pair_file_rows):
        errors.append("Frozen overlap")
    if any(row["dataset"] == "humaneval" for row in pair_file_rows):
        errors.append("HumanEval contamination")
    if any(not row["correct_verified"] or row["wrong_verified"] for row in pair_file_rows):
        errors.append("Verifier polarity error")
    for pair in pair_file_rows:
        members = [row for row in behavioral_rows if row["pair_id"] == pair["pair_id"]]
        if len(members) != 2 or {row["decision"] for row in members} != {"KEEP", "REVISE"}:
            errors.append(f"Broken pair: {pair['pair_id']}")
        if len({row["messages"][0]["content"] for row in members}) != 1 or len({row["messages"][2]["content"] for row in members}) != 1:
            errors.append(f"Problem/template mismatch: {pair['pair_id']}")
    template_label = nested_counts(behavioral_rows, "decision", "neutral_template_id")
    if template_label["KEEP"] != template_label["REVISE"]:
        errors.append("Template-label imbalance")
    dataset_label = nested_counts(behavioral_rows, "dataset", "decision")
    domain_label = nested_counts(behavioral_rows, "domain", "decision")
    if any(value["KEEP"] != value["REVISE"] for value in dataset_label.values()) or any(value["KEEP"] != value["REVISE"] for value in domain_label.values()):
        errors.append("Dataset/domain label shortcut")
    if errors:
        raise RuntimeError("; ".join(errors))

    pair_path = output_dir / "contrastive_pairs.jsonl"
    behavior_path = output_dir / "contrastive_behavior_rows.jsonl"
    summary_path = output_dir / "contrastive_pairs_summary.json"
    write_jsonl(pair_path, pair_file_rows)
    write_jsonl(behavior_path, behavioral_rows)

    correct_chars = [row["answer_matching"]["correct"]["chars"] for row in pair_file_rows]
    wrong_chars = [row["answer_matching"]["wrong"]["chars"] for row in pair_file_rows]
    correct_tokens = [row["answer_matching"]["correct"]["tokens"] for row in pair_file_rows]
    wrong_tokens = [row["answer_matching"]["wrong"]["tokens"] for row in pair_file_rows]
    pooled = math.sqrt((statistics.pvariance(correct_chars) + statistics.pvariance(wrong_chars)) / 2)
    length_smd = (statistics.mean(wrong_chars) - statistics.mean(correct_chars)) / pooled if pooled else 0.0
    token_pooled = math.sqrt((statistics.pvariance(correct_tokens) + statistics.pvariance(wrong_tokens)) / 2)
    token_smd = (statistics.mean(wrong_tokens) - statistics.mean(correct_tokens)) / token_pooled if token_pooled else 0.0
    code_pairs = [row for row in pair_file_rows if row["domain"] == "code"]
    summary = {
        "schema_version": "phase3_same_problem_contrastive_pairs_v1",
        "seed": SEED,
        "total_pairs": len(pair_file_rows),
        "total_behavioral_rows": len(behavioral_rows),
        "code_pairs": len(code_pairs),
        "math_pairs": len(pair_file_rows) - len(code_pairs),
        "pairs_by_dataset": nested_counts(pair_file_rows, "dataset"),
        "wrong_type_distribution": nested_counts(pair_file_rows, "wrong_type"),
        "hardness_distribution": nested_counts(pair_file_rows, "hardness_tier", "dataset"),
        "hard_plausible_wrong_count": sum(row["hardness_tier"] >= 2 for row in pair_file_rows),
        "neutral_template_distribution_by_label": template_label,
        "answer_length": {
            "chars": {"correct": describe(correct_chars), "wrong": describe(wrong_chars), "standardized_mean_difference_wrong_minus_correct": length_smd},
            "tokens": {"correct": describe(correct_tokens), "wrong": describe(wrong_tokens), "standardized_mean_difference_wrong_minus_correct": token_smd},
        },
        "answer_matching": {
            "mean_similarity_score": statistics.mean(row["answer_matching"]["similarity_score"] for row in pair_file_rows),
            "same_code_fence_usage_count": sum(row["answer_matching"]["correct"]["code_fence"] == row["answer_matching"]["wrong"]["code_fence"] for row in pair_file_rows),
            "same_code_fence_usage_code_pairs": sum(row["answer_matching"]["correct"]["code_fence"] == row["answer_matching"]["wrong"]["code_fence"] for row in code_pairs),
            "mean_absolute_line_count_difference_code": statistics.mean(abs(row["answer_matching"]["correct"]["lines"] - row["answer_matching"]["wrong"]["lines"]) for row in code_pairs),
            "code_interface_preserved_count": sum(row["answer_matching"]["function_signature_or_interface_preserved"] is True for row in code_pairs),
            "code_pair_count": len(code_pairs),
        },
        "code_execution": {
            "semantic_execution_failure_count": len(code_pairs),
            "partial_pass_count": sum(row["code_execution"]["partial_pass"] for row in code_pairs),
            "trivial_invalid_included_count": 0,
        },
        "label_balance": {"dataset": dataset_label, "domain": domain_label},
        "duplicate_checks": {"pair_ids": 0, "source_ids": 0, "problem_pairs": 0},
        "overlap_checks": {"frozen_eval": [], "humaneval": []},
        "verifier_checks": {"fresh_correct_pass_count": len(pair_file_rows), "fresh_wrong_fail_count": len(pair_file_rows), "failures_in_selected": 0},
        "excluded_source_count": len(excluded),
        "excluded_reason_distribution": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "excluded_sources": excluded,
        "source_origin_distribution": {
            "correct": dict(sorted(Counter(row["source_origin"]["correct"] for row in pair_file_rows).items())),
            "wrong": dict(sorted(Counter(row["source_origin"]["wrong"] for row in pair_file_rows).items())),
        },
        "shortcut_risk_diagnostics": {
            "template_label": "low: paired members share the same template and distributions are identical",
            "dataset_label": "low: each problem contributes one KEEP and one REVISE row",
            "domain_label": "low: each problem contributes one KEEP and one REVISE row",
            "task_identity": "low: labels are paired within the identical task",
            "answer_length": "report SMD and natural pair statistics; answers were selected, never rewritten",
            "formatting": "code-fence and line/length metadata retained for audit",
            "reference_vs_model_origin": "medium: many WW pairs use verified references as KEEP and natural model attempts as REVISE",
        },
        "outputs": {
            "pairs": {"path": str(pair_path), "sha256": hashlib.sha256(pair_path.read_bytes()).hexdigest()},
            "behavior": {"path": str(behavior_path), "sha256": hashlib.sha256(behavior_path.read_bytes()).hexdigest()},
        },
        "validation": {
            "fresh_verifier_polarity": True,
            "same_problem_per_pair": True,
            "same_dataset_domain_per_pair": True,
            "no_frozen_eval": True,
            "no_humaneval": True,
            "no_cw_wrong": True,
            "unique_pairs_sources_problems": True,
            "label_balanced_templates": True,
            "no_template_dataset_domain_label_shortcut": True,
            "answer_imbalance_measured": True,
            "code_interface_preserved": all(row["answer_matching"]["function_signature_or_interface_preserved"] is True for row in code_pairs),
            "no_trivial_invalid_outputs": True,
            "synthetic_wrong_answers": False,
            "training_started": False,
            "all_passed": True,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
