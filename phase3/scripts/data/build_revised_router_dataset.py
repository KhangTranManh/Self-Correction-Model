"""Build a hard-negative-heavy Phase 3 decision-router dataset.

The builder uses only cached Self_Correction_v1 attempts backed by existing
Phase 3 sources, then re-runs the deterministic verifier before admitting a
row.  Training targets contain only the strict KEEP/REVISE contract.  Frozen
two-stage evaluation sources and the original decision-only dev/test splits
are excluded to preserve honest evaluation sets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import (
    ProvenanceResolver,
    VerificationEngine,
    build_prompt,
    problem_for_prompt,
)


ROOT = Path(__file__).resolve().parents[2]
SEED = 271828
DEFAULT_INVENTORY = ROOT / "data" / "behavior" / "unified_source_inventory.jsonl"
DEFAULT_FROZEN = ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"
DEFAULT_OLD_DECISION = ROOT / "data" / "decision_only" / "decision_only_dataset.jsonl"
DEFAULT_BEHAVIOR = ROOT / "data" / "behavior" / "selection" / "behavior_selection_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "revised_router"

TEMPLATES = (
    ("neutral_review_01", "Review your previous answer carefully and decide whether it should be kept or revised."),
    ("neutral_review_02", "Re-examine your previous answer and decide whether to keep it or revise it."),
    ("neutral_review_03", "Check your previous response carefully, then decide if it should be kept or revised."),
    ("neutral_review_04", "Evaluate your previous response on its own merits and choose whether to keep or revise it."),
    ("neutral_review_05", "Inspect the answer you gave above and determine whether it should be kept or revised."),
)

# Exactly 250 rows: 100 KEEP (40%) and 150 REVISE (60%).  Every dataset and
# both domains have the same 40/60 prior, preventing a dataset/domain shortcut.
QUOTAS = {
    ("gsm8k", "KEEP"): 50,
    ("gsm8k", "REVISE"): 75,
    ("mbpp", "KEEP"): 20,
    ("mbpp", "REVISE"): 30,
    ("apps", "KEEP"): 30,
    ("apps", "REVISE"): 45,
}
DEV_QUOTAS = {
    ("gsm8k", "KEEP"): 10,
    ("gsm8k", "REVISE"): 15,
    ("mbpp", "KEEP"): 4,
    ("mbpp", "REVISE"): 6,
    ("apps", "KEEP"): 6,
    ("apps", "REVISE"): 9,
}

DECISION_RE = re.compile(r"<decision>(KEEP|REVISE)</decision>")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?")
OBVIOUS_FAILURE_RE = re.compile(
    r"syntaxerror|indentationerror|no code|no_code|empty candidate|could not extract|"
    r"modulenotfounderror|importerror|timeout",
    re.IGNORECASE,
)
INTERFACE_FAILURE_RE = re.compile(r"nameerror|is not defined|attributeerror", re.IGNORECASE)
EXECUTED_WRONG_RE = re.compile(r"wrong_answer|assertionerror", re.IGNORECASE)
REFUSAL_RE = re.compile(r"i (?:cannot|can't)|unable to|as an ai|no answer", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_rank(context: str, source_id: str) -> str:
    return sha256_bytes(f"{SEED}|revised-router|{context}|{source_id}".encode())


def normalized_hash(*parts: str) -> str:
    normalized = "\n<SEP>\n".join(" ".join(part.casefold().split()) for part in parts)
    return sha256_bytes(normalized.encode("utf-8"))


def parse_number(text: Any) -> Fraction | None:
    matches = NUMBER_RE.findall(str(text or ""))
    if not matches:
        return None
    token = matches[-1].replace(",", "").replace(" ", "")
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def historical_hardness(
    source_meta: dict[str, Any], source: dict[str, Any], attempt: dict[str, Any]
) -> dict[str, Any]:
    """Return auditable, label-independent difficulty signals for a wrong row."""
    output = str(attempt.get("initial_output", ""))
    detail = str(attempt.get("verifier_detail", ""))
    words = re.findall(r"\w+", output, flags=re.UNICODE)
    substantive = len(words) >= 24 and not REFUSAL_RE.search(output)
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        "answer_chars": len(output),
        "answer_words": len(words),
        "historical_verifier_detail": detail,
    }
    tier = 0

    if source_meta["dataset"] == "gsm8k":
        predicted = parse_number(attempt.get("extracted_answer") or output)
        reference = parse_number(source.get("reference_answer") or source.get("ground_truth"))
        metrics["predicted_numeric"] = str(predicted) if predicted is not None else None
        metrics["reference_numeric"] = str(reference) if reference is not None else None
        if predicted is not None and reference is not None:
            absolute = abs(float(predicted - reference))
            relative = absolute / max(abs(float(reference)), 1.0)
            metrics["absolute_error"] = absolute
            metrics["relative_error"] = relative
            reasons.append("numeric_final_answer_parseable")
            same_magnitude = (
                predicted != 0
                and reference != 0
                and 0.5 <= abs(float(predicted / reference)) <= 2.0
            )
            near = absolute <= 2.0 or relative <= 0.15
            if near:
                reasons.append("near_correct_numeric_answer")
            elif same_magnitude:
                reasons.append("same_order_of_magnitude")
            if substantive and near:
                tier = 3
            elif substantive and (same_magnitude or source_meta.get("bucket") == "CW"):
                tier = 2
            elif substantive:
                tier = 1
        if substantive:
            reasons.append("substantive_reasoning")
        if source_meta.get("bucket") == "CW":
            reasons.append("v1_regression_from_base_correct")
            tier = max(tier, 2 if substantive else 1)
    else:
        extracted = str(attempt.get("extracted_answer") or "")
        code_like = bool(re.search(r"\bdef\s+\w+|\bclass\s+\w+|\binput\s*\(", extracted))
        substantial_code = code_like and len(extracted) >= 100
        obvious_failure = bool(OBVIOUS_FAILURE_RE.search(detail))
        interface_failure = bool(INTERFACE_FAILURE_RE.search(detail))
        executed_wrong = bool(EXECUTED_WRONG_RE.search(detail)) and not obvious_failure
        passed_tests = int(attempt.get("passed_tests", 0) or 0)
        total_tests = int(attempt.get("total_tests", 0) or 0)
        partial_pass = 0 < passed_tests < total_tests
        metrics.update(
            {
                "extracted_code_chars": len(extracted),
                "passed_tests_historical": passed_tests,
                "total_tests_historical": total_tests,
                "partial_test_pass_historical": partial_pass,
                "executed_wrong_historical": executed_wrong,
                "interface_failure_historical": interface_failure,
                "obvious_runtime_or_format_failure_historical": obvious_failure,
            }
        )
        if substantial_code:
            reasons.append("substantive_executable_shaped_code")
        if partial_pass:
            reasons.append("passes_some_tests_but_not_all")
            tier = 3 if substantial_code else 2
        elif executed_wrong and not interface_failure:
            reasons.append("executes_but_fails_semantic_test")
            tier = 2 if substantial_code else 1
        elif substantial_code and interface_failure:
            reasons.append("plausible_code_with_interface_mismatch")
            tier = 1
        elif substantial_code and not obvious_failure:
            tier = 1

    return {
        "hardness_tier": tier,
        "hard_revise": tier >= 2,
        "hardness_reasons": reasons,
        "hardness_metrics": metrics,
    }


def update_with_fresh_code_signals(
    hardness: dict[str, Any], dataset: str, fresh: dict[str, Any]
) -> dict[str, Any]:
    updated = json.loads(json.dumps(hardness))
    if dataset not in {"mbpp", "apps"}:
        return updated
    detail = str(fresh.get("detail", ""))
    reasons = list(updated["hardness_reasons"])
    metrics = dict(updated["hardness_metrics"])
    obvious_failure = bool(OBVIOUS_FAILURE_RE.search(detail))
    interface_failure = bool(INTERFACE_FAILURE_RE.search(detail))
    executed_wrong = bool(EXECUTED_WRONG_RE.search(detail)) and not obvious_failure
    passed_tests = int(fresh.get("passed_tests", 0) or 0)
    total_tests = int(fresh.get("total_tests", 0) or 0)
    partial_pass = 0 < passed_tests < total_tests
    metrics.update(
        {
            "fresh_verifier_detail": detail,
            "passed_tests_fresh": passed_tests,
            "total_tests_fresh": total_tests,
            "partial_test_pass_fresh": partial_pass,
            "executed_wrong_fresh": executed_wrong,
            "interface_failure_fresh": interface_failure,
            "obvious_runtime_or_format_failure_fresh": obvious_failure,
        }
    )
    if partial_pass:
        tier = 3
        reasons.append("freshly_passes_some_tests_but_not_all")
    elif executed_wrong and not interface_failure:
        tier = max(2, int(updated["hardness_tier"]))
        reasons.append("freshly_executes_but_fails_semantic_test")
    else:
        tier = min(int(updated["hardness_tier"]), 1)
    updated.update(
        {
            "hardness_tier": tier,
            "hard_revise": tier >= 2,
            "hardness_reasons": sorted(set(reasons)),
            "hardness_metrics": metrics,
        }
    )
    return updated


def nested_counts(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    if len(fields) == 1:
        return dict(sorted(Counter(str(row[fields[0]]) for row in rows).items()))
    output: dict[str, Any] = {}
    for value in sorted({str(row[fields[0]]) for row in rows}):
        output[value] = nested_counts(
            [row for row in rows if str(row[fields[0]]) == value], *fields[1:]
        )
    return output


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 3),
        "max": max(values),
    }


def length_matched_order(
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
    target_lengths: list[int],
    context: str,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Greedily order KEEP candidates against REVISE answer-length quantiles."""
    remaining = list(candidates)
    ordered = []
    for target in target_lengths:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                abs(len(str(remaining[index][2].get("initial_output", ""))) - target),
                stable_rank(context, str(remaining[index][0]["id"])),
            ),
        )
        ordered.append(remaining.pop(best_index))
    remaining.sort(
        key=lambda item: (
            stable_rank(f"{context}:remainder", str(item[0]["id"])),
            str(item[0]["id"]),
        )
    )
    return ordered + remaining


def assign_splits(rows: list[dict[str, Any]]) -> None:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["decision"])].append(row)
    for key, members in groups.items():
        members.sort(key=lambda row: (stable_rank(f"split:{key}", row["source_id"]), row["source_id"]))
        dev_count = DEV_QUOTAS[key]
        for index, row in enumerate(members):
            row["split"] = "dev" if index < dev_count else "train"


def assign_templates(rows: list[dict[str, Any]]) -> None:
    # Dataset+label totals are divisible by six, giving identical template
    # proportions for KEEP and REVISE within GSM8K, MBPP, and APPS (and thus
    # within both domains as well).
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["decision"])].append(row)
    for key, members in groups.items():
        members.sort(key=lambda row: (stable_rank(f"template:{key}", row["source_id"]), row["source_id"]))
        for index, row in enumerate(members):
            template_id, template = TEMPLATES[index % len(TEMPLATES)]
            row["neutral_template_id"] = template_id
            row["neutral_review_text"] = template
            row["messages"] = [
                {"role": "user", "content": row.pop("_task_prompt")},
                {"role": "assistant", "content": row.pop("_v1_output")},
                {"role": "user", "content": template},
                {"role": "assistant", "content": f"<decision>{row['decision']}</decision>"},
            ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--frozen-eval", default=str(DEFAULT_FROZEN))
    parser.add_argument("--old-decision", default=str(DEFAULT_OLD_DECISION))
    parser.add_argument("--behavior-manifest", default=str(DEFAULT_BEHAVIOR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    inventory_path = Path(args.inventory).resolve()
    frozen_path = Path(args.frozen_eval).resolve()
    old_decision_path = Path(args.old_decision).resolve()
    behavior_path = Path(args.behavior_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()

    inventory = read_jsonl(inventory_path)
    frozen_ids = {str(row["source_id"]) for row in read_jsonl(frozen_path)}
    old_rows = read_jsonl(old_decision_path)
    old_ids_by_split = {
        split: {str(row["source_id"]) for row in old_rows if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    behavior_ids = {
        str(row["source_id"]) for row in read_jsonl(behavior_path)
    } if behavior_path.is_file() else set()
    excluded_ids = frozen_ids | old_ids_by_split["dev"] | old_ids_by_split["test"]

    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    verification_exclusions: list[dict[str, Any]] = []
    candidate_counts: dict[str, Any] = {}

    cell_order = [
        (dataset, decision)
        for decision in ("REVISE", "KEEP")
        for dataset in ("gsm8k", "mbpp", "apps")
    ]
    for dataset, decision in cell_order:
        requested = QUOTAS[(dataset, decision)]
        expected_correct = decision == "KEEP"
        candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for source_meta in inventory:
            source_id = str(source_meta["id"])
            if (
                source_meta["dataset"] != dataset
                or bool(source_meta["v1_correct"]) != expected_correct
                or source_id in excluded_ids
                or source_id in selected_ids
            ):
                continue
            source = resolver.resolve(str(source_meta["source_ref"]), source_id)
            attempt = resolver.resolve(str(source_meta["v1_attempt_ref"]), source_id)
            hardness = (
                historical_hardness(source_meta, source, attempt)
                if decision == "REVISE"
                else {"hardness_tier": 0, "hard_revise": False, "hardness_reasons": [], "hardness_metrics": {}}
            )
            candidates.append((source_meta, source, attempt, hardness))

        candidate_counts[f"{dataset}:{decision}"] = {
            "available_after_holdout_exclusion": len(candidates),
            "historically_hard_revise": sum(int(item[3]["hard_revise"]) for item in candidates),
        }
        candidates.sort(
            key=lambda item: (
                -int(item[3]["hardness_tier"]),
                stable_rank(f"select:{dataset}:{decision}", str(item[0]["id"])),
                str(item[0]["id"]),
            )
        )
        if decision == "KEEP":
            revise_lengths = sorted(
                len(str(row["_v1_output"]))
                for row in selected
                if row["dataset"] == dataset and row["decision"] == "REVISE"
            )
            if len(revise_lengths) != QUOTAS[(dataset, "REVISE")]:
                raise RuntimeError(f"REVISE cell must be selected before KEEP for {dataset}")
            targets = [
                revise_lengths[min(len(revise_lengths) - 1, int((index + 0.5) * len(revise_lengths) / requested))]
                for index in range(requested)
            ]
            candidates = length_matched_order(candidates, targets, f"length-match:{dataset}")

        accepted = 0
        for source_meta, source, attempt, hardness in candidates:
            source_id = str(source_meta["id"])
            fresh = verifier.verify(source, str(attempt["initial_output"]))
            if bool(fresh["passed"]) != expected_correct:
                verification_exclusions.append(
                    {
                        "source_id": source_id,
                        "dataset": dataset,
                        "decision": decision,
                        "historical_v1_correct": bool(source_meta["v1_correct"]),
                        "fresh_passed": bool(fresh["passed"]),
                        "fresh_verifier_detail": str(fresh["detail"]),
                    }
                )
                continue
            if decision == "REVISE":
                hardness = update_with_fresh_code_signals(hardness, dataset, fresh)
            task_prompt = build_prompt(problem_for_prompt(source))
            selected.append(
                {
                    "construction_id": f"revised_router_v1::{source_id}",
                    "source_id": source_id,
                    "dataset": dataset,
                    "domain": str(source_meta["domain"]),
                    "source_split": str(source_meta["source_split"]),
                    "bucket": str(source_meta["bucket"]),
                    "decision": decision,
                    "v1_correct": bool(source_meta["v1_correct"]),
                    "hard_revise": bool(hardness["hard_revise"]),
                    "hardness_tier": int(hardness["hardness_tier"]),
                    "hardness_reasons": list(hardness["hardness_reasons"]),
                    "hardness_metrics": dict(hardness["hardness_metrics"]),
                    "label_source": "existing_deterministic_verifier_reconfirmed",
                    "verification_method": verifier.method(source),
                    "verifier_detail": str(fresh["detail"]),
                    "teacher_used": False,
                    "feedback_type": "neutral_review",
                    "source_ref": str(source_meta["source_ref"]),
                    "v1_attempt_ref": str(source_meta["v1_attempt_ref"]),
                    "selection_seed": SEED,
                    "training_format": "messages_final_assistant_decision_only",
                    "_task_prompt": task_prompt,
                    "_v1_output": str(attempt["initial_output"]),
                }
            )
            selected_ids.add(source_id)
            accepted += 1
            if accepted == requested:
                break
        if accepted != requested:
            raise RuntimeError(f"Only accepted {accepted}/{requested} rows for {dataset}/{decision}")

    assign_splits(selected)
    assign_templates(selected)
    selected.sort(key=lambda row: (row["split"], row["domain"], row["decision"], row["dataset"], row["source_id"]))

    errors: list[str] = []
    if len(selected) != 250 or len(selected_ids) != 250:
        errors.append(f"Expected 250 unique sources; rows={len(selected)}, unique={len(selected_ids)}")
    if Counter(row["decision"] for row in selected) != Counter({"REVISE": 150, "KEEP": 100}):
        errors.append("Global label quota mismatch")
    for key, requested in QUOTAS.items():
        observed = sum(1 for row in selected if (row["dataset"], row["decision"]) == key)
        if observed != requested:
            errors.append(f"Quota mismatch {key}: {observed}!={requested}")
    if any(row["source_id"] in frozen_ids for row in selected):
        errors.append("Frozen-evaluation source contamination")
    if any(row["source_id"] in old_ids_by_split["test"] for row in selected):
        errors.append("Original decision-only test contamination")
    if any(row["source_id"] in old_ids_by_split["dev"] for row in selected):
        errors.append("Original decision-only dev contamination")
    for row in selected:
        target = str(row["messages"][-1]["content"])
        match = DECISION_RE.fullmatch(target)
        if not match or match.group(1) != row["decision"]:
            errors.append(f"Invalid decision target: {row['source_id']}")
        prompt_blob = "\n".join(message["content"] for message in row["messages"][:-1])
        if re.search(r"<decision>(?:KEEP|REVISE)</decision>", prompt_blob):
            errors.append(f"Label leaked into prompt: {row['source_id']}")
    split_ids = {
        split: {row["source_id"] for row in selected if row["split"] == split}
        for split in ("train", "dev")
    }
    if split_ids["train"] & split_ids["dev"]:
        errors.append("Train/dev source overlap")

    pair_hashes = [normalized_hash(row["messages"][0]["content"], row["messages"][1]["content"]) for row in selected]
    problem_hashes = [normalized_hash(row["messages"][0]["content"]) for row in selected]
    duplicate_pair_count = len(pair_hashes) - len(set(pair_hashes))
    duplicate_problem_count = len(problem_hashes) - len(set(problem_hashes))
    if duplicate_pair_count:
        errors.append(f"Duplicate problem+answer pairs: {duplicate_pair_count}")
    if duplicate_problem_count:
        errors.append(f"Duplicate normalized problems: {duplicate_problem_count}")

    template_domain_label = nested_counts(selected, "domain", "decision", "neutral_template_id")
    template_dataset_label = nested_counts(selected, "dataset", "decision", "neutral_template_id")
    for domain in ("math", "code"):
        keep = template_domain_label[domain]["KEEP"]
        revise = template_domain_label[domain]["REVISE"]
        keep_total = sum(keep.values())
        revise_total = sum(revise.values())
        for template_id, _ in TEMPLATES:
            if not math.isclose(keep[template_id] / keep_total, revise[template_id] / revise_total):
                errors.append(f"Template proportion mismatch: {domain}/{template_id}")
    for dataset in ("gsm8k", "mbpp", "apps"):
        keep = template_dataset_label[dataset]["KEEP"]
        revise = template_dataset_label[dataset]["REVISE"]
        keep_total = sum(keep.values())
        revise_total = sum(revise.values())
        for template_id, _ in TEMPLATES:
            if not math.isclose(keep[template_id] / keep_total, revise[template_id] / revise_total):
                errors.append(f"Template proportion mismatch: {dataset}/{template_id}")

    if errors:
        raise RuntimeError("; ".join(errors))

    dataset_path = output_dir / "revised_router_dataset.jsonl"
    train_path = output_dir / "revised_router_train.jsonl"
    dev_path = output_dir / "revised_router_dev.jsonl"
    summary_path = output_dir / "revised_router_summary.json"
    audit_path = output_dir / "hard_revise_audit.jsonl"
    write_jsonl(dataset_path, selected)
    write_jsonl(train_path, [row for row in selected if row["split"] == "train"])
    write_jsonl(dev_path, [row for row in selected if row["split"] == "dev"])

    hard_rows = [row for row in selected if row["decision"] == "REVISE" and row["hard_revise"]]
    audit_rows = []
    for row in hard_rows:
        audit_rows.append(
            {
                "source_id": row["source_id"],
                "dataset": row["dataset"],
                "domain": row["domain"],
                "split": row["split"],
                "bucket": row["bucket"],
                "hardness_tier": row["hardness_tier"],
                "hardness_reasons": row["hardness_reasons"],
                "hardness_metrics": row["hardness_metrics"],
                "question": row["messages"][0]["content"],
                "v1_answer": row["messages"][1]["content"],
                "neutral_review_text": row["neutral_review_text"],
                "verifier_detail": row["verifier_detail"],
                "source_ref": row["source_ref"],
                "v1_attempt_ref": row["v1_attempt_ref"],
            }
        )
    write_jsonl(audit_path, audit_rows)

    answer_lengths = {
        decision: [len(row["messages"][1]["content"]) for row in selected if row["decision"] == decision]
        for decision in ("KEEP", "REVISE")
    }
    mean_keep = statistics.mean(answer_lengths["KEEP"])
    mean_revise = statistics.mean(answer_lengths["REVISE"])
    pooled = math.sqrt(
        (statistics.pvariance(answer_lengths["KEEP"]) + statistics.pvariance(answer_lengths["REVISE"])) / 2
    )
    length_smd = (mean_revise - mean_keep) / pooled if pooled else 0.0

    shortcut_risks = [
        {
            "risk": "class_prior_bias",
            "level": "intentional_medium",
            "evidence": "REVISE is intentionally 60% of rows to address low recall; evaluate balanced accuracy and KEEP recall, not raw accuracy alone.",
        },
        {
            "risk": "template_label_correlation",
            "level": "low",
            "evidence": "All five neutral templates have identical proportions for KEEP and REVISE within each dataset and domain.",
        },
        {
            "risk": "dataset_or_domain_label_correlation",
            "level": "low",
            "evidence": "Every dataset and both domains use the same 40% KEEP / 60% REVISE prior.",
        },
        {
            "risk": "answer_length_correlation",
            "level": "medium" if abs(length_smd) >= 0.5 else "low",
            "evidence": f"Standardized mean difference (REVISE minus KEEP) is {length_smd:.3f}; raw answer-length distributions are included below.",
        },
        {
            "risk": "verifier_or_label_metadata_leakage",
            "level": "low",
            "evidence": "Verifier fields and hardness annotations are metadata only; none appears in the input messages consumed by SFT.",
        },
        {
            "risk": "source_reuse_from_prior_router_training",
            "level": "medium",
            "evidence": "Some original training sources are intentionally reusable for a revised mix; original dev/test and frozen two-stage sources are excluded and overlap counts are explicit.",
        },
        {
            "risk": "hardness_heuristic_noise",
            "level": "medium",
            "evidence": "Hardness is a reproducible proxy based on numerical proximity, substantive reasoning, execution, and partial tests—not a human difficulty judgment. Audit rows are provided for manual review.",
        },
    ]

    selected_set = set(selected_ids)
    summary = {
        "schema_version": "phase3_revised_router_v1",
        "seed": SEED,
        "purpose": "increase neutral-review REVISE recall without collapsing KEEP behavior",
        "total_rows": len(selected),
        "unique_sources": len(selected_ids),
        "label_counts": nested_counts(selected, "decision"),
        "label_percent": {
            decision: round(100 * sum(row["decision"] == decision for row in selected) / len(selected), 3)
            for decision in ("KEEP", "REVISE")
        },
        "hard_revise_count": len(hard_rows),
        "hard_revise_percent_of_revise": round(100 * len(hard_rows) / 150, 3),
        "hard_revise_by_dataset": nested_counts(hard_rows, "dataset", "hardness_tier"),
        "hardness_tier_distribution_for_revise": nested_counts(
            [row for row in selected if row["decision"] == "REVISE"], "dataset", "hardness_tier"
        ),
        "counts_by_split": nested_counts(selected, "split", "decision"),
        "domain_balance": nested_counts(selected, "domain", "decision"),
        "dataset_balance": nested_counts(selected, "dataset", "decision"),
        "template_distribution": {
            "overall": nested_counts(selected, "decision", "neutral_template_id"),
            "by_domain": template_domain_label,
            "by_dataset": template_dataset_label,
            "by_split": nested_counts(selected, "split", "decision", "neutral_template_id"),
        },
        "bucket_distribution": nested_counts(selected, "decision", "bucket"),
        "answer_length_chars": {
            decision: distribution(values) for decision, values in answer_lengths.items()
        },
        "candidate_pool": candidate_counts,
        "fresh_verification": {
            "selected_rows_reconfirmed": len(selected),
            "historical_label_mismatch_exclusions": verification_exclusions,
            "historical_label_mismatch_exclusion_count": len(verification_exclusions),
        },
        "source_overlap_checks": {
            "internal_train_dev": [],
            "frozen_eval": sorted(selected_set & frozen_ids),
            "original_decision_test": sorted(selected_set & old_ids_by_split["test"]),
            "original_decision_dev_count": len(selected_set & old_ids_by_split["dev"]),
            "original_decision_train_count": len(selected_set & old_ids_by_split["train"]),
            "behavior_selection_manifest_count": len(selected_set & behavior_ids),
            "note": "Original train reuse is reported. Frozen evaluation plus original dev/test are protected holdouts.",
        },
        "duplicate_checks": {
            "duplicate_source_ids": len(selected) - len(selected_ids),
            "duplicate_normalized_problem_answer_pairs": duplicate_pair_count,
            "duplicate_normalized_problems": duplicate_problem_count,
        },
        "shortcut_risk_estimates": shortcut_risks,
        "validation": {
            "all_sources_from_existing_phase3_inventory": True,
            "all_labels_freshly_reconfirmed": True,
            "all_feedback_neutral": all(row["feedback_type"] == "neutral_review" for row in selected),
            "strict_decision_only_targets": True,
            "no_teacher_labels": all(not row["teacher_used"] for row in selected),
            "no_frozen_eval_overlap": True,
            "no_original_decision_test_overlap": True,
            "no_original_decision_dev_overlap": True,
            "one_row_per_source": True,
            "no_normalized_content_duplicates": True,
            "same_template_distribution_by_label_within_domain_and_dataset": True,
            "same_label_prior_in_every_dataset_and_domain": True,
            "training_not_started": True,
            "all_passed": True,
        },
        "inputs": {
            "inventory": {"path": str(inventory_path), "sha256": file_hash(inventory_path)},
            "frozen_eval": {"path": str(frozen_path), "sha256": file_hash(frozen_path)},
            "old_decision": {"path": str(old_decision_path), "sha256": file_hash(old_decision_path)},
        },
    }
    write_json(summary_path, summary)
    summary["outputs"] = {
        "dataset": {"path": str(dataset_path), "sha256": file_hash(dataset_path)},
        "train": {"path": str(train_path), "sha256": file_hash(train_path)},
        "dev": {"path": str(dev_path), "sha256": file_hash(dev_path)},
        "summary": {"path": str(summary_path), "sha256_before_self_reference": file_hash(summary_path)},
        "hard_revise_audit": {"path": str(audit_path), "sha256": file_hash(audit_path)},
    }
    # Re-write once with output hashes. The summary records its pre-self-reference
    # hash because a file cannot contain its own final digest.
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
