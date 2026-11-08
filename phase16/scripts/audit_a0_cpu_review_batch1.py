"""Build the first fail-closed P16-R0 A0 CPU review artifact.

This batch examines the first three pending records in frozen A0 position
order.  It never uses published test inputs, published expected values, or
dataset reference solutions to derive a verdict.  The only executable checks
use trusted local oracle/reference/mutant functions.  Untrusted model
execution, full history, cross-history semantic review, and human approval
remain explicit gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "p16_a0_cpu_review_batch_v1"
EXPECTED_INVENTORY_SHA256 = (
    "00743b1561a4c0e632ca2020c91e7a1c891e6d9e75af28b3ff04ad7edc6be377"
)
EXPECTED_BANK_SHA256 = (
    "1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352"
)
FROZEN_POSITIONS = [0, 2, 6]


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def remove_parentheses_oracle(value: str) -> str:
    """Depth-counting implementation derived only from the task contract."""

    depth = 0
    output: list[str] = []
    for character in value:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses")
        elif depth == 0:
            output.append(character)
    if depth != 0:
        raise ValueError("unbalanced parentheses")
    return "".join(output)


def remove_parentheses_reference(value: str) -> str:
    """Independent innermost-group reduction used to cross-check the oracle."""

    reduced = value
    innermost = re.compile(r"\([^()]*\)")
    while "(" in reduced or ")" in reduced:
        next_value = innermost.sub("", reduced)
        if next_value == reduced:
            raise ValueError("unbalanced parentheses")
        reduced = next_value
    return reduced


def boolean_parentheses_mutant(value: str) -> str:
    """Trusted model-equivalent mutant; not execution of the model output."""

    output: list[str] = []
    inside = False
    for character in value:
        if character == "(":
            inside = True
        elif character == ")":
            inside = False
        elif not inside:
            output.append(character)
    return "".join(output)


REMOVE_PARENTHESES_CASES = [
    ("plain", "letters and spaces"),
    ("single_group", "left(remove me)right"),
    ("empty_group", "left()right"),
    ("adjacent_groups", "a(one)(two)b"),
    ("nested_once", "a(b(c)d)e"),
    ("nested_twice", "a(b(c(d)e)f)g"),
    ("all_nested", "((inside))"),
    ("spaced_nested", "keep (drop (all of) this) text"),
]


def five_grams(value: str) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    if not tokens:
        return frozenset()
    if len(tokens) < 5:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[index:index + 5])
                     for index in range(len(tokens) - 4))


def jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _verify_inventory(inventory: dict) -> dict[int, dict]:
    _require(inventory.get("schema_version") ==
             "p16_a0_admission_inventory_v1", "unexpected inventory schema")
    claimed = inventory.get("inventory_sha256")
    without_hash = copy.deepcopy(inventory)
    without_hash.pop("inventory_sha256", None)
    _require(claimed == canonical_sha256(without_hash),
             "inventory canonical hash mismatch")
    _require(claimed == EXPECTED_INVENTORY_SHA256,
             "review points to a different frozen inventory")
    records = inventory.get("records")
    _require(isinstance(records, list), "inventory records must be a list")
    by_position = {row.get("position"): row for row in records}
    _require(len(by_position) == len(records),
             "inventory positions are not unique")
    for position in FROZEN_POSITIONS:
        row = by_position.get(position)
        _require(row is not None, f"missing frozen position {position}")
        _require(row.get("admission_status") ==
                 "pending_independent_oracle_review",
                 f"position {position} is not pending review")
        record_hash = row.get("record_sha256")
        without_record_hash = copy.deepcopy(row)
        without_record_hash.pop("record_sha256", None)
        _require(record_hash == canonical_sha256(without_record_hash),
                 f"record hash mismatch at position {position}")
    return by_position


def _verify_bank(bank: dict) -> dict[str, dict]:
    _require(bank.get("schema_version") == "p16_a0_candidate_bank_v1",
             "unexpected candidate-bank schema")
    claimed = bank.get("bank_sha256")
    without_hash = copy.deepcopy(bank)
    without_hash.pop("bank_sha256", None)
    _require(claimed == canonical_sha256(without_hash),
             "candidate-bank canonical hash mismatch")
    _require(claimed == EXPECTED_BANK_SHA256,
             "review points to a different frozen candidate bank")
    tasks = bank.get("tasks")
    _require(isinstance(tasks, list), "candidate-bank tasks must be a list")
    by_id = {task.get("item_id"): task for task in tasks}
    _require(len(by_id) == len(tasks), "candidate-bank item IDs are not unique")
    return by_id


def _oracle_evidence() -> dict:
    rows = []
    for case_id, value in REMOVE_PARENTHESES_CASES:
        oracle = remove_parentheses_oracle(value)
        reference = remove_parentheses_reference(value)
        mutant = boolean_parentheses_mutant(value)
        _require(oracle == reference,
                 f"independent oracle disagreement on {case_id}")
        rows.append({
            "case_id": case_id,
            "input_sha256": canonical_sha256(value),
            "expected_sha256": canonical_sha256(oracle),
            "model_equivalent_mutant_matches": mutant == oracle,
        })
    caught = [row["case_id"] for row in rows
              if not row["model_equivalent_mutant_matches"]]
    _require("nested_once" in caught and "nested_twice" in caught,
             "nested-parentheses mutant was not detected")
    return {
        "status": "validated_on_trusted_cpu_cases",
        "derivation_boundary": "task_contract_only",
        "published_test_values_used": False,
        "dataset_reference_solutions_used": False,
        "test_count": len(rows),
        "oracle_reference_agree": True,
        "model_equivalent_mutant_caught": True,
        "failing_case_ids": caught,
        "oracle_source_sha256": source_sha256(remove_parentheses_oracle),
        "reference_source_sha256": source_sha256(remove_parentheses_reference),
        "mutant_source_sha256": source_sha256(boolean_parentheses_mutant),
        "cases_sha256": canonical_sha256(REMOVE_PARENTHESES_CASES),
        "transcript_sha256": canonical_sha256(rows),
    }


def _near_duplicate_evidence(bank: dict, target: dict) -> dict:
    target_grams = five_grams(target["question"])
    similarities = []
    for task in bank["tasks"]:
        if task["item_id"] == target["item_id"]:
            continue
        similarities.append({
            "item_id": task["item_id"],
            "prompt_sha256": task["prompt_sha256"],
            "jaccard": jaccard(target_grams, five_grams(task["question"])),
        })
    similarities.sort(key=lambda row: (-row["jaccard"], row["item_id"]))
    top = similarities[:10]
    _require(target.get("near_duplicate_component_size") == 1,
             "target is no longer a singleton frozen component")
    _require(top and top[0]["jaccard"] < 0.80,
             "unexpected bank-local near duplicate")
    return {
        "frozen_component_size": 1,
        "method": "token_5gram_jaccard",
        "threshold": 0.80,
        "maximum_other_similarity": top[0]["jaccard"],
        "top_ten_sha256": canonical_sha256(top),
        "bank_local_automated_status": "passed",
        "manual_cross_history_status": "pending",
    }


def build(inventory: dict, bank: dict) -> dict:
    by_position = _verify_inventory(inventory)
    by_id = _verify_bank(bank)
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for row in selected:
        task = by_id.get(row["item_id"])
        _require(task is not None, f"bank task missing for {row['item_id']}")
        for key in ("family_id", "function_name", "prompt_sha256"):
            _require(task.get(key) == row.get(key),
                     f"bank/inventory {key} mismatch for {row['item_id']}")

    first = selected[0]
    first_task = by_id[first["item_id"]]
    rows = [
        {
            "position": 0,
            "item_id": first["item_id"],
            "family_id": first["family_id"],
            "source_record_sha256": first["record_sha256"],
            "prompt_sha256": first["prompt_sha256"],
            "recovered_output_sha256": first["recovered_output_sha256"],
            "contract_review": {
                "status": "passed",
                "reason": "balanced nested parentheses have a deterministic depth semantics",
            },
            "independent_oracle": _oracle_evidence(),
            "history_review": {
                "frozen_phase_1_15_exact_status": "passed",
                "full_semantic_history_status": "pending",
            },
            "near_duplicate_review": _near_duplicate_evidence(bank, first_task),
            "sandbox_review": "pending_no_local_untrusted_code_sandbox",
            "human_approval": "pending",
            "review_disposition": "oracle_ready_remaining_gates_pending",
            "correction_admitted": False,
        },
        {
            "position": 2,
            "item_id": selected[1]["item_id"],
            "family_id": selected[1]["family_id"],
            "source_record_sha256": selected[1]["record_sha256"],
            "prompt_sha256": selected[1]["prompt_sha256"],
            "recovered_output_sha256": selected[1]["recovered_output_sha256"],
            "contract_review": {
                "status": "failed",
                "reason": "scheduling and resource-overlap semantics are delegated to unavailable example tests",
            },
            "independent_oracle": {"status": "not_constructed_contract_failed"},
            "history_review": "not_reviewed_ineligible",
            "near_duplicate_review": "not_reviewed_ineligible",
            "sandbox_review": "not_reviewed_ineligible",
            "human_approval": "not_applicable_rejected",
            "review_disposition": "rejected_contract_under_specified",
            "rejection_reasons": ["contract_under_specified_without_published_examples"],
            "correction_admitted": False,
        },
        {
            "position": 6,
            "item_id": selected[2]["item_id"],
            "family_id": selected[2]["family_id"],
            "source_record_sha256": selected[2]["record_sha256"],
            "prompt_sha256": selected[2]["prompt_sha256"],
            "recovered_output_sha256": selected[2]["recovered_output_sha256"],
            "contract_review": {
                "status": "passed",
                "reason": "semitone bounds and tab-alignment behavior are stated but require a dedicated grid oracle",
            },
            "independent_oracle": {"status": "pending_specialized_tab_grid_oracle"},
            "history_review": "pending",
            "near_duplicate_review": "pending",
            "sandbox_review": "pending",
            "human_approval": "pending",
            "review_disposition": "deferred_in_frozen_order",
            "correction_admitted": False,
        },
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "bank_sha256": bank["bank_sha256"],
        },
        "frozen_review_rule": "first three pending records in A0 position order",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "examined": 3,
            "contract_passed": 2,
            "contract_rejected": 1,
            "trusted_oracle_validated": 1,
            "deferred_at_oracle": 1,
            "correction_admitted": 0,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position_blocked": 6,
    }
    document["review_sha256"] = canonical_sha256(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    document = build(inventory, bank)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False)
                           + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(document["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
