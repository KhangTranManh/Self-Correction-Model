"""Continue P16-R0 CPU review at frozen positions 16, 21 and 22."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from audit_a0_cpu_review_batch1 import (
    _near_duplicate_evidence,
    _require,
    _verify_bank,
    _verify_inventory,
    canonical_sha256,
)
from audit_a0_cpu_review_batch2 import _evidence


SCHEMA_VERSION = "p16_a0_cpu_review_batch_v1"
EXPECTED_BATCH2_REVIEW_SHA256 = (
    "2369e0d44716eac286fc13327985c1bdb66ad3a5fdeda79d042af813191b48ca"
)
FROZEN_POSITIONS = [16, 21, 22]


KEBAB = re.compile(r"[a-z]+(?:-[a-z]+)+\Z")
SNAKE = re.compile(r"[a-z]+(?:_[a-z]+)+\Z")
CAMEL = re.compile(r"[a-z]+(?:[A-Z][a-z]*)+\Z")


def case_id_oracle(value: str) -> str:
    if KEBAB.fullmatch(value):
        return "kebab"
    if SNAKE.fullmatch(value):
        return "snake"
    if CAMEL.fullmatch(value):
        return "camel"
    return "none"


def case_id_reference(value: str) -> str:
    if "-" in value:
        parts = value.split("-")
        return "kebab" if (len(parts) > 1 and all(
            part and part.isalpha() and part.islower() for part in parts
        )) else "none"
    if "_" in value:
        parts = value.split("_")
        return "snake" if (len(parts) > 1 and all(
            part and part.isalpha() and part.islower() for part in parts
        )) else "none"
    if not value or not value[0].islower():
        return "none"
    cursor = 0
    while cursor < len(value) and value[cursor].islower():
        cursor += 1
    saw_capitalized_word = False
    while cursor < len(value):
        if not value[cursor].isupper():
            return "none"
        saw_capitalized_word = True
        cursor += 1
        while cursor < len(value) and value[cursor].islower():
            cursor += 1
    return "camel" if saw_capitalized_word else "none"


def case_id_mutant(value: str) -> str:
    if "-" in value:
        return "kebab"
    if "_" in value:
        return "snake"
    if value[0].islower() and value[1:].isidentifier():
        words = value.split("_")
        if len(words) == 1 and words[0].islower():
            return "snake"
        if (len(words) > 1 and words[0].islower()
                and all(word.istitle() or word.islower()
                        for word in words[1:])):
            return "camel"
    return "none"


CASE_ID_CASES = [
    ("kebab_two", "hello-world"),
    ("kebab_many", "hello-to-the-world"),
    ("kebab_short", "a-b"),
    ("snake_two", "hello_world"),
    ("snake_many", "hello_to_the_world"),
    ("camel_two", "helloWorld"),
    ("camel_many", "helloToTheWorld"),
    ("camel_short", "aB"),
    ("none_plain", "helloworld"),
    ("none_double_separator", "hello__world"),
    ("none_mixed_case_snake", "hello_World"),
    ("none_mixed_case_kebab", "hello-World"),
    ("none_upper_initial", "HelloWorld"),
    ("none_leading_separator", "-hello-world"),
    ("none_trailing_separator", "hello-world-"),
    ("none_mixed_separators", "hello_world-more"),
]


def _parse_couple(value: str) -> tuple[int, int]:
    left, right = value.split("-", 1)
    return int(left), int(right)


def difference_oracle(couples: list[str]) -> str | bool:
    best_value: str | bool = False
    best_difference = 0
    for couple in couples:
        left, right = _parse_couple(couple)
        current = abs(left - right)
        if current > best_difference:
            best_difference = current
            best_value = couple
    return best_value


def difference_reference(couples: list[str]) -> str | bool:
    differences = [abs(left - right)
                   for left, right in map(_parse_couple, couples)]
    maximum = max(differences, default=0)
    if maximum == 0:
        return False
    return couples[differences.index(maximum)]


def difference_mutant(couples: list[str]) -> str | bool:
    maximum = -1
    result: str | bool = False
    for couple in couples:
        left, right = _parse_couple(couple)
        current = abs(left - right)
        if current > maximum:
            maximum = current
            result = couple
    return result


DIFFERENCE_CASES = [
    ("contract_example", ["56-23", "1-100"]),
    ("first_tie", ["1-3", "5-7", "2-3"]),
    ("all_equal", ["11-11", "344-344"]),
    ("single_equal", ["7-7"]),
    ("single_difference", ["7-2"]),
    ("reversed_numbers", ["100-1", "2-90"]),
    ("later_larger", ["1000-999", "1-500", "2-2"]),
]


def _verify_batch2(batch2: dict) -> None:
    _require(batch2.get("schema_version") == SCHEMA_VERSION,
             "unexpected batch-2 schema")
    claimed = batch2.get("review_sha256")
    without_hash = copy.deepcopy(batch2)
    without_hash.pop("review_sha256", None)
    _require(claimed == canonical_sha256(without_hash),
             "batch-2 canonical hash mismatch")
    _require(claimed == EXPECTED_BATCH2_REVIEW_SHA256,
             "batch 3 points to a different batch 2")
    _require(batch2.get("next_position") == 16,
             "batch 2 did not stop before position 16")


def build(inventory: dict, bank: dict, batch2: dict) -> dict:
    _verify_batch2(batch2)
    by_position = _verify_inventory(inventory)
    by_id = _verify_bank(bank)
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for row in selected:
        task = by_id.get(row["item_id"])
        _require(task is not None, f"bank task missing for {row['item_id']}")
        for key in ("family_id", "function_name", "prompt_sha256"):
            _require(task.get(key) == row.get(key),
                     f"bank/inventory {key} mismatch for {row['item_id']}")

    rejected = selected[0]
    rows = [{
        "position": rejected["position"],
        "item_id": rejected["item_id"],
        "family_id": rejected["family_id"],
        "source_record_sha256": rejected["record_sha256"],
        "prompt_sha256": rejected["prompt_sha256"],
        "recovered_output_sha256": rejected["recovered_output_sha256"],
        "contract_review": {
            "status": "failed",
            "reason": (
                "word reversal does not uniquely specify placement of unequal "
                "leading, trailing and inter-word space runs"
            ),
        },
        "independent_oracle": {"status": "not_constructed_contract_failed"},
        "history_review": "not_reviewed_ineligible",
        "near_duplicate_review": "not_reviewed_ineligible",
        "sandbox_review": "not_reviewed_ineligible",
        "human_approval": "not_applicable_rejected",
        "review_disposition": "rejected_contract_under_specified",
        "rejection_reasons": ["contract_under_specified_whitespace_semantics"],
        "correction_admitted": False,
    }]

    definitions = [
        (selected[1], CASE_ID_CASES, case_id_oracle,
         case_id_reference, case_id_mutant,
         "kebab, snake, camel and none grammars are explicitly defined"),
        (selected[2], DIFFERENCE_CASES, difference_oracle,
         difference_reference, difference_mutant,
         "absolute difference, first-tie behavior and all-zero result are explicit"),
    ]
    for record, cases, oracle, reference, mutant, reason in definitions:
        task = by_id[record["item_id"]]
        rows.append({
            "position": record["position"],
            "item_id": record["item_id"],
            "family_id": record["family_id"],
            "source_record_sha256": record["record_sha256"],
            "prompt_sha256": record["prompt_sha256"],
            "recovered_output_sha256": record["recovered_output_sha256"],
            "contract_review": {"status": "passed", "reason": reason},
            "independent_oracle": _evidence(
                cases, oracle, reference, mutant),
            "history_review": {
                "frozen_phase_1_15_exact_status": "passed",
                "full_semantic_history_status": "pending",
            },
            "near_duplicate_review": _near_duplicate_evidence(bank, task),
            "sandbox_review": "pending_no_local_untrusted_code_sandbox",
            "human_approval": "pending",
            "review_disposition": "oracle_ready_remaining_gates_pending",
            "correction_admitted": False,
        })

    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "bank_sha256": bank["bank_sha256"],
            "previous_review_sha256": batch2["review_sha256"],
        },
        "frozen_review_rule": "review pending positions 16, 21 and 22",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 3,
            "contract_passed": 2,
            "contract_rejected": 1,
            "trusted_oracle_validated": 2,
            "correction_admitted": 0,
            "cumulative_positions_examined": 8,
            "cumulative_contract_rejected": 2,
            "cumulative_oracle_ready": 6,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position": 23,
    }
    document["review_sha256"] = canonical_sha256(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--previous-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    document = build(
        json.loads(args.inventory.read_text(encoding="utf-8")),
        json.loads(args.bank.read_text(encoding="utf-8")),
        json.loads(args.previous_review.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False)
                           + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(document["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
