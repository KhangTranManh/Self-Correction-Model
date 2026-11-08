"""Continue P16-R0 CPU review at frozen positions 30, 31 and 36."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from audit_a0_cpu_review_batch1 import (
    _near_duplicate_evidence,
    _require,
    _verify_bank,
    _verify_inventory,
    canonical_sha256,
)
from audit_a0_cpu_review_batch2 import _evidence


SCHEMA_VERSION = "p16_a0_cpu_review_batch_v1"
EXPECTED_BATCH4_REVIEW_SHA256 = (
    "6dc9b86182dfd3f2191ed256a4cb1f8b4afe0566029772627813db77f5d96618"
)
FROZEN_POSITIONS = [30, 31, 36]
VOWELS = frozenset("aeiouAEIOU")


def ka_cipher_oracle(word: str) -> str:
    output = ["ka"]
    index = 0
    while index < len(word):
        if word[index] not in VOWELS:
            output.append(word[index])
            index += 1
            continue
        end = index + 1
        while end < len(word) and word[end] in VOWELS:
            end += 1
        output.append(word[index:end])
        if end < len(word):
            output.append("ka")
        index = end
    return "".join(output)


def ka_cipher_reference(word: str) -> str:
    encoded = re.sub(
        r"([aeiou]+)(?=[^aeiou])",
        lambda match: match.group(1) + "ka",
        word,
        flags=re.IGNORECASE,
    )
    return "ka" + encoded


def ka_cipher_mutant(word: str) -> str:
    result = "ka"
    previous_was_vowel = False
    for character in word:
        if character.lower() in "aeiou":
            if not previous_was_vowel:
                result += "ka"
            previous_was_vowel = True
        else:
            previous_was_vowel = False
        result += character.lower()
    return result


KA_CASES = [
    ("single_vowel", "a"),
    ("consonant_then_final_vowel", "ka"),
    ("vowel_cluster", "aa"),
    ("uppercase_and_final_cluster", "Abbaa"),
    ("internal_clusters", "maintenance"),
    ("mixed_case_clusters", "Woodie"),
    ("no_vowels", "rhythm"),
    ("alternating", "banana"),
]


def fishex_oracle(name: str) -> int:
    result = 0
    for character in name.lower():
        if character in "abcdef":
            result ^= int(character, 16)
    return result


def fishex_reference(name: str) -> int:
    result = 0
    lowered = name.lower()
    values = {character: value for value, character in enumerate(
        "abcdef", start=10)}
    for character, value in values.items():
        if lowered.count(character) % 2 == 1:
            result ^= value
    return result


def fishex_mutant(name: str) -> int:
    result = 0
    for character in name:
        if character in "abcdefABCDEF":
            result ^= ord(character)
    return result


FISHEX_CASES = [
    ("no_hex_letters", "GHI XYZ"),
    ("single_lower", "a"),
    ("single_upper", "F"),
    ("contract_example", "redlionfish"),
    ("paired_cancellation", "cafe face"),
    ("mixed_case", "Bad Chef"),
    ("repeated_odd_counts", "feed a cab"),
]


def _verify_batch4(batch4: dict) -> None:
    _require(batch4.get("schema_version") == SCHEMA_VERSION,
             "unexpected batch-4 schema")
    claimed = batch4.get("review_sha256")
    without_hash = copy.deepcopy(batch4)
    without_hash.pop("review_sha256", None)
    _require(claimed == canonical_sha256(without_hash),
             "batch-4 canonical hash mismatch")
    _require(claimed == EXPECTED_BATCH4_REVIEW_SHA256,
             "batch 5 points to a different batch 4")
    _require(batch4.get("next_position") == 30,
             "batch 4 did not stop before position 30")


def _base_row(record: dict) -> dict:
    return {
        "position": record["position"],
        "item_id": record["item_id"],
        "family_id": record["family_id"],
        "source_record_sha256": record["record_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "recovered_output_sha256": record["recovered_output_sha256"],
    }


def build(inventory: dict, bank: dict, batch4: dict) -> dict:
    _verify_batch4(batch4)
    by_position = _verify_inventory(inventory)
    by_id = _verify_bank(bank)
    pending_after_batch4 = [
        row["position"] for row in inventory["records"]
        if row["position"] >= batch4["next_position"]
        and row["admission_status"] == "pending_independent_oracle_review"
    ]
    _require(pending_after_batch4[:3] == FROZEN_POSITIONS,
             "batch 5 positions are not the next three pending records")
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for row in selected:
        task = by_id.get(row["item_id"])
        _require(task is not None, f"bank task missing for {row['item_id']}")
        for key in ("family_id", "function_name", "prompt_sha256"):
            _require(task.get(key) == row.get(key),
                     f"bank/inventory {key} mismatch for {row['item_id']}")

    rows = []
    passing_definitions = [
        (selected[0], KA_CASES, ka_cipher_oracle, ka_cipher_reference,
         ka_cipher_mutant,
         "prefix, vowel-run boundaries, terminal vowels and case preservation are explicit"),
        (selected[1], FISHEX_CASES, fishex_oracle, fishex_reference,
         fishex_mutant,
         "case-insensitive hexadecimal values and XOR reduction are explicit"),
    ]
    for record, cases, oracle, reference, mutant, reason in passing_definitions:
        task = by_id[record["item_id"]]
        row = _base_row(record)
        row.update({
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
        rows.append(row)

    rejected = _base_row(selected[2])
    rejected.update({
        "contract_review": {
            "status": "failed",
            "reason": (
                "first-letter sorting leaves distinct-word tie order undefined; "
                "punctuation removal also does not define token joining"
            ),
        },
        "independent_oracle": {"status": "not_constructed_contract_failed"},
        "history_review": "not_reviewed_ineligible",
        "near_duplicate_review": "not_reviewed_ineligible",
        "sandbox_review": "not_reviewed_ineligible",
        "human_approval": "not_applicable_rejected",
        "review_disposition": "rejected_contract_under_specified",
        "rejection_reasons": [
            "contract_under_specified_tie_and_punctuation_semantics",
        ],
        "correction_admitted": False,
    })
    rows.append(rejected)

    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "bank_sha256": bank["bank_sha256"],
            "previous_review_sha256": batch4["review_sha256"],
        },
        "frozen_review_rule": "review pending positions 30, 31 and 36",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 3,
            "contract_passed": 2,
            "contract_rejected": 1,
            "trusted_oracle_validated": 2,
            "correction_admitted": 0,
            "cumulative_positions_examined": 14,
            "cumulative_contract_rejected": 3,
            "cumulative_oracle_ready": 11,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position": 37,
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
