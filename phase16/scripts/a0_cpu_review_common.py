"""Shared fail-closed assembly for Phase 16 A0 CPU review batches."""

from __future__ import annotations

from audit_a0_cpu_review_batch6 import (
    EXPECTED_BANK_SHA256, EXPECTED_INVENTORY_SHA256, SCHEMA_VERSION,
    _base_row, _evidence, _near_duplicate_evidence, _require,
    _verify_hashed, canonical_sha256,
)


def build_review_batch(
        inventory: dict, bank: dict, previous: dict, *, batch_number: int,
        positions: list[int], functions: list[str], definitions: dict,
        rejections: dict[int, tuple[str, str]], cumulative_examined: int,
        cumulative_ready: int, cumulative_rejected: int) -> dict:
    label = f"batch {batch_number}"
    previous_label = f"batch {batch_number - 1}"
    _verify_hashed(inventory, "inventory_sha256",
                   "p16_a0_admission_inventory_v1", "inventory")
    _require(inventory["inventory_sha256"] == EXPECTED_INVENTORY_SHA256,
             f"{label} points to a different frozen inventory")
    _verify_hashed(bank, "bank_sha256", "p16_a0_candidate_bank_v1", "bank")
    _require(bank["bank_sha256"] == EXPECTED_BANK_SHA256,
             f"{label} points to a different frozen bank")
    _verify_hashed(previous, "review_sha256", SCHEMA_VERSION, previous_label)
    _require(previous.get("next_position") == positions[0],
             f"{previous_label} does not hand off at position {positions[0]}")
    _require(len(positions) == len(functions), "position/function length mismatch")
    _require(set(definitions) | set(rejections) == set(positions),
             "every frozen position must have exactly one disposition")
    _require(not (set(definitions) & set(rejections)),
             "a position cannot pass and reject")
    by_position = {row["position"]: row for row in inventory["records"]}
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    pending = [row["position"] for row in inventory["records"]
               if row["position"] >= positions[0]
               and row["admission_status"] == "pending_independent_oracle_review"]
    _require(pending[:len(positions)] == positions,
             f"{label} positions are not the next pending records")
    rows = []
    for position, function_name in zip(positions, functions):
        record = by_position[position]
        task = by_id[record["item_id"]]
        _require(task["function_name"] == function_name,
                 f"unexpected function at position {position}")
        for key in ("family_id", "prompt_sha256"):
            _require(task[key] == record[key],
                     f"bank/inventory {key} mismatch at position {position}")
        row = _base_row(record)
        if position in definitions:
            _require(record["source_history_exact_clean"] is True,
                     f"exact history not clean at position {position}")
            cases, oracle, reference, mutant, reason = definitions[position]
            row.update({
                "contract_review": {"status": "passed", "reason": reason},
                "independent_oracle": _evidence(cases, oracle, reference, mutant),
                "history_review": {"frozen_phase_1_15_exact_status": "passed",
                                   "full_semantic_history_status": "pending"},
                "near_duplicate_review": _near_duplicate_evidence(bank, task),
                "sandbox_review": "pending_cpu_sandbox_closure",
                "human_approval": "pending",
                "review_disposition": "oracle_ready_remaining_gates_pending",
                "correction_admitted": False,
            })
        else:
            reason, code = rejections[position]
            row.update({
                "contract_review": {"status": "failed", "reason": reason},
                "independent_oracle": {"status": "not_constructed_contract_failed"},
                "history_review": "not_reviewed_ineligible",
                "near_duplicate_review": "not_reviewed_ineligible",
                "sandbox_review": "not_reviewed_ineligible",
                "human_approval": "not_applicable_rejected",
                "review_disposition": "rejected_contract_under_specified",
                "rejection_reasons": [code], "correction_admitted": False,
            })
        rows.append(row)
    passed = len(definitions)
    rejected = len(rejections)
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {"inventory_sha256": inventory["inventory_sha256"],
                   "bank_sha256": bank["bank_sha256"],
                   "previous_review_sha256": previous["review_sha256"]},
        "frozen_review_rule":
            f"review next {len(positions)} pending positions from {positions[0]}",
        "positions": positions,
        "counts": {"newly_examined": len(positions), "contract_passed": passed,
                   "contract_rejected": rejected,
                   "trusted_oracle_validated": passed, "correction_admitted": 0,
                   "cumulative_positions_examined": cumulative_examined,
                   "cumulative_contract_rejected": cumulative_rejected,
                   "cumulative_oracle_ready": cumulative_ready},
        "rows": rows,
        "claims": {"untrusted_model_code_executed": False,
                   "independent_oracle_admission_complete": False,
                   "correction_pool_certified": False,
                   "gpu_training_authorized": False,
                   "protected_data_opening_authorized": False},
        "next_position": pending[len(positions)] if len(pending) > len(positions)
                         else None,
    }
    document["review_sha256"] = canonical_sha256(document)
    return document
