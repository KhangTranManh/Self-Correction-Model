"""Validate Phase 16 recovery-first CPU artifacts without running a model.

The validators are deliberately fail-closed. They verify schemas, hashes,
cross-record invariants, and recorded deterministic-verifier evidence. They do
not execute candidate code or promote an LLM judgment to an oracle verdict.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


HEX = set("0123456789abcdef")
FAILURE_CLASSES = {
    "presentation_only",
    "recoverable_payload",
    "substantive_content",
    "execution_or_scaffold",
    "ambiguous_or_unscorable",
}
POOL_COUNTS = {
    "correction": 64,
    "positive_probe": 80,
    "near_negative": 20,
    "unrelated_control": 40,
}
LEVELS = (8, 16, 32, 64)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: Any, keys: set[str], prefix: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{prefix} must be an object")
    missing = keys - set(value)
    extra = set(value) - keys
    _require(not missing and not extra,
             f"{prefix} keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _text(value: Any, prefix: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str) and (allow_empty or bool(value)),
             f"{prefix} must be a {'string' if allow_empty else 'nonempty string'}")
    return value


def _boolean(value: Any, prefix: str) -> bool:
    _require(type(value) is bool, f"{prefix} must be a Boolean")
    return value


def _integer(value: Any, prefix: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum,
             f"{prefix} must be an integer >= {minimum}")
    return value


def _digest(value: Any, prefix: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and set(value) <= HEX,
             f"{prefix} must be a lowercase SHA-256 digest")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    return sha256_text(rendered)


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    return result


def _verify_record_hash(record: dict[str, Any], key: str, prefix: str) -> None:
    supplied = _digest(record[key], f"{prefix}.{key}")
    expected = canonical_sha256(_without(record, key))
    _require(supplied == expected, f"{prefix}.{key} mismatch")


def _unique(values: list[str], prefix: str) -> None:
    _require(len(values) == len(set(values)), f"{prefix} must be unique")


def validate_recovery_batch(document: dict[str, Any]) -> dict[str, Any]:
    """Validate frozen recovery-wrapper evidence and outcome accounting."""

    _exact_keys(document, {"schema_version", "batch_id", "wrapper", "records",
                           "batch_sha256"}, "recovery batch")
    _require(document["schema_version"] == "p16_recovery_batch_v1",
             "unexpected recovery schema_version")
    _text(document["batch_id"], "batch_id")
    wrapper = _exact_keys(document["wrapper"], {
        "wrapper_id", "version", "family_id", "rule_text", "rule_sha256",
        "frozen", "reference_free",
    }, "wrapper")
    for key in ("wrapper_id", "version", "family_id", "rule_text"):
        _text(wrapper[key], f"wrapper.{key}")
    _require(_digest(wrapper["rule_sha256"], "wrapper.rule_sha256") ==
             sha256_text(wrapper["rule_text"]), "wrapper.rule_sha256 mismatch")
    _require(_boolean(wrapper["frozen"], "wrapper.frozen"), "wrapper must be frozen")
    _require(_boolean(wrapper["reference_free"], "wrapper.reference_free"),
             "wrapper must be reference-free")

    records = document["records"]
    _require(isinstance(records, list) and records, "records must be a nonempty list")
    ids: list[str] = []
    task_ids: list[str] = []
    outcomes: Counter[str] = Counter()
    fixture_roles: Counter[str] = Counter()
    for index, raw_record in enumerate(records):
        prefix = f"records[{index}]"
        record = _exact_keys(raw_record, {
            "record_id", "task_id", "family_id", "source_id", "split",
            "fixture_role", "pre_intervention_class", "raw_output",
            "recovered_output", "recovery_status", "rejection_reason",
            "verification", "record_sha256",
        }, prefix)
        for key in ("record_id", "task_id", "family_id", "source_id"):
            _text(record[key], f"{prefix}.{key}")
        ids.append(record["record_id"])
        task_ids.append(record["task_id"])
        _require(record["family_id"] == wrapper["family_id"],
                 f"{prefix}.family_id differs from wrapper")
        _require(record["split"] in {"engineering_fixture", "development"},
                 f"{prefix}.split is not CPU-safe")
        _require(record["fixture_role"] in {
            "payload_recovery", "already_valid", "rejection", "residual_content"
        }, f"{prefix}.fixture_role unsupported")
        fixture_roles[record["fixture_role"]] += 1
        _require(record["pre_intervention_class"] in FAILURE_CLASSES,
                 f"{prefix}.pre_intervention_class unsupported")
        raw_output = _text(record["raw_output"], f"{prefix}.raw_output", allow_empty=True)
        recovered = _text(record["recovered_output"],
                          f"{prefix}.recovered_output", allow_empty=True)
        status = record["recovery_status"]
        _require(status in {"accepted", "unchanged", "rejected"},
                 f"{prefix}.recovery_status unsupported")
        reason = _text(record["rejection_reason"], f"{prefix}.rejection_reason",
                       allow_empty=True)
        if status == "accepted":
            _require(raw_output != recovered, f"{prefix}: accepted recovery made no change")
            _require(not reason, f"{prefix}: accepted recovery has rejection_reason")
        elif status == "unchanged":
            _require(raw_output == recovered, f"{prefix}: unchanged output differs")
            _require(not reason, f"{prefix}: unchanged recovery has rejection_reason")
        else:
            _require(raw_output == recovered,
                     f"{prefix}: rejected recovery must preserve endpoint output")
            _require(bool(reason), f"{prefix}: rejected recovery needs rejection_reason")

        verification = _exact_keys(record["verification"], {
            "oracle_sha256", "raw_output_sha256", "recovered_output_sha256",
            "raw_correct", "recovered_correct", "payload_preserved", "idempotent",
            "deterministic_rerun", "noninterference_on_valid",
            "reference_answer_accessed", "hidden_tests_accessed",
            "other_arm_accessed",
        }, f"{prefix}.verification")
        _digest(verification["oracle_sha256"], f"{prefix}.verification.oracle_sha256")
        _require(_digest(verification["raw_output_sha256"],
                         f"{prefix}.verification.raw_output_sha256") ==
                 sha256_text(raw_output), f"{prefix}: raw output hash mismatch")
        _require(_digest(verification["recovered_output_sha256"],
                         f"{prefix}.verification.recovered_output_sha256") ==
                 sha256_text(recovered), f"{prefix}: recovered output hash mismatch")
        raw_correct = _boolean(verification["raw_correct"],
                               f"{prefix}.verification.raw_correct")
        recovered_correct = _boolean(verification["recovered_correct"],
                                     f"{prefix}.verification.recovered_correct")
        for key in ("payload_preserved", "idempotent", "deterministic_rerun"):
            _require(_boolean(verification[key], f"{prefix}.verification.{key}"),
                     f"{prefix}: {key} must pass")
        noninterference = _boolean(verification["noninterference_on_valid"],
                                   f"{prefix}.verification.noninterference_on_valid")
        for key in ("reference_answer_accessed", "hidden_tests_accessed",
                    "other_arm_accessed"):
            _require(not _boolean(verification[key], f"{prefix}.verification.{key}"),
                     f"{prefix}: forbidden access recorded in {key}")

        if record["fixture_role"] == "already_valid":
            _require(status == "unchanged" and raw_correct and recovered_correct and
                     noninterference,
                     f"{prefix}: already-valid fixture must be byte-preserved and correct")
        if record["fixture_role"] == "payload_recovery":
            _require(status == "accepted" and not raw_correct and recovered_correct,
                     f"{prefix}: payload-recovery fixture must be fixed")
        if record["fixture_role"] == "rejection":
            _require(status == "rejected", f"{prefix}: rejection fixture must reject")
        if record["fixture_role"] == "residual_content":
            _require(record["pre_intervention_class"] == "substantive_content" and
                     not recovered_correct,
                     f"{prefix}: residual-content fixture is not a verified residual")
        if record["pre_intervention_class"] == "substantive_content":
            _require(not recovered_correct,
                     f"{prefix}: substantive_content cannot be recovery-fixed")

        if not raw_correct and recovered_correct:
            outcome = "fixed"
        elif raw_correct and not recovered_correct:
            outcome = "harmed"
        elif recovered_correct:
            outcome = "unchanged_correct"
        else:
            outcome = "still_wrong"
        outcomes[outcome] += 1
        _verify_record_hash(record, "record_sha256", prefix)

    _unique(ids, "recovery record ids")
    _unique(task_ids, "recovery task ids")
    for required in ("payload_recovery", "already_valid", "rejection"):
        _require(fixture_roles[required] > 0,
                 f"recovery batch lacks required {required} fixture")
    _require(outcomes["harmed"] == 0, "recovery batch contains correctness harm")
    _require(_digest(document["batch_sha256"], "batch_sha256") ==
             canonical_sha256(_without(document, "batch_sha256")),
             "batch_sha256 mismatch")
    return {
        "schema_version": "p16_recovery_batch_validation_v1",
        "valid": True,
        "record_count": len(records),
        "fixture_role_counts": {
            key: fixture_roles[key] for key in
            ("payload_recovery", "already_valid", "rejection", "residual_content")
        },
        "outcome_counts": {
            key: outcomes[key] for key in
            ("fixed", "harmed", "unchanged_correct", "still_wrong")
        },
        "wrapper_sha256": wrapper["rule_sha256"],
        "limitations": "Recorded oracle evidence only; no output or code was executed.",
    }


def _validate_example(example: Any, prefix: str) -> dict[str, Any]:
    row = _exact_keys(example, {"example_id", "task_id", "oracle_sha256",
                                "verifier_correct"}, prefix)
    _text(row["example_id"], f"{prefix}.example_id")
    _text(row["task_id"], f"{prefix}.task_id")
    _digest(row["oracle_sha256"], f"{prefix}.oracle_sha256")
    _require(_boolean(row["verifier_correct"], f"{prefix}.verifier_correct"),
             f"{prefix}: verifier evidence must pass")
    return row


def render_prompt_patch(records_by_id: dict[str, dict[str, Any]],
                        retrieved_rule_ids: list[str]) -> str:
    blocks = []
    for record_id in retrieved_rule_ids:
        record = records_by_id[record_id]
        blocks.append(
            f"[rule:{record_id}@{record['version']}]\n"
            f"Trigger: {record['applicability_trigger']}\n"
            f"Boundary: {record['non_applicability_boundary']}\n"
            f"Rule: {record['rule']}\n"
            f"Action: {record['permitted_action']}"
        )
    return "\n\n".join(blocks)


def harness_memory_sha256(document: dict[str, Any]) -> str:
    """Hash only immutable memory/config, never append-only inference logs."""

    payload = {
        key: copy.deepcopy(document[key]) for key in (
            "schema_version", "memory_id", "version", "frozen", "top_k",
            "token_budget", "token_count_method", "records",
        )
    }
    return canonical_sha256(payload)


def validate_harness_memory(document: dict[str, Any]) -> dict[str, Any]:
    """Validate immutable harness memory and deterministic prompt patches."""

    _exact_keys(document, {"schema_version", "memory_id", "version", "frozen",
                           "top_k", "token_budget", "token_count_method",
                           "records", "runs", "memory_sha256"}, "harness memory")
    _require(document["schema_version"] == "p16_harness_memory_v1",
             "unexpected harness schema_version")
    for key in ("memory_id", "version"):
        _text(document[key], key)
    _require(_boolean(document["frozen"], "frozen"), "harness memory must be frozen")
    top_k = _integer(document["top_k"], "top_k", minimum=1)
    token_budget = _integer(document["token_budget"], "token_budget", minimum=1)
    _require(document["token_count_method"] == "whitespace_v1",
             "token_count_method must be whitespace_v1")
    records = document["records"]
    _require(isinstance(records, list) and records, "harness records must be nonempty")
    records_by_id: dict[str, dict[str, Any]] = {}
    correction_ids: list[str] = []
    for index, raw_record in enumerate(records):
        prefix = f"records[{index}]"
        record = _exact_keys(raw_record, {
            "record_id", "version", "error_family_id", "correction_record_id",
            "applicability_trigger", "non_applicability_boundary", "rule",
            "permitted_action", "positive_examples", "near_negative_examples",
            "provenance", "schema_validated", "leakage_clear", "human_approved",
            "record_sha256",
        }, prefix)
        for key in ("record_id", "version", "error_family_id", "correction_record_id",
                    "applicability_trigger", "non_applicability_boundary", "rule"):
            _text(record[key], f"{prefix}.{key}")
        _require(record["record_id"] not in records_by_id,
                 f"{prefix}: duplicate record_id")
        records_by_id[record["record_id"]] = record
        correction_ids.append(record["correction_record_id"])
        _require(record["permitted_action"] in {
            "prompt_rule", "retrieval_hint", "output_contract_reminder"
        }, f"{prefix}.permitted_action unsupported")
        for key in ("schema_validated", "leakage_clear", "human_approved"):
            _require(_boolean(record[key], f"{prefix}.{key}"),
                     f"{prefix}: {key} must pass")
        positive = record["positive_examples"]
        near = record["near_negative_examples"]
        _require(isinstance(positive, list) and positive,
                 f"{prefix}.positive_examples must be nonempty")
        _require(isinstance(near, list) and near,
                 f"{prefix}.near_negative_examples must be nonempty")
        pos_rows = [_validate_example(row, f"{prefix}.positive_examples[{i}]")
                    for i, row in enumerate(positive)]
        near_rows = [_validate_example(row, f"{prefix}.near_negative_examples[{i}]")
                     for i, row in enumerate(near)]
        example_ids = [row["example_id"] for row in pos_rows + near_rows]
        task_ids = [row["task_id"] for row in pos_rows + near_rows]
        _unique(example_ids, f"{prefix} example ids")
        _unique(task_ids, f"{prefix} example task ids")
        provenance = _exact_keys(record["provenance"], {
            "source_id", "source_revision", "reviewer_id", "oracle_sha256"
        }, f"{prefix}.provenance")
        for key in ("source_id", "source_revision", "reviewer_id"):
            _text(provenance[key], f"{prefix}.provenance.{key}")
        _digest(provenance["oracle_sha256"], f"{prefix}.provenance.oracle_sha256")
        _verify_record_hash(record, "record_sha256", prefix)
    _unique(correction_ids, "harness correction_record_ids")

    runs = document["runs"]
    _require(isinstance(runs, list), "runs must be a list")
    run_ids: list[str] = []
    run_tasks: list[str] = []
    for index, raw_run in enumerate(runs):
        prefix = f"runs[{index}]"
        run = _exact_keys(raw_run, {
            "run_id", "task_id", "family_id", "retrieved_rule_ids", "prompt_patch",
            "prompt_patch_sha256", "retrieved_token_count", "current_verdict_accessed",
            "hidden_tests_accessed", "probe_answer_accessed", "later_memory_accessed",
            "run_sha256",
        }, prefix)
        for key in ("run_id", "task_id", "family_id"):
            _text(run[key], f"{prefix}.{key}")
        run_ids.append(run["run_id"])
        run_tasks.append(run["task_id"])
        retrieved = run["retrieved_rule_ids"]
        _require(isinstance(retrieved, list) and
                 all(isinstance(value, str) and value for value in retrieved),
                 f"{prefix}.retrieved_rule_ids must be strings")
        _unique(retrieved, f"{prefix}.retrieved_rule_ids")
        _require(len(retrieved) <= top_k, f"{prefix}: retrieval exceeds top_k")
        _require(set(retrieved) <= set(records_by_id),
                 f"{prefix}: retrieval references unknown rule")
        expected_patch = render_prompt_patch(records_by_id, retrieved)
        _require(_text(run["prompt_patch"], f"{prefix}.prompt_patch", allow_empty=True) ==
                 expected_patch, f"{prefix}: prompt_patch is not deterministic")
        _require(_digest(run["prompt_patch_sha256"],
                         f"{prefix}.prompt_patch_sha256") ==
                 sha256_text(expected_patch), f"{prefix}: prompt patch hash mismatch")
        expected_tokens = len(expected_patch.split())
        _require(_integer(run["retrieved_token_count"],
                          f"{prefix}.retrieved_token_count") == expected_tokens,
                 f"{prefix}: retrieved token count mismatch")
        _require(expected_tokens <= token_budget,
                 f"{prefix}: retrieval exceeds token_budget")
        for key in ("current_verdict_accessed", "hidden_tests_accessed",
                    "probe_answer_accessed", "later_memory_accessed"):
            _require(not _boolean(run[key], f"{prefix}.{key}"),
                     f"{prefix}: forbidden access recorded in {key}")
        _verify_record_hash(run, "run_sha256", prefix)
    _unique(run_ids, "harness run ids")
    _unique(run_tasks, "harness run task ids")
    _require(_digest(document["memory_sha256"], "memory_sha256") ==
             harness_memory_sha256(document),
             "memory_sha256 mismatch")
    return {
        "schema_version": "p16_harness_memory_validation_v1",
        "valid": True,
        "record_count": len(records),
        "run_count": len(runs),
        "top_k": top_k,
        "token_budget": token_budget,
        "memory_sha256": document["memory_sha256"],
        "limitations": "Retrieval selection quality and oracle claims are not re-executed.",
    }


def _self_edit_rejection_reasons(record: dict[str, Any]) -> list[str]:
    reasons = []
    normalized = record["normalized"]
    if not record["schema_valid"]:
        reasons.append("schema_invalid")
    elif not all(isinstance(normalized[key], str) and normalized[key]
                 for key in ("general_rule", "applicability_trigger",
                             "non_applicability_boundary")):
        reasons.append("normalized_fields_empty")
    if any(record["leakage"].values()):
        reasons.append("leakage_detected")
    roles = Counter(row["role"] for row in record["oracle_examples"])
    if roles["positive"] < 1 or roles["near_negative"] < 1:
        reasons.append("example_roles_incomplete")
    if any(not row["verifier_correct"] for row in record["oracle_examples"]):
        reasons.append("oracle_example_failed")
    if record["exact_duplicate"]:
        reasons.append("exact_duplicate")
    if record["semantic_near_duplicate"]:
        reasons.append("semantic_near_duplicate")
    if not record["human_approved"]:
        reasons.append("human_not_approved")
    return reasons


def validate_self_edit_batch(document: dict[str, Any]) -> dict[str, Any]:
    """Validate self-edit proposals as audited candidate data, never updates."""

    _exact_keys(document, {"schema_version", "batch_id", "records", "batch_sha256"},
                "self-edit batch")
    _require(document["schema_version"] == "p16_self_edit_batch_v1",
             "unexpected self-edit schema_version")
    _text(document["batch_id"], "batch_id")
    records = document["records"]
    _require(isinstance(records, list) and records, "self-edit records must be nonempty")
    edit_ids: list[str] = []
    decisions: Counter[str] = Counter()
    for index, raw_record in enumerate(records):
        prefix = f"records[{index}]"
        record = _exact_keys(raw_record, {
            "edit_id", "correction_record_id", "error_family_id", "raw_proposal",
            "raw_proposal_sha256", "normalized", "normalized_sha256", "schema_valid",
            "leakage", "oracle_examples", "exact_duplicate",
            "semantic_near_duplicate", "human_approved", "applied_in_same_episode",
            "decision", "rejection_reasons", "record_sha256",
        }, prefix)
        for key in ("edit_id", "correction_record_id", "error_family_id", "raw_proposal"):
            _text(record[key], f"{prefix}.{key}")
        edit_ids.append(record["edit_id"])
        _require(_digest(record["raw_proposal_sha256"],
                         f"{prefix}.raw_proposal_sha256") ==
                 sha256_text(record["raw_proposal"]),
                 f"{prefix}: raw proposal hash mismatch")
        normalized = _exact_keys(record["normalized"], {
            "general_rule", "applicability_trigger", "non_applicability_boundary"
        }, f"{prefix}.normalized")
        for key in normalized:
            _text(normalized[key], f"{prefix}.normalized.{key}", allow_empty=True)
        _require(_digest(record["normalized_sha256"],
                         f"{prefix}.normalized_sha256") == canonical_sha256(normalized),
                 f"{prefix}: normalized hash mismatch")
        _boolean(record["schema_valid"], f"{prefix}.schema_valid")
        leakage = _exact_keys(record["leakage"], {
            "probe_text", "control_text", "sealed_text", "protected_text",
            "answer_copy", "source_id_copy",
        }, f"{prefix}.leakage")
        for key in leakage:
            _boolean(leakage[key], f"{prefix}.leakage.{key}")
        examples = record["oracle_examples"]
        _require(isinstance(examples, list), f"{prefix}.oracle_examples must be a list")
        example_ids: list[str] = []
        for example_index, raw_example in enumerate(examples):
            example_prefix = f"{prefix}.oracle_examples[{example_index}]"
            example = _exact_keys(raw_example, {
                "example_id", "role", "prompt_sha256", "oracle_sha256",
                "verifier_correct",
            }, example_prefix)
            _text(example["example_id"], f"{example_prefix}.example_id")
            example_ids.append(example["example_id"])
            _require(example["role"] in {"positive", "near_negative"},
                     f"{example_prefix}.role unsupported")
            _digest(example["prompt_sha256"], f"{example_prefix}.prompt_sha256")
            _digest(example["oracle_sha256"], f"{example_prefix}.oracle_sha256")
            _boolean(example["verifier_correct"], f"{example_prefix}.verifier_correct")
        _unique(example_ids, f"{prefix} oracle example ids")
        for key in ("exact_duplicate", "semantic_near_duplicate", "human_approved",
                    "applied_in_same_episode"):
            _boolean(record[key], f"{prefix}.{key}")
        _require(not record["applied_in_same_episode"],
                 f"{prefix}: self-edit cannot be applied in the same episode")
        reasons = record["rejection_reasons"]
        _require(isinstance(reasons, list) and
                 all(isinstance(value, str) and value for value in reasons),
                 f"{prefix}.rejection_reasons must be strings")
        _unique(reasons, f"{prefix}.rejection_reasons")
        expected_reasons = _self_edit_rejection_reasons(record)
        _require(reasons == expected_reasons,
                 f"{prefix}: rejection_reasons differ; expected {expected_reasons}")
        expected_decision = "accepted" if not expected_reasons else "rejected"
        _require(record["decision"] == expected_decision,
                 f"{prefix}: decision must be {expected_decision}")
        decisions[record["decision"]] += 1
        _verify_record_hash(record, "record_sha256", prefix)
    _unique(edit_ids, "self-edit ids")
    _require(_digest(document["batch_sha256"], "batch_sha256") ==
             canonical_sha256(_without(document, "batch_sha256")),
             "batch_sha256 mismatch")
    return {
        "schema_version": "p16_self_edit_batch_validation_v1",
        "valid": True,
        "record_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "same_episode_updates": 0,
        "limitations": "Recorded verifier/human evidence only; no proposal was applied.",
    }


def validate_pool(document: dict[str, Any]) -> dict[str, Any]:
    """Validate capacity and disjointness for the recovery-first study roles."""

    _exact_keys(document, {"schema_version", "pool_id", "frozen", "records",
                           "correction_order", "pool_sha256"}, "pool")
    _require(document["schema_version"] == "p16_recovery_first_pool_v1",
             "unexpected pool schema_version")
    _text(document["pool_id"], "pool_id")
    _require(_boolean(document["frozen"], "frozen"), "pool must be frozen")
    records = document["records"]
    _require(isinstance(records, list), "pool records must be a list")
    _require(len(records) == sum(POOL_COUNTS.values()),
             f"pool must contain exactly {sum(POOL_COUNTS.values())} records")
    counts: Counter[str] = Counter()
    ids: list[str] = []
    task_ids: list[str] = []
    family_ids: list[str] = []
    source_ids: list[str] = []
    corrections: set[str] = set()
    for index, raw_record in enumerate(records):
        prefix = f"records[{index}]"
        record = _exact_keys(raw_record, {
            "item_id", "task_id", "family_id", "source_id", "role",
            "pre_intervention_class", "recovery_correct",
            "oracle_validated", "history_clean", "near_duplicate_clean",
            "sandbox_pass", "protected_overlap", "phase15_reserved_overlap",
            "prompt_sha256", "oracle_sha256", "record_sha256",
        }, prefix)
        for key in ("item_id", "task_id", "family_id", "source_id"):
            _text(record[key], f"{prefix}.{key}")
        ids.append(record["item_id"])
        task_ids.append(record["task_id"])
        family_ids.append(record["family_id"])
        source_ids.append(record["source_id"])
        role = record["role"]
        _require(role in POOL_COUNTS, f"{prefix}.role unsupported")
        counts[role] += 1
        _require(record["pre_intervention_class"] in FAILURE_CLASSES,
                 f"{prefix}.pre_intervention_class unsupported")
        recovery_correct = _boolean(record["recovery_correct"],
                                    f"{prefix}.recovery_correct")
        for key in ("oracle_validated", "history_clean", "near_duplicate_clean",
                    "sandbox_pass"):
            _require(_boolean(record[key], f"{prefix}.{key}"),
                     f"{prefix}: {key} must pass")
        for key in ("protected_overlap", "phase15_reserved_overlap"):
            _require(not _boolean(record[key], f"{prefix}.{key}"),
                     f"{prefix}: forbidden overlap in {key}")
        for key in ("prompt_sha256", "oracle_sha256"):
            _digest(record[key], f"{prefix}.{key}")
        if role == "correction":
            corrections.add(record["item_id"])
            _require(record["pre_intervention_class"] == "substantive_content",
                     f"{prefix}: correction must be substantive_content")
            _require(not recovery_correct,
                     f"{prefix}: correction must remain wrong after recovery")
        _verify_record_hash(record, "record_sha256", prefix)
    _require(dict(counts) == POOL_COUNTS, f"pool role counts differ: {dict(counts)}")
    _unique(ids, "pool item ids")
    _unique(task_ids, "pool task ids")
    _unique(family_ids, "pool family ids")
    _unique(source_ids, "pool source ids")
    correction_order = document["correction_order"]
    _require(isinstance(correction_order, list) and len(correction_order) == 64,
             "correction_order must contain 64 ids")
    _unique(correction_order, "correction_order")
    _require(set(correction_order) == corrections,
             "correction_order must exactly cover correction records")
    _require(_digest(document["pool_sha256"], "pool_sha256") ==
             canonical_sha256(_without(document, "pool_sha256")),
             "pool_sha256 mismatch")
    prefix_hashes = {
        str(n): canonical_sha256(correction_order[:n]) for n in LEVELS
    }
    return {
        "schema_version": "p16_recovery_first_pool_validation_v1",
        "valid": True,
        "record_count": len(records),
        "role_counts": dict(sorted(counts.items())),
        "nested_prefix_sha256": prefix_hashes,
        "pool_sha256": document["pool_sha256"],
        "limitations": "Capacity/recorded-evidence validation only; oracles were not executed.",
    }


def validate_release(document: dict[str, Any]) -> dict[str, Any]:
    """Cross-check a complete CPU contract bundle before GPU protocol design."""

    _exact_keys(document, {
        "schema_version", "release_id", "recovery_batch", "harness_memory",
        "self_edit_batch", "pool", "release_sha256",
    }, "CPU release")
    _require(document["schema_version"] == "p16_recovery_first_cpu_release_v1",
             "unexpected CPU release schema_version")
    _text(document["release_id"], "release_id")
    recovery_report = validate_recovery_batch(document["recovery_batch"])
    harness_report = validate_harness_memory(document["harness_memory"])
    self_edit_report = validate_self_edit_batch(document["self_edit_batch"])
    pool_report = validate_pool(document["pool"])

    correction_order = document["pool"]["correction_order"]
    memory_corrections = [row["correction_record_id"]
                          for row in document["harness_memory"]["records"]]
    _require(len(memory_corrections) in LEVELS,
             f"harness memory size must be one of {list(LEVELS)}")
    _require(memory_corrections == correction_order[:len(memory_corrections)],
             "harness records must follow the frozen correction_order prefix")
    correction_ids = set(correction_order)
    for index, record in enumerate(document["self_edit_batch"]["records"]):
        _require(record["correction_record_id"] in correction_ids,
                 f"self-edit record {index} references a non-correction item")

    evaluation_rows = [row for row in document["pool"]["records"]
                       if row["role"] != "correction"]
    evaluation_by_task = {row["task_id"]: row for row in evaluation_rows}
    for index, run in enumerate(document["harness_memory"]["runs"]):
        _require(run["task_id"] in evaluation_by_task,
                 f"harness run {index} is not a frozen evaluation task")
        _require(run["family_id"] == evaluation_by_task[run["task_id"]]["family_id"],
                 f"harness run {index} family differs from frozen evaluation task")

    _require(_digest(document["release_sha256"], "release_sha256") ==
             canonical_sha256(_without(document, "release_sha256")),
             "release_sha256 mismatch")
    return {
        "schema_version": "p16_recovery_first_cpu_release_validation_v1",
        "valid": True,
        "cpu_contract_complete": True,
        "gpu_execution_authorized": False,
        "release_sha256": document["release_sha256"],
        "artifact_hashes": {
            "recovery_batch": document["recovery_batch"]["batch_sha256"],
            "harness_memory": document["harness_memory"]["memory_sha256"],
            "self_edit_batch": document["self_edit_batch"]["batch_sha256"],
            "pool": document["pool"]["pool_sha256"],
        },
        "artifact_reports": {
            "recovery": recovery_report,
            "harness": harness_report,
            "self_edit": self_edit_report,
            "pool": pool_report,
        },
        "limitations": (
            "Completes the CPU artifact contract only. It does not re-execute oracles, "
            "authorize GPU execution, or establish an update effect."
        ),
    }


VALIDATORS = {
    "p16_recovery_batch_v1": validate_recovery_batch,
    "p16_harness_memory_v1": validate_harness_memory,
    "p16_self_edit_batch_v1": validate_self_edit_batch,
    "p16_recovery_first_pool_v1": validate_pool,
    "p16_recovery_first_cpu_release_v1": validate_release,
}


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(document, dict), "document must be an object")
    schema = document.get("schema_version")
    _require(schema in VALIDATORS, f"unsupported schema_version: {schema}")
    return VALIDATORS[schema](document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_document(json.loads(args.input.read_text(encoding="utf-8")))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            parser.error(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
