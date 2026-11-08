"""Synthetic tests for the Phase 16 recovery-first CPU contracts."""

from __future__ import annotations

import copy
import unittest

from phase16.scripts.validate_recovery_first_cpu import (
    POOL_COUNTS,
    canonical_sha256,
    harness_memory_sha256,
    render_prompt_patch,
    sha256_text,
    validate_document,
    validate_harness_memory,
    validate_pool,
    validate_recovery_batch,
    validate_release,
    validate_self_edit_batch,
)


def seal_record(record: dict, key: str = "record_sha256") -> dict:
    record[key] = canonical_sha256(record)
    return record


def recovery_record(record_id: str, fixture_role: str, raw: str, recovered: str,
                    status: str, raw_correct: bool, recovered_correct: bool,
                    failure_class: str, rejection_reason: str = "") -> dict:
    record = {
        "record_id": record_id,
        "task_id": f"task-{record_id}",
        "family_id": "format-family",
        "source_id": f"source-{record_id}",
        "split": "engineering_fixture",
        "fixture_role": fixture_role,
        "pre_intervention_class": failure_class,
        "raw_output": raw,
        "recovered_output": recovered,
        "recovery_status": status,
        "rejection_reason": rejection_reason,
        "verification": {
            "oracle_sha256": sha256_text(f"oracle-{record_id}"),
            "raw_output_sha256": sha256_text(raw),
            "recovered_output_sha256": sha256_text(recovered),
            "raw_correct": raw_correct,
            "recovered_correct": recovered_correct,
            "payload_preserved": True,
            "idempotent": True,
            "deterministic_rerun": True,
            "noninterference_on_valid": fixture_role == "already_valid",
            "reference_answer_accessed": False,
            "hidden_tests_accessed": False,
            "other_arm_accessed": False,
        },
    }
    return seal_record(record)


def valid_recovery_batch() -> dict:
    wrapper = {
        "wrapper_id": "suffix-normalizer",
        "version": "1",
        "family_id": "format-family",
        "rule_text": "Extract the final declared integer payload.",
        "rule_sha256": sha256_text("Extract the final declared integer payload."),
        "frozen": True,
        "reference_free": True,
    }
    document = {
        "schema_version": "p16_recovery_batch_v1",
        "batch_id": "recovery-fixtures-v1",
        "wrapper": wrapper,
        "records": [
            recovery_record("fixed", "payload_recovery", "answer: 42", "42",
                            "accepted", False, True, "presentation_only"),
            recovery_record("valid", "already_valid", "7", "7", "unchanged",
                            True, True, "presentation_only"),
            recovery_record("reject", "rejection", "unknown", "unknown", "rejected",
                            False, False, "ambiguous_or_unscorable", "no payload"),
            recovery_record("residual", "residual_content", "41", "41", "unchanged",
                            False, False, "substantive_content"),
        ],
    }
    document["batch_sha256"] = canonical_sha256(document)
    return document


def example(name: str) -> dict:
    return {
        "example_id": name,
        "task_id": f"task-{name}",
        "oracle_sha256": sha256_text(f"oracle-{name}"),
        "verifier_correct": True,
    }


def valid_harness_memory() -> dict:
    record = {
        "record_id": "rule-1",
        "version": "1",
        "error_family_id": "upper-bound",
        "correction_record_id": "correction-0",
        "applicability_trigger": "The contract declares an inclusive upper bound.",
        "non_applicability_boundary": "Do not apply to exclusive intervals.",
        "rule": "Include values equal to the upper endpoint.",
        "permitted_action": "prompt_rule",
        "positive_examples": [example("positive")],
        "near_negative_examples": [example("near")],
        "provenance": {
            "source_id": "source-correction-0",
            "source_revision": "revision-1",
            "reviewer_id": "human-1",
            "oracle_sha256": sha256_text("correction-oracle"),
        },
        "schema_validated": True,
        "leakage_clear": True,
        "human_approved": True,
    }
    seal_record(record)
    patch = render_prompt_patch({"rule-1": record}, ["rule-1"])
    run = {
        "run_id": "run-1",
        "task_id": "heldout-task-1",
        "family_id": "heldout-family-1",
        "retrieved_rule_ids": ["rule-1"],
        "prompt_patch": patch,
        "prompt_patch_sha256": sha256_text(patch),
        "retrieved_token_count": len(patch.split()),
        "current_verdict_accessed": False,
        "hidden_tests_accessed": False,
        "probe_answer_accessed": False,
        "later_memory_accessed": False,
    }
    seal_record(run, "run_sha256")
    document = {
        "schema_version": "p16_harness_memory_v1",
        "memory_id": "memory-v1",
        "version": "1",
        "frozen": True,
        "top_k": 1,
        "token_budget": 100,
        "token_count_method": "whitespace_v1",
        "records": [record],
        "runs": [run],
    }
    document["memory_sha256"] = harness_memory_sha256(document)
    return document


def oracle_example(name: str, role: str) -> dict:
    return {
        "example_id": name,
        "role": role,
        "prompt_sha256": sha256_text(f"prompt-{name}"),
        "oracle_sha256": sha256_text(f"oracle-{name}"),
        "verifier_correct": True,
    }


def self_edit_record(edit_id: str, *, human_approved: bool = True) -> dict:
    normalized = {
        "general_rule": "Respect the declared inclusive endpoint.",
        "applicability_trigger": "The problem says both endpoints are included.",
        "non_applicability_boundary": "Do not apply to half-open intervals.",
    }
    reasons = [] if human_approved else ["human_not_approved"]
    record = {
        "edit_id": edit_id,
        "correction_record_id": f"correction-{edit_id}",
        "error_family_id": "upper-bound",
        "raw_proposal": f"proposal-{edit_id}",
        "raw_proposal_sha256": sha256_text(f"proposal-{edit_id}"),
        "normalized": normalized,
        "normalized_sha256": canonical_sha256(normalized),
        "schema_valid": True,
        "leakage": {
            "probe_text": False,
            "control_text": False,
            "sealed_text": False,
            "protected_text": False,
            "answer_copy": False,
            "source_id_copy": False,
        },
        "oracle_examples": [
            oracle_example(f"{edit_id}-positive", "positive"),
            oracle_example(f"{edit_id}-near", "near_negative"),
        ],
        "exact_duplicate": False,
        "semantic_near_duplicate": False,
        "human_approved": human_approved,
        "applied_in_same_episode": False,
        "decision": "accepted" if human_approved else "rejected",
        "rejection_reasons": reasons,
    }
    return seal_record(record)


def valid_self_edit_batch() -> dict:
    document = {
        "schema_version": "p16_self_edit_batch_v1",
        "batch_id": "self-edit-v1",
        "records": [self_edit_record("accepted"),
                    self_edit_record("rejected", human_approved=False)],
    }
    document["batch_sha256"] = canonical_sha256(document)
    return document


def pool_record(index: int, role: str) -> dict:
    record = {
        "item_id": f"item-{index}",
        "task_id": f"task-{index}",
        "family_id": f"family-{index}",
        "source_id": f"source-{index}",
        "role": role,
        "pre_intervention_class": "substantive_content",
        "recovery_correct": False,
        "oracle_validated": True,
        "history_clean": True,
        "near_duplicate_clean": True,
        "sandbox_pass": True,
        "protected_overlap": False,
        "phase15_reserved_overlap": False,
        "prompt_sha256": sha256_text(f"prompt-{index}"),
        "oracle_sha256": sha256_text(f"oracle-{index}"),
    }
    return seal_record(record)


def valid_pool() -> dict:
    records = []
    index = 0
    for role, count in POOL_COUNTS.items():
        for _ in range(count):
            records.append(pool_record(index, role))
            index += 1
    correction_order = [row["item_id"] for row in records
                        if row["role"] == "correction"]
    document = {
        "schema_version": "p16_recovery_first_pool_v1",
        "pool_id": "pool-v1",
        "frozen": True,
        "records": records,
        "correction_order": correction_order,
    }
    document["pool_sha256"] = canonical_sha256(document)
    return document


def valid_release() -> dict:
    pool = valid_pool()
    base = valid_harness_memory()["records"][0]
    memory_records = []
    for index, correction_id in enumerate(pool["correction_order"][:8]):
        record = copy.deepcopy(base)
        record["record_id"] = f"rule-{index}"
        record["correction_record_id"] = correction_id
        record["positive_examples"] = [example(f"positive-{index}")]
        record["near_negative_examples"] = [example(f"near-{index}")]
        record["provenance"]["source_id"] = f"source-correction-{index}"
        record["record_sha256"] = canonical_sha256({
            key: value for key, value in record.items() if key != "record_sha256"
        })
        memory_records.append(record)
    patch = render_prompt_patch({row["record_id"]: row for row in memory_records},
                                ["rule-0"])
    evaluation = pool["records"][64]
    run = {
        "run_id": "release-run-1",
        "task_id": evaluation["task_id"],
        "family_id": evaluation["family_id"],
        "retrieved_rule_ids": ["rule-0"],
        "prompt_patch": patch,
        "prompt_patch_sha256": sha256_text(patch),
        "retrieved_token_count": len(patch.split()),
        "current_verdict_accessed": False,
        "hidden_tests_accessed": False,
        "probe_answer_accessed": False,
        "later_memory_accessed": False,
    }
    seal_record(run, "run_sha256")
    memory = {
        "schema_version": "p16_harness_memory_v1",
        "memory_id": "release-memory-v1",
        "version": "1",
        "frozen": True,
        "top_k": 2,
        "token_budget": 100,
        "token_count_method": "whitespace_v1",
        "records": memory_records,
        "runs": [run],
    }
    memory["memory_sha256"] = harness_memory_sha256(memory)
    self_edits = valid_self_edit_batch()
    for index, record in enumerate(self_edits["records"]):
        record["correction_record_id"] = pool["correction_order"][index]
        record["record_sha256"] = canonical_sha256({
            key: value for key, value in record.items() if key != "record_sha256"
        })
    self_edits["batch_sha256"] = canonical_sha256({
        key: value for key, value in self_edits.items() if key != "batch_sha256"
    })
    release = {
        "schema_version": "p16_recovery_first_cpu_release_v1",
        "release_id": "cpu-release-v1",
        "recovery_batch": valid_recovery_batch(),
        "harness_memory": memory,
        "self_edit_batch": self_edits,
        "pool": pool,
    }
    release["release_sha256"] = canonical_sha256(release)
    return release


def reseal_release(release: dict) -> None:
    release["release_sha256"] = canonical_sha256({
        key: value for key, value in release.items() if key != "release_sha256"
    })


class RecoveryBatchTests(unittest.TestCase):
    def test_valid_recovery_batch(self):
        report = validate_recovery_batch(valid_recovery_batch())
        self.assertTrue(report["valid"])
        self.assertEqual(report["outcome_counts"]["fixed"], 1)
        self.assertEqual(report["outcome_counts"]["harmed"], 0)

    def test_tampered_raw_output_hash_rejected(self):
        data = valid_recovery_batch()
        data["records"][0]["raw_output"] = "tampered"
        with self.assertRaisesRegex(ValueError, "raw output hash"):
            validate_recovery_batch(data)

    def test_forbidden_oracle_access_rejected(self):
        data = valid_recovery_batch()
        data["records"][0]["verification"]["reference_answer_accessed"] = True
        with self.assertRaisesRegex(ValueError, "forbidden access"):
            validate_recovery_batch(data)

    def test_substantive_failure_cannot_be_recovery_fixed(self):
        data = valid_recovery_batch()
        row = data["records"][0]
        row["pre_intervention_class"] = "substantive_content"
        row["record_sha256"] = canonical_sha256({k: v for k, v in row.items()
                                                  if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "substantive_content"):
            validate_recovery_batch(data)


class HarnessMemoryTests(unittest.TestCase):
    def test_valid_harness_memory(self):
        report = validate_harness_memory(valid_harness_memory())
        self.assertTrue(report["valid"])
        self.assertEqual(report["run_count"], 1)

    def test_unlogged_prompt_patch_rejected(self):
        data = valid_harness_memory()
        data["runs"][0]["prompt_patch"] += "\nsecret patch"
        with self.assertRaisesRegex(ValueError, "not deterministic"):
            validate_harness_memory(data)

    def test_top_k_is_enforced(self):
        data = valid_harness_memory()
        second = copy.deepcopy(data["records"][0])
        second["record_id"] = "rule-2"
        second["correction_record_id"] = "correction-2"
        second["record_sha256"] = canonical_sha256({k: v for k, v in second.items()
                                                     if k != "record_sha256"})
        data["records"].append(second)
        data["runs"][0]["retrieved_rule_ids"] = ["rule-1", "rule-2"]
        with self.assertRaisesRegex(ValueError, "top_k"):
            validate_harness_memory(data)

    def test_current_verdict_access_rejected(self):
        data = valid_harness_memory()
        data["runs"][0]["current_verdict_accessed"] = True
        with self.assertRaisesRegex(ValueError, "forbidden access"):
            validate_harness_memory(data)

    def test_append_only_runs_do_not_change_memory_hash(self):
        data = valid_harness_memory()
        original_hash = data["memory_sha256"]
        second = copy.deepcopy(data["runs"][0])
        second["run_id"] = "run-2"
        second["task_id"] = "heldout-task-2"
        second["run_sha256"] = canonical_sha256({
            k: v for k, v in second.items() if k != "run_sha256"
        })
        data["runs"].append(second)
        report = validate_harness_memory(data)
        self.assertEqual(report["memory_sha256"], original_hash)


class SelfEditTests(unittest.TestCase):
    def test_accepts_and_preserves_rejected_records(self):
        report = validate_self_edit_batch(valid_self_edit_batch())
        self.assertEqual(report["decision_counts"], {"accepted": 1, "rejected": 1})
        self.assertEqual(report["same_episode_updates"], 0)

    def test_same_episode_application_rejected(self):
        data = valid_self_edit_batch()
        data["records"][0]["applied_in_same_episode"] = True
        with self.assertRaisesRegex(ValueError, "same episode"):
            validate_self_edit_batch(data)

    def test_false_acceptance_rejected(self):
        data = valid_self_edit_batch()
        row = data["records"][0]
        row["leakage"]["protected_text"] = True
        row["record_sha256"] = canonical_sha256({k: v for k, v in row.items()
                                                  if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "rejection_reasons differ"):
            validate_self_edit_batch(data)

    def test_duplicate_is_recorded_as_rejection(self):
        data = valid_self_edit_batch()
        row = data["records"][0]
        row["exact_duplicate"] = True
        row["decision"] = "rejected"
        row["rejection_reasons"] = ["exact_duplicate"]
        row["record_sha256"] = canonical_sha256({k: v for k, v in row.items()
                                                  if k != "record_sha256"})
        data["batch_sha256"] = canonical_sha256({k: v for k, v in data.items()
                                                  if k != "batch_sha256"})
        report = validate_self_edit_batch(data)
        self.assertEqual(report["decision_counts"], {"rejected": 2})


class PoolTests(unittest.TestCase):
    def test_valid_pool_and_prefix_hashes(self):
        report = validate_pool(valid_pool())
        self.assertTrue(report["valid"])
        self.assertEqual(report["role_counts"], POOL_COUNTS)
        self.assertEqual(set(report["nested_prefix_sha256"]), {"8", "16", "32", "64"})

    def test_correction_must_survive_recovery(self):
        data = valid_pool()
        data["records"][0]["recovery_correct"] = True
        with self.assertRaisesRegex(ValueError, "wrong after recovery"):
            validate_pool(data)

    def test_protected_overlap_rejected(self):
        data = valid_pool()
        data["records"][70]["protected_overlap"] = True
        with self.assertRaisesRegex(ValueError, "forbidden overlap"):
            validate_pool(data)

    def test_duplicate_family_rejected(self):
        data = valid_pool()
        data["records"][1]["family_id"] = data["records"][0]["family_id"]
        data["records"][1]["record_sha256"] = canonical_sha256({
            k: v for k, v in data["records"][1].items() if k != "record_sha256"
        })
        with self.assertRaisesRegex(ValueError, "family ids must be unique"):
            validate_pool(data)


class DispatchTests(unittest.TestCase):
    def test_dispatch_and_input_not_mutated(self):
        data = valid_recovery_batch()
        original = copy.deepcopy(data)
        report = validate_document(data)
        self.assertTrue(report["valid"])
        self.assertEqual(data, original)

    def test_unknown_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            validate_document({"schema_version": "unknown"})


class ReleaseTests(unittest.TestCase):
    def test_valid_cross_artifact_release(self):
        report = validate_release(valid_release())
        self.assertTrue(report["cpu_contract_complete"])
        self.assertFalse(report["gpu_execution_authorized"])
        self.assertEqual(report["artifact_reports"]["harness"]["record_count"], 8)

    def test_harness_must_follow_pool_prefix(self):
        data = valid_release()
        record = data["harness_memory"]["records"][0]
        record["correction_record_id"] = data["pool"]["correction_order"][9]
        record["record_sha256"] = canonical_sha256({
            key: value for key, value in record.items() if key != "record_sha256"
        })
        data["harness_memory"]["memory_sha256"] = harness_memory_sha256(
            data["harness_memory"])
        reseal_release(data)
        with self.assertRaisesRegex(ValueError, "correction_order prefix"):
            validate_release(data)

    def test_harness_run_cannot_use_correction_task(self):
        data = valid_release()
        run = data["harness_memory"]["runs"][0]
        correction = data["pool"]["records"][0]
        run["task_id"] = correction["task_id"]
        run["family_id"] = correction["family_id"]
        run["run_sha256"] = canonical_sha256({
            key: value for key, value in run.items() if key != "run_sha256"
        })
        reseal_release(data)
        with self.assertRaisesRegex(ValueError, "not a frozen evaluation task"):
            validate_release(data)


if __name__ == "__main__":
    unittest.main()
