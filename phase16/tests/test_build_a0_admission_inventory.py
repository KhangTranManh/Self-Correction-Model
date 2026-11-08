from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_a0_admission_inventory import build_inventory  # noqa: E402


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def task(position: int) -> dict:
    return {
        "item_id": f"item-{position}",
        "family_id": f"family-{position}",
        "source_id_sha256": digest(f"source-{position}"),
        "source_site": "fixture",
        "function_name": f"solve_{position}",
        "prompt_sha256": digest(f"prompt-{position}"),
        "tests_sha256": digest(f"tests-{position}"),
        "history_exact_clean": True,
        "near_duplicate_component_size": 1,
    }


def completion(position: int, label: str, failure_class: str) -> dict:
    source = task(position)
    return {
        "position": position,
        "item_id": source["item_id"],
        "family_id": source["family_id"],
        "prompt_sha256": source["prompt_sha256"],
        "tests_sha256": source["tests_sha256"],
        "recovered_output_sha256": digest(f"output-{position}"),
        "verifier_label": label,
        "pre_intervention_class": failure_class,
        "verification": {
            "test_count": 10,
            "match_count": 4 if label == "wrong_answer" else None,
        },
    }


class AdmissionInventoryTests(unittest.TestCase):
    def setUp(self):
        self.bank = {"tasks": [task(index) for index in range(4)]}
        self.completions = [
            completion(0, "correct", "presentation_only"),
            completion(1, "wrong_answer", "substantive_content"),
            completion(2, "wrong_answer", "substantive_content"),
            completion(3, "runtime_error", "execution_or_scaffold"),
        ]
        self.rescore = {
            "rescored_counts": {
                "correct": 1,
                "runtime_error": 1,
                "wrong_answer": 2,
            }
        }
        self.nondeterministic = {2: {"item_id": "item-2"}}

    def test_inventory_is_fail_closed(self):
        result = build_inventory(
            self.bank, self.completions, self.rescore,
            self.nondeterministic)
        self.assertEqual(result["counts"]["dataset_side_failures"], 3)
        self.assertEqual(result["counts"]["admission_status"], {
            "pending_independent_oracle_review": 1,
            "rejected_before_independent_oracle_review": 2,
        })
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertFalse(result["claims"]["gpu_training_authorized"])
        by_position = {row["position"]: row for row in result["records"]}
        self.assertEqual(by_position[1]["gates"]["independent_oracle"],
                         "pending")
        self.assertEqual(by_position[2]["rejection_reasons"],
                         ["deterministic_rerun_failed"])
        self.assertEqual(by_position[3]["rejection_reasons"],
                         ["execution_or_scaffold_not_correction_data"])

    def test_correct_completion_is_not_a_failure_record(self):
        result = build_inventory(
            self.bank, self.completions, self.rescore,
            self.nondeterministic)
        self.assertNotIn(0, {row["position"] for row in result["records"]})

    def test_class_mismatch_fails(self):
        self.completions[1]["pre_intervention_class"] = (
            "execution_or_scaffold"
        )
        with self.assertRaisesRegex(RuntimeError, "failure class mismatch"):
            build_inventory(self.bank, self.completions, self.rescore,
                            self.nondeterministic)

    def test_duplicate_completion_identity_fails(self):
        self.completions[2]["item_id"] = self.completions[1]["item_id"]
        with self.assertRaisesRegex(RuntimeError, "item IDs are not unique"):
            build_inventory(self.bank, self.completions, self.rescore,
                            self.nondeterministic)


if __name__ == "__main__":
    unittest.main()
