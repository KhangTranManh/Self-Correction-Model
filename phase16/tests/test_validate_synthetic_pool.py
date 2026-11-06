import copy
import hashlib
import unittest

from phase16.scripts.validate_synthetic_pool import validate


ROLE_VARIANTS = (
    [("screen", "count")] * 54
    + [("screen", "sum")] * 53
    + [("screen", "select")] * 53
    + [("positive_probe", "count")] * 14
    + [("positive_probe", "sum")] * 13
    + [("positive_probe", "select")] * 13
    + [("near_negative", "none")] * 20
    + [("unrelated_control", "none")] * 20
    + [("reserve", "count")] * 20
    + [("reserve", "sum")] * 20
    + [("reserve", "select")] * 20
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def valid_manifest() -> dict:
    tasks = []
    subsets = ["Algorithm", "Data_Structure", "Prefill", "Taco", "Codeforces", "Leetcode"]
    applicable_index = 0
    for index, (role, variant) in enumerate(ROLE_VARIANTS):
        applies = variant != "none"
        subset = subsets[applicable_index % len(subsets)] if applies else "Filter"
        applicable_index += int(applies)
        tasks.append({
            "task_id": f"task-{index}",
            "family_id": f"family-{index}",
            "subset": subset,
            "role": role,
            "variant": variant,
            "rule_applicable": applies,
            "universe_mode": "explicit_collection" if applies else "not_applicable",
            "contract_reviewed": True,
            "history_status": "clean",
            "near_duplicate_status": "clean",
            "license_status": "accepted_for_noncommercial_research",
            "oracle_validated": True,
            "sandbox_smoke": True,
            "deterministic_rerun": True,
            "test_count": 10,
            "reference_all_pass": True,
            "unrelated_mutant_caught": True,
            "upper_pair_count": 2 if applies else 0,
            "exclusive_upper_signature": applies,
            "prompt_sha256": digest(f"prompt-{index}"),
            "oracle_sha256": digest(f"oracle-{index}"),
            "reference_sha256": digest(f"reference-{index}"),
            "tests_sha256": digest(f"tests-{index}"),
        })
    return {
        "schema_version": "p16_synthetic_development_pool_v1",
        "source_revision": "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5",
        "tasks": tasks,
    }


class SyntheticPoolValidatorTests(unittest.TestCase):
    def test_valid_manifest(self):
        report = validate(valid_manifest())
        self.assertTrue(report["valid"])
        self.assertEqual(report["applicable_count"], 260)
        self.assertEqual(report["family_count"], 300)

    def test_duplicate_family_rejected(self):
        manifest = valid_manifest()
        manifest["tasks"][1]["family_id"] = manifest["tasks"][0]["family_id"]
        with self.assertRaisesRegex(ValueError, "duplicate family_id"):
            validate(manifest)

    def test_unknown_history_rejected(self):
        manifest = valid_manifest()
        manifest["tasks"][0]["history_status"] = "pending"
        with self.assertRaisesRegex(ValueError, "history status"):
            validate(manifest)

    def test_missing_upper_signature_rejected(self):
        manifest = valid_manifest()
        manifest["tasks"][0]["exclusive_upper_signature"] = False
        with self.assertRaisesRegex(ValueError, "signature"):
            validate(manifest)

    def test_subset_cap_rejected(self):
        manifest = valid_manifest()
        applicable = [task for task in manifest["tasks"] if task["rule_applicable"]]
        for task in applicable[:61]:
            task["subset"] = "Algorithm"
        with self.assertRaisesRegex(ValueError, "more than 60"):
            validate(manifest)


if __name__ == "__main__":
    unittest.main()

