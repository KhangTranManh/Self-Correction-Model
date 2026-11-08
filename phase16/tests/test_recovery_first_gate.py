"""Synthetic checks for the P16-R0 recovery-first curve scorer."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_first_gate import LEVELS, score  # noqa: E402


def fixture(pass_levels: set[int] | None = None) -> dict:
    pass_levels = pass_levels or set()
    levels = []
    training_ids = [f"train_{index:02d}" for index in range(64)]
    for n in LEVELS:
        seeds = []
        for seed_index in range(3):
            items = []
            for index in range(80):
                items.append({
                    "id": f"p{index:02d}", "family": f"pf{index:02d}", "role": "positive",
                    "harness_correct": False,
                    "update_correct": n in pass_levels and index < 4,
                })
            for role, count, prefix in (("near_negative", 20, "n"), ("unrelated", 40, "u")):
                for index in range(count):
                    items.append({
                        "id": f"{prefix}{index:02d}", "family": f"{prefix}f{index:02d}", "role": role,
                        "harness_correct": True, "update_correct": True,
                    })
            seeds.append({"seed": f"s{seed_index}", "items": items})
        levels.append({
            "n": n,
            "training_ids": training_ids[:n],
            "integrity": {
                "reset_ok": True,
                "hashes_ok": True,
                "audit_complete": True,
                "leakage_clear": True,
                "local_rescore_ok": True,
            },
            "seeds": seeds,
        })
    return {
        "schema_version": "p16_recovery_first_curve_v1",
        "pool_sha256": "a" * 64,
        "correction_order": training_ids,
        "levels": levels,
    }


class RecoveryFirstGateTests(unittest.TestCase):
    def test_stable_onset_requires_adjacent_passes(self):
        result = score(fixture({16, 32, 64}))
        self.assertEqual(result["stable_update_onset_n"], 16)
        self.assertEqual([row["level_pass"] for row in result["levels"]], [False, True, True, True])

    def test_largest_level_alone_is_unresolved(self):
        result = score(fixture({64}))
        self.assertIsNone(result["stable_update_onset_n"])
        self.assertTrue(result["largest_level_pass_without_confirmation"])

    def test_control_harm_blocks_level(self):
        data = fixture({16, 32})
        level = data["levels"][1]
        for seed in level["seeds"]:
            harmed = [row for row in seed["items"] if row["role"] == "unrelated"][:2]
            for row in harmed:
                row["update_correct"] = False
        result = score(data)
        self.assertFalse(result["levels"][1]["level_pass"])
        self.assertIsNone(result["stable_update_onset_n"])

    def test_integrity_failure_blocks_level(self):
        data = fixture({16, 32})
        data["levels"][1]["integrity"]["leakage_clear"] = False
        result = score(data)
        self.assertFalse(result["levels"][1]["integrity_pass"])
        self.assertFalse(result["levels"][1]["level_pass"])
        self.assertIsNone(result["stable_update_onset_n"])

    def test_rejects_non_nested_training_prefix(self):
        data = fixture()
        data["levels"][1]["training_ids"][0] = "replacement"
        with self.assertRaisesRegex(ValueError, "nested"):
            score(data)

    def test_rejects_prefix_that_differs_from_pool_order(self):
        data = fixture()
        data["correction_order"][0] = "different"
        with self.assertRaisesRegex(ValueError, "correction_order prefix"):
            score(data)

    def test_rejects_non_boolean_verdict(self):
        data = fixture()
        data["levels"][0]["seeds"][0]["items"][0]["update_correct"] = 1
        with self.assertRaisesRegex(ValueError, "Boolean"):
            score(data)

    def test_input_is_not_mutated(self):
        data = fixture({8, 16})
        original = copy.deepcopy(data)
        score(data)
        self.assertEqual(data, original)

    def test_reports_training_prefix_hash(self):
        result = score(fixture())
        digest = result["levels"][0]["training_ids_sha256"]
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r"^[0-9a-f]+$")


if __name__ == "__main__":
    unittest.main()
