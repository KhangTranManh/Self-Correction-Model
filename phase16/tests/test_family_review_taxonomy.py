"""Mechanical checks for the frozen 100-family taxonomy review artifacts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


class FamilyReviewTaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load("kodcode_family_review_manifest.json")
        cls.artifacts = [
            load("kodcode_family_lane_review_40_v1.json"),
            load("kodcode_family_lane_review_batch3.json"),
            load("kodcode_family_lane_review_60_99.json"),
        ]
        cls.rows = [row for artifact in cls.artifacts for row in artifact["rows"]]

    def test_frozen_prefix_coverage_and_identity(self) -> None:
        self.assertEqual([row["review_index"] for row in self.rows], list(range(100)))
        self.assertEqual(len({row["family_id"] for row in self.rows}), 100)
        for index, row in enumerate(self.rows):
            family = self.manifest["families"][index]
            self.assertEqual(row["family_id"], family["family_id"])
            self.assertEqual(row["representative_id"], family["representative_id"])
            self.assertEqual(row["question_sha256"], family["representative_question_sha256"])

    def test_cumulative_lane_counts(self) -> None:
        expected = {
            "direct_candidate": 17,
            "execution_adapter_candidate": 9,
            "control_candidate": 25,
            "hard_reject": 49,
        }
        observed = {
            lane: sum(row["review_lane"] == lane for row in self.rows)
            for lane in expected
        }
        self.assertEqual(observed, expected)
        self.assertEqual(observed["direct_candidate"] + observed["execution_adapter_candidate"], 26)

    def test_controls_and_deferred_status(self) -> None:
        self.assertEqual(sum(row["control_subtype"] == "near_negative" for row in self.rows), 1)
        self.assertEqual(sum(row["control_subtype"] == "unrelated" for row in self.rows), 24)
        deferred = [row for row in self.rows if row["deferred_status"] == "deferred_multiop"]
        self.assertEqual(len(deferred), 24)
        self.assertTrue(all(row["review_lane"] == "hard_reject" for row in deferred))

    def test_no_certification_or_role_assignment(self) -> None:
        for row in self.rows:
            self.assertEqual(row["oracle_status"], "pending")
            self.assertEqual(row["history_status"], "pending")
            self.assertEqual(row["near_duplicate_status"], "pending")
            self.assertEqual(row["sandbox_status"], "pending")
            self.assertNotIn("role", row)


if __name__ == "__main__":
    unittest.main()
