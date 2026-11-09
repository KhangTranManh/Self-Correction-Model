from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from track_a0_cpu_admission import (  # noqa: E402
    build_tracker,
    canonical_sha256,
    wilson_interval,
)


class A0CPUAdmissionTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((
            ROOT / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.reviews = [json.loads((
            ROOT / "results" / f"a0_cpu_review_batch{index}_v1.json"
        ).read_text(encoding="utf-8")) for index in range(1, 6)]

    def test_current_funnel_and_projection(self) -> None:
        report = build_tracker(self.inventory, self.reviews)
        self.assertEqual(report["funnel"]["unique_positions_reviewed"], 14)
        self.assertEqual(report["funnel"]["oracle_ready_remaining_gates"], 11)
        self.assertEqual(report["funnel"]["contract_rejected"], 3)
        self.assertEqual(report["funnel"]["unreviewed_queue"], 154)
        self.assertEqual(report["funnel"]["correction_admitted"], 0)
        self.assertEqual(report["positions"]["next_frozen_position"], 37)
        self.assertEqual(
            report["capacity_projection"]
                  ["estimated_total_reviews_for_64_at_point_rate"],
            82,
        )
        self.assertEqual(report["admission_projection"]["admission_rate"], None)
        claimed = report["tracker_sha256"]
        without_hash = copy.deepcopy(report)
        without_hash.pop("tracker_sha256")
        self.assertEqual(claimed, canonical_sha256(without_hash))

    def test_tampered_inventory_fails(self) -> None:
        inventory = copy.deepcopy(self.inventory)
        inventory["counts"]["correction_admitted"] = 1
        with self.assertRaisesRegex(RuntimeError, "inventory canonical hash"):
            build_tracker(inventory, self.reviews)

    def test_broken_review_chain_fails(self) -> None:
        reviews = copy.deepcopy(self.reviews)
        reviews[2]["source"]["previous_review_sha256"] = "0" * 64
        reviews[2]["review_sha256"] = canonical_sha256({
            key: value for key, value in reviews[2].items()
            if key != "review_sha256"
        })
        with self.assertRaisesRegex(RuntimeError, "review hash chain"):
            build_tracker(self.inventory, reviews)

    def test_duplicate_non_deferred_review_fails(self) -> None:
        reviews = copy.deepcopy(self.reviews[:2])
        duplicate = copy.deepcopy(reviews[1])
        duplicate["source"]["previous_review_sha256"] = reviews[1]["review_sha256"]
        duplicate["positions"] = [10]
        duplicate["rows"] = [copy.deepcopy(reviews[1]["rows"][1])]
        duplicate["review_sha256"] = canonical_sha256({
            key: value for key, value in duplicate.items()
            if key != "review_sha256"
        })
        reviews.append(duplicate)
        with self.assertRaisesRegex(RuntimeError, "reviewed twice"):
            build_tracker(self.inventory, reviews)

    def test_wilson_interval_contains_point(self) -> None:
        lower, upper = wilson_interval(11, 14)
        self.assertLess(lower, 11 / 14)
        self.assertGreater(upper, 11 / 14)


if __name__ == "__main__":
    unittest.main()
