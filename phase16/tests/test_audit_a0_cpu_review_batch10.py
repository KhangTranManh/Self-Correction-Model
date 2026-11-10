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

from audit_a0_cpu_review_batch10 import (  # noqa: E402
    FROZEN_POSITIONS, build, canonical_sha256,
)


class A0CPUReviewBatch10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((ROOT / "results" /
            "a0_admission_inventory_v1.json").read_text(encoding="utf-8"))
        cls.bank = json.loads((ROOT / "data" / "a0_discovery_v1" /
            "candidate_bank.json").read_text(encoding="utf-8"))
        cls.previous = json.loads((ROOT / "results" /
            "a0_cpu_review_batch9_v1.json").read_text(encoding="utf-8"))
        cls.saved = json.loads((ROOT / "results" /
            "a0_cpu_review_batch10_v1.json").read_text(encoding="utf-8"))

    def test_rebuild_matches_saved_artifact(self) -> None:
        rebuilt = build(self.inventory, self.bank, self.previous)
        self.assertEqual(rebuilt, self.saved)
        claimed = rebuilt["review_sha256"]
        without_hash = copy.deepcopy(rebuilt)
        without_hash.pop("review_sha256")
        self.assertEqual(claimed, canonical_sha256(without_hash))

    def test_positions_counts_and_handoff(self) -> None:
        self.assertEqual(self.saved["positions"], FROZEN_POSITIONS)
        self.assertEqual(self.saved["counts"]["cumulative_positions_examined"], 39)
        self.assertEqual(self.saved["counts"]["cumulative_oracle_ready"], 31)
        self.assertEqual(self.saved["counts"]["cumulative_contract_rejected"], 8)
        self.assertEqual(self.saved["next_position"], 80)

    def test_three_pass_and_two_reject_fail_closed(self) -> None:
        ready = [row for row in self.saved["rows"]
                 if row["review_disposition"] ==
                 "oracle_ready_remaining_gates_pending"]
        rejected = [row for row in self.saved["rows"]
                    if row["review_disposition"].startswith("rejected_")]
        self.assertEqual([row["position"] for row in ready], [71, 73, 77])
        self.assertEqual([row["position"] for row in rejected], [74, 76])
        for row in ready:
            evidence = row["independent_oracle"]
            self.assertTrue(evidence["oracle_reference_agree"])
            self.assertTrue(evidence["model_equivalent_mutant_caught"])
        self.assertTrue(all(not row["correction_admitted"]
                            for row in self.saved["rows"]))

    def test_tampered_previous_batch_fails(self) -> None:
        previous = copy.deepcopy(self.previous)
        previous["next_position"] = 73
        previous["review_sha256"] = canonical_sha256({
            key: value for key, value in previous.items()
            if key != "review_sha256"
        })
        with self.assertRaisesRegex(RuntimeError, "position 71"):
            build(self.inventory, self.bank, previous)

    def test_claims_remain_closed(self) -> None:
        self.assertFalse(self.saved["claims"]["untrusted_model_code_executed"])
        self.assertFalse(self.saved["claims"]["correction_pool_certified"])
        self.assertFalse(self.saved["claims"]["gpu_training_authorized"])


if __name__ == "__main__":
    unittest.main()
