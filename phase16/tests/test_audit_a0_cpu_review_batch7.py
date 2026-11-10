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

from audit_a0_cpu_review_batch7 import (  # noqa: E402
    FROZEN_POSITIONS,
    build,
    canonical_sha256,
)


class A0CPUReviewBatch7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((
            ROOT / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.bank = json.loads((
            ROOT / "data" / "a0_discovery_v1" / "candidate_bank.json"
        ).read_text(encoding="utf-8"))
        cls.batch6 = json.loads((
            ROOT / "results" / "a0_cpu_review_batch6_v1.json"
        ).read_text(encoding="utf-8"))
        cls.saved = json.loads((
            ROOT / "results" / "a0_cpu_review_batch7_v1.json"
        ).read_text(encoding="utf-8"))

    def test_rebuild_matches_saved_artifact(self) -> None:
        rebuilt = build(self.inventory, self.bank, self.batch6)
        self.assertEqual(rebuilt, self.saved)
        claimed = rebuilt["review_sha256"]
        without_hash = copy.deepcopy(rebuilt)
        without_hash.pop("review_sha256")
        self.assertEqual(claimed, canonical_sha256(without_hash))

    def test_five_frozen_positions_and_counts(self) -> None:
        self.assertEqual(self.saved["positions"], FROZEN_POSITIONS)
        self.assertEqual(self.saved["counts"]["newly_examined"], 5)
        self.assertEqual(self.saved["counts"]["cumulative_positions_examined"], 24)
        self.assertEqual(self.saved["counts"]["cumulative_oracle_ready"], 20)
        self.assertEqual(self.saved["counts"]["cumulative_contract_rejected"], 4)
        self.assertEqual(self.saved["next_position"], 53)

    def test_all_oracles_agree_and_catch_mutants(self) -> None:
        for row in self.saved["rows"]:
            evidence = row["independent_oracle"]
            self.assertEqual(evidence["status"], "validated_on_trusted_cpu_cases")
            self.assertTrue(evidence["oracle_reference_agree"])
            self.assertTrue(evidence["model_equivalent_mutant_caught"])
            self.assertEqual(row["review_disposition"],
                             "oracle_ready_remaining_gates_pending")

    def test_tampered_previous_batch_fails(self) -> None:
        batch6 = copy.deepcopy(self.batch6)
        batch6["next_position"] = 44
        batch6["review_sha256"] = canonical_sha256({
            key: value for key, value in batch6.items()
            if key != "review_sha256"
        })
        with self.assertRaisesRegex(RuntimeError, "position 43"):
            build(self.inventory, self.bank, batch6)

    def test_claims_remain_closed(self) -> None:
        self.assertEqual(self.saved["counts"]["correction_admitted"], 0)
        self.assertFalse(self.saved["claims"]["untrusted_model_code_executed"])
        self.assertFalse(self.saved["claims"]["correction_pool_certified"])
        self.assertFalse(self.saved["claims"]["gpu_training_authorized"])


if __name__ == "__main__":
    unittest.main()
