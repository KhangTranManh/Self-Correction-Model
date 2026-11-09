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

from audit_a0_cpu_review_batch6 import (  # noqa: E402
    FROZEN_POSITIONS,
    build,
    canonical_sha256,
)


class A0CPUReviewBatch6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((
            ROOT / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        bank_path = ROOT / "data" / "a0_discovery_v1" / "candidate_bank.json"
        cls.bank = json.loads(bank_path.read_text(encoding="utf-8"))
        cls.batch5 = json.loads((
            ROOT / "results" / "a0_cpu_review_batch5_v1.json"
        ).read_text(encoding="utf-8"))
        cls.saved = json.loads((
            ROOT / "results" / "a0_cpu_review_batch6_v1.json"
        ).read_text(encoding="utf-8"))

    def test_rebuild_matches_saved_artifact(self) -> None:
        rebuilt = build(self.inventory, self.bank, self.batch5)
        self.assertEqual(rebuilt, self.saved)
        claimed = rebuilt["review_sha256"]
        without_hash = copy.deepcopy(rebuilt)
        without_hash.pop("review_sha256")
        self.assertEqual(claimed, canonical_sha256(without_hash))

    def test_five_frozen_positions_and_counts(self) -> None:
        self.assertEqual(self.saved["positions"], FROZEN_POSITIONS)
        self.assertEqual(self.saved["counts"]["newly_examined"], 5)
        self.assertEqual(self.saved["counts"]["cumulative_positions_examined"], 19)
        self.assertEqual(self.saved["counts"]["cumulative_oracle_ready"], 15)
        self.assertEqual(self.saved["counts"]["cumulative_contract_rejected"], 4)
        self.assertEqual(self.saved["next_position"], 43)

    def test_parse_fen_is_rejected_fail_closed(self) -> None:
        row = self.saved["rows"][0]
        self.assertEqual(row["position"], 37)
        self.assertEqual(row["contract_review"]["status"], "failed")
        self.assertEqual(row["review_disposition"],
                         "rejected_contract_under_specified")
        self.assertFalse(row["correction_admitted"])

    def test_four_oracles_agree_and_catch_mutants(self) -> None:
        ready = self.saved["rows"][1:]
        self.assertEqual(len(ready), 4)
        for row in ready:
            evidence = row["independent_oracle"]
            self.assertEqual(evidence["status"],
                             "validated_on_trusted_cpu_cases")
            self.assertTrue(evidence["oracle_reference_agree"])
            self.assertTrue(evidence["model_equivalent_mutant_caught"])
            self.assertFalse(evidence["published_test_values_used"])
            self.assertFalse(evidence["dataset_reference_solutions_used"])
            self.assertEqual(row["review_disposition"],
                             "oracle_ready_remaining_gates_pending")

    def test_tampered_previous_batch_fails(self) -> None:
        batch5 = copy.deepcopy(self.batch5)
        batch5["next_position"] = 39
        batch5["review_sha256"] = canonical_sha256({
            key: value for key, value in batch5.items()
            if key != "review_sha256"
        })
        with self.assertRaisesRegex(RuntimeError, "position 37"):
            build(self.inventory, self.bank, batch5)

    def test_claims_remain_closed(self) -> None:
        self.assertEqual(self.saved["counts"]["correction_admitted"], 0)
        self.assertFalse(self.saved["claims"]["untrusted_model_code_executed"])
        self.assertFalse(self.saved["claims"]["correction_pool_certified"])
        self.assertFalse(self.saved["claims"]["gpu_training_authorized"])


if __name__ == "__main__":
    unittest.main()
