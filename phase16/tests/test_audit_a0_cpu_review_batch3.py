from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "phase16" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_a0_cpu_review_batch3 import (  # noqa: E402
    CASE_ID_CASES,
    DIFFERENCE_CASES,
    build,
    case_id_mutant,
    case_id_oracle,
    case_id_reference,
    difference_mutant,
    difference_oracle,
    difference_reference,
)


class A0CPUReviewBatch3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((
            ROOT / "phase16" / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.bank = json.loads((
            ROOT / "phase16" / "data" / "a0_discovery_v1" /
            "candidate_bank.json"
        ).read_text(encoding="utf-8"))
        cls.batch2 = json.loads((
            ROOT / "phase16" / "results" / "a0_cpu_review_batch2_v1.json"
        ).read_text(encoding="utf-8"))

    def test_case_id_oracles_agree(self):
        for _, value in CASE_ID_CASES:
            self.assertEqual(case_id_oracle(value), case_id_reference(value))

    def test_case_id_mutant_fails_contract_examples(self):
        for value in ("helloworld", "hello__world", "hello-World"):
            self.assertNotEqual(case_id_mutant(value), case_id_oracle(value))

    def test_difference_oracles_agree(self):
        for _, value in DIFFERENCE_CASES:
            self.assertEqual(difference_oracle(value),
                             difference_reference(value))

    def test_difference_mutant_fails_all_equal(self):
        value = ["11-11", "344-344"]
        self.assertNotEqual(difference_mutant(value), difference_oracle(value))

    def test_batch_chain_is_fail_closed(self):
        result = build(self.inventory, self.bank, self.batch2)
        self.assertEqual(result["positions"], [16, 21, 22])
        self.assertEqual(result["next_position"], 23)
        self.assertEqual(result["counts"]["cumulative_positions_examined"], 8)
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertEqual(result["rows"][0]["review_disposition"],
                         "rejected_contract_under_specified")
        self.assertFalse(result["claims"]["untrusted_model_code_executed"])

    def test_tampered_previous_review_fails(self):
        previous = copy.deepcopy(self.batch2)
        previous["next_position"] = 21
        with self.assertRaisesRegex(RuntimeError,
                                    "batch-2 canonical hash mismatch"):
            build(self.inventory, self.bank, previous)


if __name__ == "__main__":
    unittest.main()
