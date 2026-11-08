from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "phase16" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_a0_cpu_review_batch1 import (  # noqa: E402
    REMOVE_PARENTHESES_CASES,
    boolean_parentheses_mutant,
    build,
    remove_parentheses_oracle,
    remove_parentheses_reference,
)


class A0CPUReviewBatch1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((
            ROOT / "phase16" / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.bank = json.loads((
            ROOT / "phase16" / "data" / "a0_discovery_v1" /
            "gpu_bundle" / "candidate_bank.json"
        ).read_text(encoding="utf-8"))

    def test_independent_implementations_agree(self):
        for _, value in REMOVE_PARENTHESES_CASES:
            self.assertEqual(remove_parentheses_oracle(value),
                             remove_parentheses_reference(value))

    def test_boolean_mutant_fails_nested_cases(self):
        self.assertNotEqual(boolean_parentheses_mutant("a(b(c)d)e"),
                            remove_parentheses_oracle("a(b(c)d)e"))

    def test_batch_is_fail_closed_and_ordered(self):
        result = build(self.inventory, self.bank)
        self.assertEqual(result["positions"], [0, 2, 6])
        self.assertEqual(result["next_position_blocked"], 6)
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertFalse(result["claims"]["gpu_training_authorized"])
        self.assertFalse(result["claims"]["untrusted_model_code_executed"])
        self.assertEqual(result["rows"][1]["review_disposition"],
                         "rejected_contract_under_specified")

    def test_tampered_inventory_record_fails(self):
        inventory = copy.deepcopy(self.inventory)
        row = next(row for row in inventory["records"]
                   if row["position"] == 0)
        row["item_id"] = "tampered"
        with self.assertRaisesRegex(RuntimeError,
                                    "inventory canonical hash mismatch"):
            build(inventory, self.bank)


if __name__ == "__main__":
    unittest.main()
