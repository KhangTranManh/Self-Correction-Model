from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "phase16" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_a0_cpu_review_batch5 import (  # noqa: E402
    FISHEX_CASES,
    KA_CASES,
    build,
    fishex_mutant,
    fishex_oracle,
    fishex_reference,
    ka_cipher_mutant,
    ka_cipher_oracle,
    ka_cipher_reference,
)


class A0CPUReviewBatch5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((
            ROOT / "phase16" / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.bank = json.loads((
            ROOT / "phase16" / "data" / "a0_discovery_v1" /
            "candidate_bank.json"
        ).read_text(encoding="utf-8"))
        cls.batch4 = json.loads((
            ROOT / "phase16" / "results" / "a0_cpu_review_batch4_v1.json"
        ).read_text(encoding="utf-8"))

    def test_ka_oracles_agree_and_mutant_fails(self):
        for _, argument in KA_CASES:
            self.assertEqual(ka_cipher_oracle(argument),
                             ka_cipher_reference(argument))
        self.assertEqual(ka_cipher_oracle("Abbaa"), "kaAkabbaa")
        self.assertNotEqual(ka_cipher_mutant("Abbaa"),
                            ka_cipher_oracle("Abbaa"))

    def test_fishex_oracles_agree_and_mutant_fails(self):
        for _, argument in FISHEX_CASES:
            self.assertEqual(fishex_oracle(argument),
                             fishex_reference(argument))
        self.assertEqual(fishex_oracle("redlionfish"), 12)
        self.assertNotEqual(fishex_mutant("redlionfish"),
                            fishex_oracle("redlionfish"))

    def test_batch_chain_is_fail_closed(self):
        result = build(self.inventory, self.bank, self.batch4)
        self.assertEqual(result["positions"], [30, 31, 36])
        self.assertEqual(result["next_position"], 37)
        self.assertEqual(result["counts"]["cumulative_positions_examined"],
                         14)
        self.assertEqual(result["counts"]["cumulative_oracle_ready"], 11)
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertEqual(result["rows"][2]["contract_review"]["status"],
                         "failed")
        self.assertFalse(result["claims"]["untrusted_model_code_executed"])

    def test_tampered_previous_review_fails(self):
        previous = copy.deepcopy(self.batch4)
        previous["next_position"] = 36
        with self.assertRaisesRegex(RuntimeError,
                                    "batch-4 canonical hash mismatch"):
            build(self.inventory, self.bank, previous)


if __name__ == "__main__":
    unittest.main()
