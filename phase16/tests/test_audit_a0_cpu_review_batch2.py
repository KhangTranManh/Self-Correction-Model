from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "phase16" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_a0_cpu_review_batch2 import (  # noqa: E402
    PAIR_CASES,
    TAB_CASES,
    WEAPON_CASES,
    build,
    closest_pair_mutant,
    closest_pair_oracle,
    closest_pair_reference,
    identify_weapon_mutant,
    identify_weapon_oracle,
    identify_weapon_reference,
    transpose_grid_oracle,
    transpose_grid_reference,
    transpose_linewise_mutant,
)


class A0CPUReviewBatch2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((
            ROOT / "phase16" / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.bank = json.loads((
            ROOT / "phase16" / "data" / "a0_discovery_v1" /
            "gpu_bundle" / "candidate_bank.json"
        ).read_text(encoding="utf-8"))
        cls.batch1 = json.loads((
            ROOT / "phase16" / "results" / "a0_cpu_review_batch1_v1.json"
        ).read_text(encoding="utf-8"))

    def test_tab_oracles_agree(self):
        for _, (amount, tab) in TAB_CASES:
            self.assertEqual(transpose_grid_oracle(amount, tab),
                             transpose_grid_reference(amount, tab))

    def test_tab_mutant_misses_grid_alignment(self):
        _, (amount, tab) = TAB_CASES[1]
        self.assertNotEqual(transpose_linewise_mutant(amount, tab),
                            transpose_grid_oracle(amount, tab))

    def test_weapon_oracles_agree_and_mutant_fails(self):
        for _, value in WEAPON_CASES:
            self.assertEqual(identify_weapon_oracle(value),
                             identify_weapon_reference(value))
        self.assertNotEqual(identify_weapon_mutant("Unknown"),
                            identify_weapon_oracle("Unknown"))

    def test_pair_oracles_agree_and_mutant_fails(self):
        for _, value in PAIR_CASES:
            self.assertEqual(closest_pair_oracle(value),
                             closest_pair_reference(value))
        self.assertNotEqual(closest_pair_mutant(10), closest_pair_oracle(10))

    def test_batch_chain_is_fail_closed(self):
        result = build(self.inventory, self.bank, self.batch1)
        self.assertEqual(result["positions"], [6, 10, 12])
        self.assertEqual(result["next_position"], 16)
        self.assertEqual(result["counts"]["cumulative_positions_examined"], 5)
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertFalse(result["claims"]["untrusted_model_code_executed"])

    def test_tampered_previous_review_fails(self):
        previous = copy.deepcopy(self.batch1)
        previous["next_position_blocked"] = 10
        with self.assertRaisesRegex(RuntimeError,
                                    "batch-1 canonical hash mismatch"):
            build(self.inventory, self.bank, previous)


if __name__ == "__main__":
    unittest.main()
