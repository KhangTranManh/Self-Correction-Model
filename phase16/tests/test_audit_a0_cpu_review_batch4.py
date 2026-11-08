from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "phase16" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_a0_cpu_review_batch4 import (  # noqa: E402
    DAY_PLAN_CASES,
    RACE_CASES,
    SCRAP_CASES,
    build,
    calculate_scrap_mutant,
    calculate_scrap_oracle,
    calculate_scrap_reference,
    day_plan_mutant,
    day_plan_oracle,
    day_plan_reference,
    race_mutant,
    race_oracle,
    race_reference,
)


class A0CPUReviewBatch4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads((
            ROOT / "phase16" / "results" / "a0_admission_inventory_v1.json"
        ).read_text(encoding="utf-8"))
        cls.bank = json.loads((
            ROOT / "phase16" / "data" / "a0_discovery_v1" /
            "gpu_bundle" / "candidate_bank.json"
        ).read_text(encoding="utf-8"))
        cls.batch3 = json.loads((
            ROOT / "phase16" / "results" / "a0_cpu_review_batch3_v1.json"
        ).read_text(encoding="utf-8"))

    def test_scrap_oracles_agree_and_mutant_fails(self):
        for _, arguments in SCRAP_CASES:
            self.assertEqual(calculate_scrap_oracle(*arguments),
                             calculate_scrap_reference(*arguments))
        self.assertNotEqual(calculate_scrap_mutant([10], 1),
                            calculate_scrap_oracle([10], 1))

    def test_race_oracles_agree_and_mutant_fails(self):
        for _, arguments in RACE_CASES:
            self.assertEqual(race_oracle(*arguments),
                             race_reference(*arguments))
        self.assertNotEqual(race_mutant(720, 850, 70),
                            race_oracle(720, 850, 70))

    def test_day_plan_oracles_agree_and_mutant_fails(self):
        for _, arguments in DAY_PLAN_CASES:
            self.assertEqual(day_plan_oracle(*arguments),
                             day_plan_reference(*arguments))
        self.assertNotEqual(day_plan_mutant(8, 5, 30),
                            day_plan_oracle(8, 5, 30))

    def test_batch_chain_is_fail_closed(self):
        result = build(self.inventory, self.bank, self.batch3)
        self.assertEqual(result["positions"], [23, 28, 29])
        self.assertEqual(result["next_position"], 30)
        self.assertEqual(result["counts"]["cumulative_positions_examined"], 11)
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertFalse(result["claims"]["untrusted_model_code_executed"])

    def test_tampered_previous_review_fails(self):
        previous = copy.deepcopy(self.batch3)
        previous["next_position"] = 28
        with self.assertRaisesRegex(RuntimeError,
                                    "batch-3 canonical hash mismatch"):
            build(self.inventory, self.bank, previous)


if __name__ == "__main__":
    unittest.main()
