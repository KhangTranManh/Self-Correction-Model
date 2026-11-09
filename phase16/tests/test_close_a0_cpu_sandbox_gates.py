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

from close_a0_cpu_sandbox_gates import (  # noqa: E402
    _load_completions, build, canonical_sha256,
)


class A0CPUSandboxClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((ROOT / "results" /
            "a0_admission_inventory_v1.json").read_text(encoding="utf-8"))
        cls.reviews = [json.loads((ROOT / "results" /
            f"a0_cpu_review_batch{i}_v1.json").read_text(encoding="utf-8"))
            for i in range(1, 7)]
        cls.completions = _load_completions(
            ROOT / "results" / "a0_discovery_run_v2")

    def test_all_15_failed_observations_reproduce_residuals(self) -> None:
        calls = []
        def execute(code, task):
            calls.append((code, task))
            return {"status": "completed", "test_count": len(task["test_inputs"]),
                    "match_count": 0, "all_correct": False,
                    "actual_sha256": "0" * 64}
        report = build(self.inventory, self.reviews, self.completions,
                       execute=execute, preflight=lambda: None)
        self.assertEqual(len(calls), 15)
        self.assertEqual(report["counts"]["residual_reproduced"], 15)
        self.assertEqual(report["counts"]["correction_admitted"], 0)
        self.assertFalse(report["claims"]["gpu_training_authorized"])
        claimed = report["closure_sha256"]
        without_hash = copy.deepcopy(report)
        without_hash.pop("closure_sha256")
        self.assertEqual(claimed, canonical_sha256(without_hash))

    def test_all_correct_is_fail_closed_not_reproduced(self) -> None:
        def execute(code, task):
            return {"status": "completed", "test_count": len(task["test_inputs"]),
                    "match_count": len(task["test_inputs"]), "all_correct": True,
                    "actual_sha256": "1" * 64}
        report = build(self.inventory, self.reviews, self.completions,
                       execute=execute, preflight=lambda: None)
        self.assertEqual(report["counts"]["residual_not_reproduced"], 15)
        self.assertTrue(all(row["sandbox_gate"] ==
            "failed_residual_not_reproduced" for row in report["records"]))

    def test_tampered_completion_identity_fails(self) -> None:
        completions = copy.deepcopy(self.completions)
        completions[0]["item_id"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "item_id mismatch"):
            build(self.inventory, self.reviews, completions,
                  execute=lambda code, task: {}, preflight=lambda: None)


if __name__ == "__main__":
    unittest.main()
