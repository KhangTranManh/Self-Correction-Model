from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_a0_cpu_sandbox_v2 import aggregate  # noqa: E402


class SandboxSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = ROOT / "results"
        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
        cls.plan = read(results / "a0_cpu_sandbox_plan_v2.json")
        cls.tracker = read(results / "a0_cpu_admission_tracker_v14.json")
        cls.previous = read(results / "a0_cpu_sandbox_v2/previous_24_exact.json")
        cls.batches = [read(p) for p in sorted((results / "a0_cpu_sandbox_v2").glob("batch_*.json"))]

    def test_all_records_and_unresolved_position(self):
        report = aggregate(self.plan, self.tracker, self.previous, self.batches, source_files={})
        self.assertEqual(report["counts"]["oracle_ready_checked"], 142)
        self.assertEqual(report["counts"]["substantive_residual_reproduced"], 141)
        self.assertEqual(report["positions_by_verdict"]["execution_or_scaffold_unresolved"], [392])
        self.assertFalse(report["claims"]["gpu_training_authorized"])

    def test_tampered_batch_or_missing_batch_is_rejected(self):
        tampered = copy.deepcopy(self.batches)
        tampered[0]["records"][0]["position"] = 999
        with self.assertRaisesRegex(RuntimeError, "batch hash mismatch"):
            aggregate(self.plan, self.tracker, self.previous, tampered, source_files={})
        with self.assertRaisesRegex(RuntimeError, "batch count mismatch"):
            aggregate(self.plan, self.tracker, self.previous, self.batches[:-1], source_files={})


if __name__ == "__main__":
    unittest.main()
