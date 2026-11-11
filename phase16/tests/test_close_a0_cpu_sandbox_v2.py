from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from close_a0_cpu_sandbox_v2 import build_batch, prepare  # noqa: E402
from colab_a0_discovery import sandbox_execute  # noqa: E402
from rescore_a0_discovery import load_lock, verified_completions  # noqa: E402


class SandboxV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = ROOT / "results"
        cls.inventory = json.loads((results / "a0_admission_inventory_v1.json").read_text(encoding="utf-8"))
        cls.reviews = [json.loads((results / f"a0_cpu_review_batch{i}_v1.json").read_text(encoding="utf-8"))
                       for i in range(1, 37)]
        cls.previous = json.loads((results / "a0_cpu_sandbox_closure_v1.json").read_text(encoding="utf-8"))
        run = results / "a0_discovery_run_v2"
        cls.completions, _ = verified_completions(run, load_lock(run))

    def test_registry_covers_and_verifies_all_remaining(self):
        _, prepared = prepare(self.inventory, self.reviews, self.completions, self.previous)
        self.assertEqual(len(prepared), 118)
        self.assertEqual((prepared[0][0]["position"], prepared[-1][0]["position"]), (62, 399))

    def test_recovered_code_tampering_is_rejected(self):
        completions = copy.deepcopy(self.completions)
        completions[62]["recovered_output"] += "\n# altered"
        with self.assertRaisesRegex(RuntimeError, "recovered output hash mismatch"):
            prepare(self.inventory, self.reviews, completions, self.previous)

    def test_runtime_failure_is_not_counted_as_substantive(self):
        def execute(code, task, *, exact_expected):
            self.assertTrue(exact_expected)
            return {"status": "runtime_error", "returncode": 1}
        result = build_batch(self.inventory, self.reviews, self.completions,
                             self.previous, start=0, count=1, execute=execute,
                             preflight=lambda: None)
        self.assertEqual(result["records"][0]["verdict"], "execution_or_scaffold_unresolved")
        self.assertEqual(result["counts"]["substantive_residual_reproduced"], 0)

    def test_completed_wrong_and_correct_outputs_are_distinct(self):
        for matched, verdict in ((0, "substantive_residual_reproduced"),
                                 (None, "residual_not_reproduced")):
            def execute(code, task, *, exact_expected):
                count = len(task["test_inputs"])
                matches = count if matched is None else matched
                return {"status": "completed", "test_count": count,
                        "match_count": matches, "all_correct": matches == count}
            result = build_batch(self.inventory, self.reviews, self.completions,
                                 self.previous, start=0, count=1, execute=execute,
                                 preflight=lambda: None)
            self.assertEqual(result["records"][0]["verdict"], verdict)
            self.assertFalse(result["claims"]["gpu_training_authorized"])

    def test_exact_comparison_does_not_unwrap_independent_list(self):
        result = mock.Mock(returncode=0, stdout='__P16_A0_RESULT__=[5]\n', stderr='')
        with mock.patch("colab_a0_discovery._sandbox_command", return_value=["unused"]), \
             mock.patch("colab_a0_discovery.subprocess.run", return_value=result):
            task = {"function_name": "f", "test_inputs": [[]], "test_outputs": [[5]]}
            self.assertTrue(sandbox_execute("def f(): return 5", task)["all_correct"])
            self.assertFalse(sandbox_execute("def f(): return 5", task, exact_expected=True)["all_correct"])


if __name__ == "__main__":
    unittest.main()
