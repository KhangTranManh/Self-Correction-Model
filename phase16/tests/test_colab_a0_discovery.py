from __future__ import annotations

import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from colab_a0_discovery import (  # noqa: E402
    callable_shape,
    classify,
    extract_code,
    result_label,
)


class RecoveryTests(unittest.TestCase):
    def test_single_fence_is_recovered(self):
        code, status = extract_code("```python\ndef solve(x):\n    return x\n```")
        self.assertEqual(status, "accepted")
        self.assertTrue(code.startswith("def solve"))

    def test_unclosed_fence_is_rejected_without_mutation(self):
        raw = "```python\ndef solve(x):\n    return x"
        code, status = extract_code(raw)
        self.assertEqual(status, "rejected")
        self.assertEqual(code, raw)

    def test_plain_code_is_unchanged(self):
        raw = "def solve(x):\n    return x"
        self.assertEqual(extract_code(raw), (raw, "unchanged"))


class ClassificationTests(unittest.TestCase):
    def test_callable_shape_supports_function_and_solution_method(self):
        self.assertEqual(callable_shape("def f(x): return x", "f"), "top_level")
        self.assertEqual(
            callable_shape("class Solution:\n def f(self, x): return x", "f"),
            "Solution_method",
        )
        self.assertIsNone(callable_shape("def other(x): return x", "f"))

    def test_wrong_answer_is_substantive_candidate(self):
        self.assertEqual(classify("parseable", "wrong_answer", "unchanged"),
                         "substantive_content")

    def test_execution_failure_is_not_substantive(self):
        self.assertEqual(classify("not_directly_parseable", "runtime_error", "unchanged"),
                         "execution_or_scaffold")

    def test_result_label_uses_full_test_correctness(self):
        self.assertEqual(result_label({"status": "completed", "all_correct": True}),
                         "correct")
        self.assertEqual(result_label({"status": "completed", "all_correct": False}),
                         "wrong_answer")


if __name__ == "__main__":
    unittest.main()
