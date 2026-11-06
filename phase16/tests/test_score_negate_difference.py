"""Synthetic scorer fixtures; not evidence about a model or source pool."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_negate_difference import score  # noqa: E402


BASE = {"task_id": "synthetic", "a": 3, "b": 5, "status": "ok",
        "expansion": "-a+b", "final": 2}


class NegateDifferenceTest(unittest.TestCase):
    def test_correct(self):
        self.assertEqual(score(BASE)["label"], "correct")

    def test_signature_even_if_final_is_correct(self):
        self.assertEqual(score({**BASE, "expansion": "-a-b"})["label"],
                         "negate_difference_signature")

    def test_wrong_final_is_not_signature(self):
        self.assertEqual(score({**BASE, "final": -8})["label"], "other_failure")

    def test_missing_and_unsafe_expressions_are_unscorable(self):
        for expression in ("__import__('os').system('echo unsafe')", "a**b", "a/0"):
            self.assertEqual(score({**BASE, "expansion": expression})["label"],
                             "unscorable")
        self.assertEqual(score({**BASE, "status": "missing"})["label"], "unscorable")

    def test_degenerate_b_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            score({**BASE, "b": 0})


if __name__ == "__main__":
    unittest.main()
