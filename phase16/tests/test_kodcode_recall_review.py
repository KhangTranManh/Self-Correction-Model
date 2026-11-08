"""Mechanical checks for the frozen KodCode recall review partition."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from record_kodcode_recall_review_v1 import (
    ADAPTER, DEFERRED_MULTIOP, DIRECT, NEAR_NEGATIVE, PLAIN_REJECT,
    UNRELATED, wilson,
)


class KodCodeRecallReviewTests(unittest.TestCase):
    def test_partition_is_exact(self):
        groups = [DIRECT, ADAPTER, NEAR_NEGATIVE, UNRELATED, DEFERRED_MULTIOP, PLAIN_REJECT]
        values = [value for group in groups for value in group]
        self.assertEqual(sorted(values), list(range(100)))
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual((len(DIRECT), len(ADAPTER)), (3, 1))

    def test_wilson_interval_contains_point(self):
        low, high = wilson(4, 100)
        self.assertLess(low, 0.04)
        self.assertGreater(high, 0.04)


if __name__ == "__main__":
    unittest.main()
