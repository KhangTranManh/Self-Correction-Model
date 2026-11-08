"""Unit tests for recall-audit family construction; no network or raw data."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sample_kodcode_recall_audit import build_sample, parse_seed_ids


class KodCodeRecallSampleTests(unittest.TestCase):
    def test_seed_parser(self):
        self.assertEqual(parse_seed_ids({"seed_ids": "['b', 'a', 'a']"}), ["a", "b"])
        self.assertEqual(parse_seed_ids({"seed_ids": [2, 1]}), ["1", "2"])
        with self.assertRaises(ValueError):
            parse_seed_ids({"seed_ids": "not data"})

    def test_positive_bridge_excludes_whole_component(self):
        rows = []
        for index in range(102):
            seeds = [str(index)]
            if index == 1:
                seeds = ["0", "1"]
            rows.append({
                "id": f"r{index}", "subset": "Algorithm", "seed_ids": seeds,
                "question_sha256": f"h{index}", "benchmark_similarity": 0.1,
                "text": f"text {index}",
            })
        result = build_sample(rows, {"r0"})
        self.assertEqual(result["components_with_shortlist_row"], 1)
        self.assertEqual(result["predicted_negative_components"], 100)
        sampled_ids = {row["representative_id"] for row in result["sample"]}
        self.assertNotIn("r0", sampled_ids)
        self.assertNotIn("r1", sampled_ids)

    def test_duplicate_candidate_id_is_rejected(self):
        row = {
            "id": "same", "subset": "Algorithm", "seed_ids": ["1"],
            "question_sha256": "h", "benchmark_similarity": 0.1,
            "text": "text",
        }
        with self.assertRaisesRegex(ValueError, "duplicate candidate row ID"):
            build_sample([row, dict(row)], set())


if __name__ == "__main__":
    unittest.main()
