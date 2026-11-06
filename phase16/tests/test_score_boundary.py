"""Mechanical fixtures for the Phase 16 error-signature definition."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_boundary import score  # noqa: E402


def case(case_id, kind, expected, *, pair_id=None, upper=None, **extra):
    result = {"case_id": case_id, "kind": kind, "expected": expected, "actual": expected}
    if pair_id is not None:
        result.update(pair_id=pair_id, upper=upper)
    result.update(extra)
    return result


def count_record():
    return {
        "task_id": "synthetic_count_fixture",
        "variant": "count",
        "execution_status": "ok",
        "cases": [
            case("ordinary", "ordinary", 1),
            case("lower", "lower", 1),
            case("outside", "outside", 0),
            case("empty", "empty", 0),
            case("single_point", "single_point", 1),
            case("duplicate", "duplicate", 2),
            case("p1_anchor", "upper_anchor", 1, pair_id="p1", upper=5),
            case("p1_added", "upper_added", 2, pair_id="p1", upper=5),
            case("p2_anchor", "upper_anchor", 2, pair_id="p2", upper=7),
            case("p2_added", "upper_added", 3, pair_id="p2", upper=7),
        ],
    }


def cases_by_id(record):
    return {item["case_id"]: item for item in record["cases"]}


class BoundaryScoreTest(unittest.TestCase):
    def test_correct_count(self):
        result = score(count_record())
        self.assertEqual(result["label"], "correct")
        self.assertTrue(result["full_correct"])
        self.assertEqual(result["passed_cases"], 10)

    def test_upper_boundary_signature_count(self):
        record = count_record()
        items = cases_by_id(record)
        items["single_point"]["actual"] = 0
        items["duplicate"]["actual"] = 0
        items["p1_added"]["actual"] = 1
        items["p2_added"]["actual"] = 2
        result = score(record)
        self.assertEqual(result["label"], "upper_boundary_signature")
        self.assertEqual(result["signature_pairs"], 2)
        self.assertFalse(result["full_correct"])

    def test_one_boundary_failure_is_not_signature(self):
        record = count_record()
        cases_by_id(record)["p1_added"]["actual"] = 1
        self.assertEqual(score(record)["label"], "other_failure")

    def test_interior_failure_is_not_signature(self):
        record = count_record()
        items = cases_by_id(record)
        items["ordinary"]["actual"] = 0
        items["p1_added"]["actual"] = 1
        items["p2_added"]["actual"] = 2
        self.assertEqual(score(record)["label"], "other_failure")

    def test_unrelated_failure_with_correct_boundary_is_other(self):
        record = count_record()
        cases_by_id(record)["empty"]["actual"] = 1
        self.assertEqual(score(record)["label"], "other_failure")

    def test_sum_requires_nonzero_oracle_delta(self):
        record = count_record()
        record["variant"] = "sum"
        items = cases_by_id(record)
        items["p1_added"]["expected"] = 6
        items["p1_added"]["actual"] = 1
        items["p1_added"]["added_value"] = 5
        items["p2_added"]["expected"] = 9
        items["p2_added"]["actual"] = 2
        items["p2_added"]["added_value"] = 7
        self.assertEqual(score(record)["label"], "upper_boundary_signature")
        items["p2_added"]["added_value"] = 0
        with self.assertRaisesRegex(ValueError, "Oracle pair relation"):
            score(record)

    def test_select_requires_unique_appended_id(self):
        record = count_record()
        record["variant"] = "select"
        mapping = {
            "ordinary": ["mid"], "lower": ["low"], "outside": [],
            "empty": [], "single_point": ["right"],
            "duplicate": ["right_a", "right_b"],
            "p1_anchor": ["mid"], "p1_added": ["mid", "right_5"],
            "p2_anchor": ["low", "mid"],
            "p2_added": ["low", "mid", "right_7"],
        }
        items = cases_by_id(record)
        for case_id, expected in mapping.items():
            items[case_id]["expected"] = deepcopy(expected)
            items[case_id]["actual"] = deepcopy(expected)
        items["p1_added"]["added_item"] = "right_5"
        items["p2_added"]["added_item"] = "right_7"
        items["p1_added"]["actual"] = ["mid"]
        items["p2_added"]["actual"] = ["low", "mid"]
        self.assertEqual(score(record)["label"], "upper_boundary_signature")

    def test_execution_failure_is_separate(self):
        record = {"task_id": "x", "variant": "count", "execution_status": "timeout"}
        self.assertEqual(score(record)["label"], "timeout")

    def test_missing_pair_is_invalid_audit(self):
        record = count_record()
        record["cases"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete boundary pair"):
            score(record)

    def test_bool_is_not_numeric_answer(self):
        record = count_record()
        cases_by_id(record)["ordinary"]["actual"] = True
        self.assertEqual(score(record)["label"], "other_failure")


if __name__ == "__main__":
    unittest.main()
