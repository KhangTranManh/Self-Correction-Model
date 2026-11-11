from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_a0_oracle_input_shapes import shape, signature  # noqa: E402
from build_a0_cpu_followup_worklist import build  # noqa: E402
from review_a0_position398_v2 import oracle, reference  # noqa: E402
from track_a0_cpu_admission import canonical_sha256  # noqa: E402


class A0FollowupReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = ROOT / "results"
        read = lambda path: json.loads(path.read_text(encoding="utf-8"))
        cls.inputs = [read(result / "a0_admission_inventory_v1.json"),
                      read(result / "a0_cpu_admission_tracker_v14.json"),
                      read(result / "a0_cpu_sandbox_v2/summary.json"),
                      read(result / "a0_oracle_input_shape_triage_v1.json"),
                      read(result / "a0_cpu_sandbox_v2/position_398_amendment_v2.json")]

    def test_corrected_prime_factor_order_and_fractional_message(self):
        for value, expected in ((100, [5, 50]), (1969, [179, 179]),
                                (-1800.0, [5, 900]), (997, []),
                                (-1800.1, "The number has a decimal part. No Results")):
            self.assertEqual(oracle(value), expected)
            self.assertEqual(reference(value), expected)

    def test_input_shapes_keep_booleans_distinct_from_numbers(self):
        self.assertEqual(signature([[1], True, 1, None]), ("list", "bool", "number", "null"))
        self.assertEqual(shape(1.2), "number")

    def test_worklist_keeps_gate_states_and_contract_hold(self):
        result = build(*self.inputs, source_files={})
        self.assertEqual(result["counts"]["records"], 142)
        self.assertEqual(result["counts"]["correction_admitted"], 0)
        self.assertEqual(result["counts"]["contract_held"], 1)
        self.assertEqual(result["counts"]["human_approval_pending"], 142)
        by_position = {r["position"]: r for r in result["records"]}
        self.assertEqual(by_position[392]["contract_gate"], "held_currency_string_format_unspecified")
        self.assertEqual(by_position[398]["sandbox_evidence"], "position_398_amendment_v2")

    def test_changed_shape_flags_are_never_silently_accepted(self):
        altered = copy.deepcopy(self.inputs)
        triage = altered[3]
        triage["flags"] = triage["flags"][:-1]
        triage.pop("audit_sha256")
        triage["audit_sha256"] = canonical_sha256(triage)
        with self.assertRaisesRegex(RuntimeError, "shape flags require"):
            build(*altered, source_files={})


if __name__ == "__main__":
    unittest.main()
