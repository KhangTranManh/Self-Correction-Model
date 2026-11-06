"""Synthetic-only checks of the Phase 16 S0 Boolean-verdict scorer."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from signal_gate import score  # noqa: E402


def fixture() -> dict:
    episodes = []
    for index in range(20):
        prefix = f"e{index:02d}"
        def s2_row(suffix: str, a0: bool = False) -> dict:
            return {"id": f"{prefix}_{suffix}", "a0": a0,
                    "a1": a0, "a1_sham": a0,
                    "a2": a0, "a2_sham": a0}
        episodes.append({
            "id": prefix,
            "family": f"family_{index}",
            "h1_wrong": {"id": f"{prefix}_wrong", "a0": False,
                         "blind": False, "visible": False, "guided": False},
            "h1_correct": {"id": f"{prefix}_correct", "a0": True,
                           "blind": True, "visible": True},
            "positive": [s2_row("positive_0"), s2_row("positive_1")],
            "near_negative": [s2_row("near", True)],
            "unrelated": [s2_row("unrelated", True)],
        })
    return {"schema_version": "p16_signal_only_cpu_v1", "episodes": episodes}


class SignalGateTests(unittest.TestCase):
    def test_all_unchanged_has_no_signal(self):
        result = score(fixture())
        self.assertEqual(result["counts"]["positive"], 40)
        self.assertFalse(result["s1"]["autonomous_follow_up_signal"])
        self.assertFalse(result["s1"]["guided_follow_up_signal"])
        self.assertFalse(result["s2"]["a1"]["follow_up_signal"])
        self.assertFalse(result["s2"]["a2"]["follow_up_signal"])

    def test_distinct_autonomous_and_guided_signals(self):
        data = fixture()
        for index in range(4):
            data["episodes"][index]["h1_wrong"]["visible"] = True
            data["episodes"][index]["h1_wrong"]["guided"] = True
        for index in range(4, 8):
            data["episodes"][index]["h1_wrong"]["guided"] = True
        result = score(data)
        self.assertTrue(result["s1"]["autonomous_follow_up_signal"])
        self.assertTrue(result["s1"]["guided_follow_up_signal"])

    def test_repair_examples_below_blind_do_not_pass(self):
        data = fixture()
        for index in range(4):
            data["episodes"][index]["h1_wrong"]["blind"] = True
            data["episodes"][index]["h1_wrong"]["visible"] = True
        result = score(data)
        self.assertFalse(result["s1"]["autonomous_follow_up_signal"])

    def test_harmful_autonomous_hint_is_not_follow_up_signal(self):
        data = fixture()
        for index in range(4):
            data["episodes"][index]["h1_wrong"]["visible"] = True
        for index in range(3):
            data["episodes"][index]["h1_correct"]["visible"] = False
        result = score(data)["s1"]
        self.assertTrue(result["autonomous_behavioral_hint"])
        self.assertFalse(result["autonomous_follow_up_signal"])

    def test_application_signal_requires_specificity_and_controls(self):
        data = fixture()
        for index in range(4):
            data["episodes"][index]["positive"][0]["a2"] = True
        self.assertTrue(score(data)["s2"]["a2"]["follow_up_signal"])
        for index in range(4):
            data["episodes"][index]["positive"][0]["a2_sham"] = True
        self.assertFalse(score(data)["s2"]["a2"]["follow_up_signal"])
        for index in range(4):
            data["episodes"][index]["positive"][0]["a2_sham"] = False
        for index in range(3):
            data["episodes"][index]["near_negative"][0]["a2"] = False
        damaged = score(data)["s2"]["a2"]
        self.assertTrue(damaged["behavioral_hint"])
        self.assertFalse(damaged["follow_up_signal"])

    def test_rejects_invalid_and_duplicate_rows(self):
        data = fixture()
        data["episodes"][0]["positive"][0]["a0"] = 0
        with self.assertRaisesRegex(ValueError, "Boolean"):
            score(data)
        data = fixture()
        data["episodes"][1]["positive"][0]["id"] = data["episodes"][0]["positive"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score(data)
        data = fixture()
        data["episodes"][0]["positive"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            score(data)

    def test_input_is_not_mutated(self):
        data = fixture()
        original = copy.deepcopy(data)
        score(data)
        self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
