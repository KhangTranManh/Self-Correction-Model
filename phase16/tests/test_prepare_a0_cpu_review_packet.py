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

from close_a0_cpu_sandbox_gates import _load_completions  # noqa: E402
from prepare_a0_cpu_review_packet import build_packet  # noqa: E402


class A0CPUReviewPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((ROOT / "results" /
            "a0_admission_inventory_v1.json").read_text(encoding="utf-8"))
        cls.bank = json.loads((ROOT / "data" / "a0_discovery_v1" /
            "candidate_bank.json").read_text(encoding="utf-8"))
        cls.reviews = [json.loads((ROOT / "results" /
            f"a0_cpu_review_batch{i}_v1.json").read_text(encoding="utf-8"))
            for i in range(1, 10)]
        cls.completions = _load_completions(
            ROOT / "results" / "a0_discovery_run_v2")

    def test_next_packet_follows_verified_chain(self) -> None:
        packet = build_packet(
            self.inventory, self.bank, self.reviews, self.completions, 5)
        self.assertEqual(packet["start_position"], 71)
        self.assertEqual([row["position"] for row in packet["records"]],
                         [71, 73, 74, 76, 77])
        self.assertEqual(packet["remaining_after_packet"], 129)

    def test_completion_identity_mismatch_fails(self) -> None:
        completions = copy.deepcopy(self.completions)
        completions[71]["item_id"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "item_id mismatch"):
            build_packet(
                self.inventory, self.bank, self.reviews, completions, 5)

    def test_nonpositive_count_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "count must be positive"):
            build_packet(
                self.inventory, self.bank, self.reviews, self.completions, 0)

    def test_offset_selects_later_verified_window(self) -> None:
        packet = build_packet(
            self.inventory, self.bank, self.reviews, self.completions, 3, 5)
        self.assertEqual(packet["offset"], 5)
        self.assertEqual([row["position"] for row in packet["records"]],
                         [80, 81, 82])
        self.assertEqual(packet["remaining_after_packet"], 126)

    def test_negative_offset_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "offset must be non-negative"):
            build_packet(
                self.inventory, self.bank, self.reviews, self.completions,
                5, -1)


if __name__ == "__main__":
    unittest.main()
