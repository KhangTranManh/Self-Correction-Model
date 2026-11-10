from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_a0_cpu_review_batch6 import canonical_sha256  # noqa: E402


class A0CPUCommonBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = json.loads((ROOT / "results" /
            "a0_admission_inventory_v1.json").read_text(encoding="utf-8"))
        cls.bank = json.loads((ROOT / "data" / "a0_discovery_v1" /
            "candidate_bank.json").read_text(encoding="utf-8"))

    def test_every_common_batch_rebuilds_exactly(self) -> None:
        paths = sorted((ROOT / "results").glob("a0_cpu_review_batch*_v1.json"),
                       key=lambda p: int(re.search(r"batch(\d+)", p.name).group(1)))
        checked = 0
        for path in paths:
            number = int(re.search(r"batch(\d+)", path.name).group(1))
            if number < 11:
                continue
            with self.subTest(batch=number):
                module = importlib.import_module(f"audit_a0_cpu_review_batch{number}")
                previous = json.loads((ROOT / "results" /
                    f"a0_cpu_review_batch{number - 1}_v1.json"
                ).read_text(encoding="utf-8"))
                saved = json.loads(path.read_text(encoding="utf-8"))
                rebuilt = module.build(self.inventory, self.bank, previous)
                self.assertEqual(rebuilt, saved)
                claimed = rebuilt["review_sha256"]
                without_hash = copy.deepcopy(rebuilt)
                without_hash.pop("review_sha256")
                self.assertEqual(claimed, canonical_sha256(without_hash))
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
