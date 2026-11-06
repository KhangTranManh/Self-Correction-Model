"""Phase 16 CPU engineering smoke on Colab; no model generation.

Run only with the local score_boundary.py uploaded to /content/phase16.
All candidate functions below are trusted, hand-written fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys


ROOT = Path("/content/phase16")
sys.path.insert(0, str(ROOT))
from score_boundary import score  # noqa: E402


def count_values(values, lower, upper, *, inclusive):
    return sum(lower <= key <= upper if inclusive else lower <= key < upper for key in values)


def sum_records(records, lower, upper, *, inclusive):
    return sum(amount for key, amount in records
               if (lower <= key <= upper if inclusive else lower <= key < upper))


def select_records(records, lower, upper, *, inclusive):
    return [item_id for key, item_id in records
            if (lower <= key <= upper if inclusive else lower <= key < upper)]


CASES = {
    "count": [
        ("ordinary", "ordinary", [2], 1, 5, None, None),
        ("lower", "lower", [1], 1, 5, None, None),
        ("outside", "outside", [0, 6], 1, 5, None, None),
        ("empty", "empty", [], 1, 5, None, None),
        ("single_point", "single_point", [5], 5, 5, None, None),
        ("duplicate", "duplicate", [5, 5], 1, 5, None, None),
        ("p1_anchor", "upper_anchor", [2], 1, 5, "p1", None),
        ("p1_added", "upper_added", [2, 5], 1, 5, "p1", None),
        ("p2_anchor", "upper_anchor", [1, 3], 1, 7, "p2", None),
        ("p2_added", "upper_added", [1, 3, 7], 1, 7, "p2", None),
    ],
    "sum": [
        ("ordinary", "ordinary", [(2, 3)], 1, 5, None, None),
        ("lower", "lower", [(1, 4)], 1, 5, None, None),
        ("outside", "outside", [(0, 9), (6, 9)], 1, 5, None, None),
        ("empty", "empty", [], 1, 5, None, None),
        ("single_point", "single_point", [(5, 6)], 5, 5, None, None),
        ("duplicate", "duplicate", [(5, 6), (5, 2)], 1, 5, None, None),
        ("p1_anchor", "upper_anchor", [(2, 3)], 1, 5, "p1", None),
        ("p1_added", "upper_added", [(2, 3), (5, 6)], 1, 5, "p1", 6),
        ("p2_anchor", "upper_anchor", [(1, 4), (3, 2)], 1, 7, "p2", None),
        ("p2_added", "upper_added", [(1, 4), (3, 2), (7, 5)], 1, 7, "p2", 5),
    ],
    "select": [
        ("ordinary", "ordinary", [(2, "mid")], 1, 5, None, None),
        ("lower", "lower", [(1, "low")], 1, 5, None, None),
        ("outside", "outside", [(0, "out0"), (6, "out6")], 1, 5, None, None),
        ("empty", "empty", [], 1, 5, None, None),
        ("single_point", "single_point", [(5, "right")], 5, 5, None, None),
        ("duplicate", "duplicate", [(5, "right_a"), (5, "right_b")], 1, 5, None, None),
        ("p1_anchor", "upper_anchor", [(2, "mid")], 1, 5, "p1", None),
        ("p1_added", "upper_added", [(2, "mid"), (5, "right_5")], 1, 5, "p1", "right_5"),
        ("p2_anchor", "upper_anchor", [(1, "low"), (3, "mid")], 1, 7, "p2", None),
        ("p2_added", "upper_added", [(1, "low"), (3, "mid"), (7, "right_7")], 1, 7, "p2", "right_7"),
    ],
}
FUNCTIONS = {
    "count": count_values,
    "sum": sum_records,
    "select": select_records,
}


def transcript(variant: str, *, inclusive: bool) -> dict:
    cases = []
    function = FUNCTIONS[variant]
    for case_id, kind, values, lower, upper, pair_id, added in CASES[variant]:
        case = {
            "case_id": case_id,
            "kind": kind,
            "expected": function(values, lower, upper, inclusive=True),
            "actual": function(values, lower, upper, inclusive=inclusive),
        }
        if pair_id is not None:
            case.update(pair_id=pair_id, upper=upper)
        if kind == "upper_added" and variant == "sum":
            case["added_value"] = added
        if kind == "upper_added" and variant == "select":
            case["added_item"] = added
        cases.append(case)
    return {
        "task_id": f"engineering_{variant}_{'correct' if inclusive else 'exclusive'}",
        "variant": variant,
        "execution_status": "ok",
        "cases": cases,
    }


def main() -> None:
    results = {}
    for variant in ("count", "sum", "select"):
        correct = score(transcript(variant, inclusive=True))
        exclusive = score(transcript(variant, inclusive=False))
        if correct["label"] != "correct" or not correct["full_correct"]:
            raise AssertionError(f"Oracle fixture failed: {variant}")
        if exclusive["label"] != "upper_boundary_signature" or exclusive["signature_pairs"] != 2:
            raise AssertionError(f"Exclusive fixture failed: {variant}")
        results[variant] = {"correct": correct, "exclusive_upper": exclusive}
    result = {
        "schema_version": "phase16_colab_definition_smoke_v1",
        "purpose": "trusted_fixture_mechanics_only",
        "model_generation": False,
        "gpu_used": False,
        "python": platform.python_version(),
        "scorer_sha256": hashlib.sha256((ROOT / "score_boundary.py").read_bytes()).hexdigest(),
        "variants": results,
    }
    output = ROOT / "definition_smoke.json"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
