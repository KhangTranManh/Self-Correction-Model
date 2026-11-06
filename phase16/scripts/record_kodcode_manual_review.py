"""Record the preregistered-order manual review of the first 50 shortlist rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ELIGIBLE = {0, 1, 2, 4, 6, 8, 9, 12, 18, 19, 22, 23, 24, 25, 26,
            32, 33, 34, 36, 39, 40, 42, 46, 49}
REASONS = {
    3: "summarizes consecutive runs; does not select/count/sum by key range",
    5: "malformed/nested scaffold makes the target function ambiguous",
    7: "returns index runs whose contents are prime, not elements keyed by [L,R]",
    10: "multi-operation sequence API; range mention is incidental",
    11: "license-key formatting; range mention is incidental",
    13: "hard-coded 100..200 conflicts with general start/end scaffold",
    14: "median-stream task; range mention is incidental",
    15: "simple-statements API; range mention is incidental",
    16: "multi-operation sequence API; range mention is incidental",
    17: "mode-dependent prime/palindrome contract is outside one frozen variant",
    20: "elevator simulation; range mention is incidental",
    21: "generator demonstration with ambiguous scored output",
    27: "list rotation; range mention is incidental",
    28: "median-maintenance task; range mention is incidental",
    29: "oracle dispute: stated output 9 conflicts with shown totient sum 10",
    30: "single missing-number task, not count/sum/selection of qualifying elements",
    31: "Roman-numeral conversion; range mention is a constraint",
    35: "undefined 'good number' predicate; contract is incomplete",
    37: "shared-memory task; range mention is incidental",
    38: "returns median after filtering, not count/sum/selection",
    41: "two-sum task; range mention is incidental",
    43: "Roman-numeral conversion; range mention is a constraint",
    44: "single missing-number task, not count/sum/selection",
    45: "balanced-tree validation; range mention is incidental",
    47: "random array generation; output is not the named operation",
    48: "returns missing intervals, not qualifying element IDs",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shortlist", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    raw = args.shortlist.read_bytes()
    source = json.loads(raw)
    sample = source["rows"][:50]
    if len(sample) != 50 or set(REASONS) | ELIGIBLE != set(range(50)):
        raise ValueError("manual labels do not cover the frozen first 50 rows")
    rows = []
    for index, row in enumerate(sample):
        eligible = index in ELIGIBLE
        rows.append({
            "review_index": index, "id": row["id"], "subset": row["subset"],
            "contract_text_eligible": eligible,
            "reason": ("explicit inclusive range and direct selection/count/sum; oracle still unvalidated"
                       if eligible else REASONS[index]),
            "question_sha256": row["question_sha256"],
        })
    report = {
        "schema_version": "p16_kodcode_manual_contract_review_v1",
        "shortlist_sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_sample_rule": "first 50 rows in hashed review_order",
        "counts": {"reviewed": 50, "contract_text_eligible": len(ELIGIBLE),
                   "rejected": 50 - len(ELIGIBLE), "oracle_validated": 0},
        "rows": rows,
        "limit": "Manual text review only; tests, family independence and oracle are not certified.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
