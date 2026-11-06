"""Conservative semantic-text shortlist for KodCode C-upper-inclusive review.

This is deliberately a triage heuristic. Every retained item still requires
human contract review, family deduplication, test inspection and a new oracle.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


INCLUDE_SUBSETS = {"Algorithm", "Data_Structure", "Docs", "Evol", "Filter",
                   "Package", "Prefill"}
DIRECT = re.compile(
    r"(?:return|find|generate|filter|select|calculate|compute|count|sum|list)"
    r".{0,180}(?:all|the number of|the sum of|values?|elements?|nodes?|integers?|numbers?)"
    r".{0,180}(?:inclusive range|range\s*\[|between|within the range|from .{1,30} to)"
    r"|(?:inclusive range|range\s*\[|between|within the range|from .{1,30} to)"
    r".{0,180}(?:return|find|generate|filter|select|count|sum|list)",
    re.I | re.S,
)
REJECT = re.compile(
    r"\b(?:subarrays?|sub-arrays?|substrings?|paths?|grids?|matri(?:x|ces)|"
    r"queries|query|segment tree|fenwick|overlap(?:ping)?|schedule|meeting rooms?|"
    r"maximum|minim(?:um|ize)|number of ways|random|simpson|integrat(?:e|ion)|"
    r"bisection|plot|write .{0,40} file|filename|async|network|dataframe)\b",
    re.I,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-source-derived", action="store_true",
                        help="Include TACO/APPS/CodeContests/Codeforces/LeetCode subsets")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    candidates = json.loads(args.audit.read_text(encoding="utf-8"))["result"]["candidates"]
    rows = []
    counts = Counter()
    for row in candidates:
        counts["text_candidates"] += 1
        if not row["flags"]["conservative_text_gate"]:
            continue
        if (row.get("benchmark_similarity") or 0) >= 0.8:
            continue
        if not args.include_source_derived and row.get("subset") not in INCLUDE_SUBSETS:
            continue
        counts["independence_preflight"] += 1
        text = row["text"]
        direct = bool(DIRECT.search(text))
        rejected_surface = bool(REJECT.search(text))
        if not direct or rejected_surface:
            continue
        seed_ids = ast.literal_eval(row["metadata"]["seed_ids"])
        family = f"{row['subset']}:" + ",".join(sorted(map(str, seed_ids)))
        rows.append({
            "id": row["id"], "subset": row["subset"], "seed_family_provisional": family,
            "question_sha256": row["source_text_sha256"],
            "benchmark_similarity": row.get("benchmark_similarity"),
            "review_order": hashlib.sha256(("p16-kodcode-review:" + row["id"]).encode()).hexdigest(),
            "status": "needs_manual_contract_and_oracle_review",
            "text": text,
        })
        counts["semantic_shortlist"] += 1
    rows.sort(key=lambda x: x["review_order"])
    counts.update({f"shortlist_{k}": v for k, v in Counter(r["subset"] for r in rows).items()})
    result = {
        "schema_version": "p16_kodcode_c_shortlist_v2",
        "source_audit": str(args.audit), "counts": dict(counts), "rows": rows,
        "include_source_derived": args.include_source_derived,
        "interpretation": "Heuristic shortlist only. Certified task count remains zero.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
