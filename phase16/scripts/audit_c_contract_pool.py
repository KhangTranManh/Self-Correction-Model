"""Conservative source-only audit of TACO C-upper-inclusive candidates.

No model output or dataset solution is executed. Passing a text gate is NOT
task eligibility: a human must verify the contract and construct an oracle.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

EXPLICIT_BOUNDS = re.compile(
    r"\b(?:lower|left|start|lo|l)\b.{0,100}\b(?:upper|right|end|hi|r)\b"
    r"|\b(?:upper|right|end|hi|r)\b.{0,100}\b(?:lower|left|start|lo|l)\b",
    re.I | re.S,
)
CLOSED = re.compile(
    r"\b(?:inclusiv(?:e|ely)|closed\s+(?:interval|range))\b"
    r"|\b(?:less\s+than\s+or\s+equal|at\s+most|not\s+greater\s+than)\b"
    r"|\[\s*[A-Za-z]\w*\s*,\s*[A-Za-z]\w*\s*\]",
    re.I,
)
AGGREGATE = re.compile(r"\b(?:count|number|sum|total|select|filter|collect)\b", re.I)
SEQUENCE = re.compile(r"\b(?:array|list|elements?|items?|numbers?|values?)\b", re.I)


def audit(raw_dir: Path, metadata_path: Path, prefilter_path: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to audit raw TACO parquet shards") from exc

    metadata = [json.loads(line) for line in metadata_path.open(encoding="utf-8")]
    prefilter = json.loads(prefilter_path.read_text(encoding="utf-8"))
    candidates = {row["id"] for row in prefilter["candidate_ids"]}
    if len(candidates) != prefilter["candidate_count"]:
        raise ValueError("prefilter ids are not unique")
    rows = []
    counts = Counter()
    index = 0
    for shard in sorted((raw_dir / "ALL").glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(columns=["question"]):
            for row in batch.to_pylist():
                if index >= len(metadata):
                    raise ValueError("metadata shorter than raw data")
                item = metadata[index]
                index += 1
                if item["id"] not in candidates:
                    continue
                question = row["question"] or ""
                flags = {
                    "explicit_bounds": bool(EXPLICIT_BOUNDS.search(question)),
                    "closed_language": bool(CLOSED.search(question)),
                    "aggregate": bool(AGGREGATE.search(question)),
                    "sequence": bool(SEQUENCE.search(question)),
                }
                for key, value in flags.items():
                    counts[key] += value
                strict = all(flags.values())
                counts["all_text_gates"] += strict
                rows.append({"id": item["id"], "source": item["source"],
                             "flags": flags, "needs_manual_contract_audit": strict})
    if index != len(metadata) or len(rows) != len(candidates):
        raise ValueError("raw/metadata/prefilter row mismatch")
    return {
        "schema_version": "p16_c_contract_text_audit_v1",
        "source": "TACO train, pinned in taco_inventory.py",
        "prefilter_count": len(candidates),
        "counts": dict(counts),
        "rows": rows,
        "limit": "Text gates only; no contract, oracle, family or 300-task validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("prefilter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = audit(args.raw_dir, args.metadata, args.prefilter)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prefilter_count": result["prefilter_count"],
                      "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
