"""Convert the immutable two-stage manifest into decision-router messages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
NEUTRAL_REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "two_stage_selective_repair" / "frozen_router_benchmark.jsonl"))
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    source_rows = read_jsonl(manifest)
    if len(source_rows) != 200 or len({row["source_id"] for row in source_rows}) != 200:
        raise RuntimeError("Frozen manifest must contain exactly 200 unique sources")
    rows = []
    for source in source_rows:
        expected = str(source["expected_decision"])
        if expected not in {"KEEP", "REVISE"}:
            raise RuntimeError(f"Invalid frozen label: {source['source_id']}")
        rows.append(
            {
                "construction_id": f"frozen_router::{source['source_id']}",
                "source_id": source["source_id"],
                "split": "frozen",
                "dataset": source["dataset"],
                "domain": source["domain"],
                "bucket": "initial_correct" if source["initial_correct"] else "initial_wrong",
                "neutral_template_id": "canonical_frozen_neutral_review",
                "decision": expected,
                "messages": [
                    {"role": "user", "content": source["task_prompt"]},
                    {"role": "assistant", "content": source["initial_answer"]},
                    {"role": "user", "content": NEUTRAL_REVIEW},
                    {"role": "assistant", "content": f"<decision>{expected}</decision>"},
                ],
            }
        )
    write_jsonl(output, rows)
    print(json.dumps({
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(rows),
        "keep": sum(row["decision"] == "KEEP" for row in rows),
        "revise": sum(row["decision"] == "REVISE" for row in rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
