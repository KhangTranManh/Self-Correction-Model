"""Build balanced KEEP/REVISE preference pairs from verified SFT rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pair(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    if len(messages) < 5 or messages[-1].get("role") != "assistant":
        raise ValueError(f"Malformed SFT row: {row.get('construction_id')}")
    decision = str(row["decision"])
    chosen = messages[-1]
    if decision == "REVISE":
        rejected_content = "<decision>KEEP</decision>"
        preference_reason = "wrong_to_correct_over_wrong_keep"
    elif decision == "KEEP":
        initial_answer = str(messages[-3]["content"])
        rejected_content = (
            "<decision>REVISE</decision>\n"
            f"<answer>{initial_answer}</answer>"
        )
        preference_reason = "correct_keep_over_unnecessary_revision"
    else:
        raise ValueError(f"Unsupported decision {decision!r}")
    return {
        "pair_id": row["construction_id"],
        "source_id": row["source_id"],
        "decision": decision,
        "preference_reason": preference_reason,
        "prompt": messages[:-1],
        "chosen": [chosen],
        "rejected": [{"role": "assistant", "content": rejected_content}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    outputs: dict[str, dict[str, Any]] = {}
    for split, input_path in (("train", args.train_input), ("dev", args.dev_input)):
        source_rows = read_jsonl(input_path)
        pairs = [make_pair(row) for row in source_rows]
        counts = Counter(row["decision"] for row in pairs)
        if counts["KEEP"] != counts["REVISE"]:
            raise ValueError(f"{split} is not action-balanced: {dict(counts)}")
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, pairs)
        outputs[split] = {
            "source": str(input_path),
            "rows": len(pairs),
            "decision_counts": dict(sorted(counts.items())),
            "output": str(output_path),
            "sha256": sha256(output_path),
        }

    manifest = {
        "schema_version": "phase4_preference_pairs_v1",
        "construction": {
            "wrong_initial": "verified REVISE is preferred over KEEP",
            "correct_initial": "KEEP is preferred over an unnecessary same-answer revision",
        },
        "splits": outputs,
        "confirmation_candidates_read": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
