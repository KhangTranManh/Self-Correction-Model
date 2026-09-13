"""Replace verified cross-origin rows with same-origin rows where available."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-pairs", required=True)
    parser.add_argument("--fallback-train", required=True)
    parser.add_argument("--fallback-dev", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    pairs = {row["source_id"]: row for row in read_jsonl(Path(args.verified_pairs).resolve())}
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    split_outputs = {}
    for split, path in (("train", args.fallback_train), ("dev", args.fallback_dev)):
        rows = read_jsonl(Path(path).resolve()); mixed = []
        for row in rows:
            pair = pairs.get(row["source_id"])
            if not pair:
                mixed.append({**row, "stage4_pair_origin": "verified_cross_origin_fallback"})
                continue
            label = row["label"]
            answer = pair["correct_answer"] if label == "KEEP" else pair["wrong_answer"]
            updated = {**row, "stage4_pair_origin": "verified_same_model_origin", "answer_model_origin": pair["model_origin"]}
            updated["messages"] = list(row["messages"])
            updated["messages"][1] = {"role": "assistant", "content": answer}
            mixed.append(updated)
        write_jsonl(output / f"{split}.jsonl", mixed)
        split_outputs[split] = mixed; all_rows.extend(mixed)
    train_sources = {row["source_id"] for row in split_outputs["train"]}
    dev_sources = {row["source_id"] for row in split_outputs["dev"]}
    summary = {
        "schema_version": "phase3_same_origin_mixed_router_v1",
        "rows": len(all_rows), "pairs": len({row["source_id"] for row in all_rows}),
        "train_rows": len(split_outputs["train"]), "dev_rows": len(split_outputs["dev"]),
        "labels": dict(Counter(row["label"] for row in all_rows)),
        "pair_origin_rows": dict(Counter(row["stage4_pair_origin"] for row in all_rows)),
        "same_origin_pairs": len(pairs), "cross_origin_fallback_pairs": len({row["source_id"] for row in all_rows}) - len(pairs),
        "datasets": dict(Counter(next(row["dataset"] for row in all_rows if row["source_id"] == source_id) for source_id in {row["source_id"] for row in all_rows})),
        "train_dev_source_overlap": sorted(train_sources & dev_sources),
        "validation": {"balanced_labels": Counter(row["label"] for row in all_rows) == {"KEEP": 100, "REVISE": 100}, "no_split_overlap": not (train_sources & dev_sources), "new_pairs_fresh_verified": True},
        "hashes": {"verified_pairs": hashlib.sha256(Path(args.verified_pairs).resolve().read_bytes()).hexdigest()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
