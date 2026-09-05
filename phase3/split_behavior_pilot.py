"""Split the 150-row behavior pilot into source-disjoint 120 train / 30 dev.

Counterfactual pair members are indivisible groups, so the same source never
crosses the train/dev boundary. The split is deterministic with seed 314159.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "behavior" / "construction_pilot" / "pilot_behavior_rows.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "behavior" / "mini_train"
SEED = 314159
CONDITIONS = (
    "preserve_neutral",
    "preserve_false_feedback",
    "repair_neutral",
    "repair_true_feedback",
    "normal_solve",
    "regression_recovery",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(context: str, identifier: str) -> str:
    return hashlib.sha256(f"{SEED}|{context}|{identifier}".encode()).hexdigest()


def choose(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    domain: str,
    count: int,
    context: str,
    dataset: str | None = None,
    paired: bool | None = None,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["condition"] == condition
        and row["domain"] == domain
        and (dataset is None or row["dataset"] == dataset)
        and (paired is None or bool(row["is_counterfactual_pair"]) is paired)
    ]
    candidates.sort(key=lambda row: (rank(context, row["selection_id"]), row["selection_id"]))
    if len(candidates) < count:
        raise RuntimeError(f"Insufficient candidates for {context}: {len(candidates)} < {count}")
    return candidates[:count]


def split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev_ids: set[str] = set()

    # Keep two complete counterfactual pairs per family in dev: one math and one
    # code. Their four source groups contribute 8 of the 30 dev rows.
    for family, conditions in (
        ("preserve", {"preserve_neutral", "preserve_false_feedback"}),
        ("repair", {"repair_neutral", "repair_true_feedback"}),
    ):
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["pair_id"] and row["condition"] in conditions:
                groups[row["pair_id"]].append(row)
        for domain in ("math", "code"):
            candidates = [
                (pair_id, members)
                for pair_id, members in groups.items()
                if len(members) == 2
                and {member["condition"] for member in members} == conditions
                and {member["domain"] for member in members} == {domain}
            ]
            candidates.sort(key=lambda item: (rank(f"dev_{family}_pair_{domain}", item[0]), item[0]))
            if not candidates:
                raise RuntimeError(f"No complete {family}/{domain} pilot pair")
            dev_ids.update(member["construction_id"] for member in candidates[0][1])

    # Each review condition needs 5 dev rows. Pair selection above contributes
    # one math and one code, so add one math + two code unpaired rows.
    for condition in (
        "preserve_neutral",
        "preserve_false_feedback",
        "repair_neutral",
        "repair_true_feedback",
    ):
        available = [row for row in rows if row["construction_id"] not in dev_ids]
        dev_ids.update(
            row["construction_id"]
            for row in choose(
                available,
                condition=condition,
                domain="math",
                count=1,
                context=f"dev_{condition}_math_single",
                paired=False,
            )
        )
        available = [row for row in rows if row["construction_id"] not in dev_ids]
        # Prefer one MBPP and one APPS code row when both are available.
        for dataset in ("mbpp", "apps"):
            dev_ids.update(
                row["construction_id"]
                for row in choose(
                    available,
                    condition=condition,
                    domain="code",
                    dataset=dataset,
                    count=1,
                    context=f"dev_{condition}_{dataset}_single",
                    paired=False,
                )
            )
            available = [row for row in rows if row["construction_id"] not in dev_ids]

    # Fresh conditions use two math + three code dev rows. Select one MBPP and
    # two APPS for code so executable behavior is represented across sources.
    for condition in ("normal_solve", "regression_recovery"):
        available = [row for row in rows if row["construction_id"] not in dev_ids]
        dev_ids.update(
            row["construction_id"]
            for row in choose(
                available,
                condition=condition,
                domain="math",
                count=2,
                context=f"dev_{condition}_math",
            )
        )
        available = [row for row in rows if row["construction_id"] not in dev_ids]
        for dataset, count in (("mbpp", 1), ("apps", 2)):
            dev_ids.update(
                row["construction_id"]
                for row in choose(
                    available,
                    condition=condition,
                    domain="code",
                    dataset=dataset,
                    count=count,
                    context=f"dev_{condition}_{dataset}",
                )
            )
            available = [row for row in rows if row["construction_id"] not in dev_ids]

    train = [{**row, "pilot_split": "train"} for row in rows if row["construction_id"] not in dev_ids]
    dev = [{**row, "pilot_split": "dev"} for row in rows if row["construction_id"] in dev_ids]
    train.sort(key=lambda row: row["construction_id"])
    dev.sort(key=lambda row: row["construction_id"])
    return train, dev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(input_path)
    train, dev = split(rows)

    errors = []
    train_sources = {row["source_id"] for row in train}
    dev_sources = {row["source_id"] for row in dev}
    overlap = sorted(train_sources & dev_sources)
    if overlap:
        errors.append(f"source leakage across train/dev: {overlap}")
    for name, split_rows, expected_per_condition in (
        ("train", train, 20),
        ("dev", dev, 5),
    ):
        counts = Counter(row["condition"] for row in split_rows)
        if any(counts[condition] != expected_per_condition for condition in CONDITIONS):
            errors.append(f"{name} condition counts mismatch: {dict(counts)}")
    if len(train) != 120 or len(dev) != 30:
        errors.append(f"split size mismatch: train={len(train)}, dev={len(dev)}")
    pair_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in train + dev:
        if row["pair_id"]:
            pair_splits[row["pair_id"]].add(row["pilot_split"])
    split_pairs = sorted(pair_id for pair_id, values in pair_splits.items() if len(values) > 1)
    if split_pairs:
        errors.append(f"counterfactual pair leakage: {split_pairs}")
    if errors:
        raise RuntimeError("; ".join(errors))

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    summary_path = output_dir / "split_summary.json"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    summary = {
        "seed": SEED,
        "algorithm": "source-group-aware SHA-256 deterministic split",
        "input": {"path": str(input_path), "sha256": file_hash(input_path)},
        "train": {
            "rows": len(train),
            "unique_sources": len(train_sources),
            "counts_by_condition": dict(sorted(Counter(row["condition"] for row in train).items())),
            "counts_by_domain": dict(sorted(Counter(row["domain"] for row in train).items())),
            "counts_by_dataset": dict(sorted(Counter(row["dataset"] for row in train).items())),
        },
        "dev": {
            "rows": len(dev),
            "unique_sources": len(dev_sources),
            "counts_by_condition": dict(sorted(Counter(row["condition"] for row in dev).items())),
            "counts_by_domain": dict(sorted(Counter(row["domain"] for row in dev).items())),
            "counts_by_dataset": dict(sorted(Counter(row["dataset"] for row in dev).items())),
            "selection_ids": [row["selection_id"] for row in dev],
        },
        "train_dev_source_overlap": overlap,
        "counterfactual_pairs_crossing_split": split_pairs,
        "validation_errors": errors,
    }
    write_json(summary_path, summary)
    print(json.dumps({"train": summary["train"], "dev": summary["dev"], "validation_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
