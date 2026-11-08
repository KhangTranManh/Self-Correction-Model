"""Freeze a family sample outside the KodCode semantic shortlist.

This streams projected train columns from the pinned revision, executes no
dataset content, and writes only summary counts plus sampled representatives.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_extended_candidate_sources import flags, text_hash  # noqa: E402
from build_kodcode_family_manifest import SOURCE_DERIVED, UnionFind  # noqa: E402


REVISION = "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5"
COLUMNS = ["question", "question_id", "subset", "metadata",
           "benchmark_similarity", "benchmark_task_id"]
SAMPLE_SIZE = 100


def parse_seed_ids(metadata: object) -> list[str]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata is not a mapping")
    value = metadata.get("seed_ids")
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid seed_ids literal") from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("seed_ids is empty or invalid")
    result = sorted({str(item) for item in value})
    if not result:
        raise ValueError("seed_ids is empty")
    return result


def stable_family_id(subset: str, seeds: list[str]) -> str:
    material = subset + "\0" + "\0".join(sorted(seeds))
    return "kcn_" + hashlib.sha256(material.encode()).hexdigest()[:20]


def build_sample(rows: list[dict], shortlist_ids: set[str]) -> dict:
    uf = UnionFind()
    seeds_by_row: dict[str, list[str]] = {}
    row_by_id: dict[str, dict] = {}
    for row in rows:
        row_id = row["id"]
        if row_id in row_by_id:
            raise ValueError(f"duplicate candidate row ID: {row_id}")
        seeds = row["seed_ids"]
        seeds_by_row[row_id] = seeds
        row_by_id[row_id] = row
        keys = [(row["subset"], seed) for seed in seeds]
        for key in keys[1:]:
            uf.union(keys[0], key)

    groups: dict[tuple[str, tuple[str, str]], list[dict]] = defaultdict(list)
    for row in rows:
        root = uf.find((row["subset"], seeds_by_row[row["id"]][0]))
        groups[(row["subset"], root)].append(row)

    negatives = []
    positive_components = 0
    for (subset, _), members in groups.items():
        if any(member["id"] in shortlist_ids for member in members):
            positive_components += 1
            continue
        seeds = sorted({seed for member in members for seed in member["seed_ids"]})
        family_id = stable_family_id(subset, seeds)
        representative = min(
            members,
            key=lambda row: hashlib.sha256(
                ("p16-kodcode-recall-representative-v1:" + row["id"]).encode()
            ).hexdigest(),
        )
        negatives.append({
            "family_id": family_id,
            "subset": subset,
            "source_derived_subset": subset in SOURCE_DERIVED,
            "seed_ids": seeds,
            "candidate_count": len(members),
            "candidate_ids": sorted(member["id"] for member in members),
            "representative_id": representative["id"],
            "representative_question_sha256": representative["question_sha256"],
            "benchmark_similarity": representative["benchmark_similarity"],
            "sample_order": hashlib.sha256(
                ("p16-kodcode-recall-v1:" + family_id).encode()
            ).hexdigest(),
            "text": representative["text"],
            "review_lane": "pending",
        })
    negatives.sort(key=lambda row: (row["sample_order"], row["family_id"]))
    sample = negatives[:SAMPLE_SIZE]
    if len(sample) != SAMPLE_SIZE:
        raise ValueError(f"only {len(sample)} predicted-negative families available")
    return {
        "broad_components": len(groups),
        "components_with_shortlist_row": positive_components,
        "predicted_negative_components": len(negatives),
        "sample": sample,
    }


def scan(shortlist: dict) -> dict:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("datasets and pyarrow are required for the projected KodCode scan") from exc

    shortlist_ids = {row["id"] for row in shortlist["rows"]}
    dataset = load_dataset(
        "KodCode/KodCode-V1", "default", split="train", streaming=True,
        revision=REVISION,
    ).select_columns(COLUMNS)
    counts = Counter()
    candidates = []
    seen_shortlist_ids: set[str] = set()
    for row in dataset:
        counts["rows"] += 1
        if counts["rows"] % 50000 == 0:
            print(json.dumps({"rows": counts["rows"],
                              "candidates": counts["broad_candidate_rows"]}), flush=True)
        text = row.get("question") or ""
        item_flags = flags(text)
        similarity = row.get("benchmark_similarity") or 0
        if not item_flags["conservative_text_gate"] or similarity >= 0.8:
            continue
        counts["broad_candidate_rows"] += 1
        try:
            seeds = parse_seed_ids(row.get("metadata"))
        except ValueError:
            counts["invalid_seed_rows"] += 1
            continue
        row_id = str(row.get("question_id"))
        candidates.append({
            "id": row_id,
            "subset": str(row.get("subset")),
            "seed_ids": seeds,
            "question_sha256": text_hash(text),
            "benchmark_similarity": similarity,
            "text": text,
        })
        counts["retained_for_family_graph"] += 1
        counts["retained_shortlist_rows"] += row_id in shortlist_ids
        if row_id in shortlist_ids:
            seen_shortlist_ids.add(row_id)
    missing_shortlist_ids = shortlist_ids - seen_shortlist_ids
    if missing_shortlist_ids:
        raise ValueError(
            f"pinned scan missed {len(missing_shortlist_ids)} of "
            f"{len(shortlist_ids)} shortlist IDs"
        )
    counts["shortlist_ids_expected"] = len(shortlist_ids)
    family_result = build_sample(candidates, shortlist_ids)
    return {"counts": dict(counts), **family_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shortlist", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    shortlist_raw = args.shortlist.read_bytes()
    shortlist = json.loads(shortlist_raw)
    result = scan(shortlist)
    document = {
        "schema_version": "p16_kodcode_recall_sample_v1",
        "source_revision": REVISION,
        "shortlist_sha256": hashlib.sha256(shortlist_raw).hexdigest(),
        "sample_rule": "first 100 predicted-negative broad components by p16-kodcode-recall-v1 SHA-256 order",
        "projected_columns": COLUMNS,
        "sample_size": SAMPLE_SIZE,
        **result,
        "interpretation": "Frozen recall-audit sample only; zero tasks are certified or assigned a study role.",
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: document[k] for k in
                      ("broad_components", "components_with_shortlist_row",
                       "predicted_negative_components", "sample_size")},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
