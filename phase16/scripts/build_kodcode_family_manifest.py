"""Collapse KodCode candidates into connected seed-family components.

The output is a review manifest, not a study pool. It assigns no study role
and contains no oracle-validity claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SOURCE_DERIVED = {"Taco", "Codeforces", "Code_Contests", "Leetcode", "Apps"}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, item: tuple[str, str]) -> tuple[str, str]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: tuple[str, str], second: tuple[str, str]) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def build(document: dict) -> dict:
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError("shortlist rows missing")
    uf = UnionFind()
    seeds_by_id: dict[str, list[str]] = {}
    for row in rows:
        subset = row["subset"]
        prefix, raw = row["seed_family_provisional"].split(":", 1)
        if prefix != subset:
            raise ValueError("subset/family mismatch")
        seeds = [value for value in raw.split(",") if value]
        if not seeds:
            raise ValueError("empty seed family")
        seeds_by_id[row["id"]] = seeds
        keys = [(subset, seed) for seed in seeds]
        for key in keys[1:]:
            uf.union(keys[0], key)
    groups: dict[tuple[str, tuple[str, str]], list[dict]] = defaultdict(list)
    for row in rows:
        subset = row["subset"]
        root = uf.find((subset, seeds_by_id[row["id"]][0]))
        groups[(subset, root)].append(row)
    families = []
    for (subset, _), members in groups.items():
        all_seeds = sorted({seed for row in members for seed in seeds_by_id[row["id"]]})
        family_material = subset + "\0" + "\0".join(all_seeds)
        family_id = "kc_" + hashlib.sha256(family_material.encode()).hexdigest()[:20]
        ordered = sorted(members, key=lambda row: (row["review_order"], row["id"]))
        representative = ordered[0]
        families.append({
            "family_id": family_id, "subset": subset,
            "source_derived_subset": subset in SOURCE_DERIVED,
            "seed_ids": all_seeds, "candidate_count": len(members),
            "candidate_ids": [row["id"] for row in ordered],
            "representative_id": representative["id"],
            "representative_question_sha256": representative["question_sha256"],
            "review_order": hashlib.sha256(("p16-family-review:" + family_id).encode()).hexdigest(),
            "contract_status": "pending", "oracle_status": "pending",
            "history_status": "pending", "near_duplicate_status": "pending",
            "role": "unassigned",
        })
    families.sort(key=lambda row: row["review_order"])
    counts = Counter(row["subset"] for row in families)
    return {
        "schema_version": "p16_kodcode_family_review_manifest_v1",
        "source_revision": "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5",
        "shortlist_schema": document.get("schema_version"),
        "include_source_derived": document.get("include_source_derived"),
        "counts": {"candidate_rows": len(rows), "families": len(families),
                   "source_derived_families": sum(r["source_derived_subset"] for r in families),
                   "other_families": sum(not r["source_derived_subset"] for r in families),
                   "by_subset": dict(sorted(counts.items()))},
        "families": families,
        "interpretation": "Review inventory only; zero roles or oracle eligibility assigned.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shortlist", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = build(json.loads(args.shortlist.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
