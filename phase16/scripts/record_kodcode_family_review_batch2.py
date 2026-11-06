"""Record manual contract review for frozen KodCode families 20 through 39."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LABELS = {
    "kc_9f0e1beea36207a478f0": ("reject", "none", "not_applicable",
        "largest-BST-subtree task; range language is only a value constraint"),
    "kc_c0371fdb56f10d96eb0c": ("reject", "none", "not_applicable",
        "multi-method class contract rather than one locked function"),
    "kc_14257ddcaa70a9f5afc6": ("reject", "none", "not_applicable",
        "Fibonacci output is ambiguous about the duplicated value 1"),
    "kc_06226b84f2928a6835dd": ("applicable_candidate", "count", "explicit_collection",
        "explicit function counts distinct user IDs from records in inclusive timestamp bounds"),
    "kc_569d8dfac5cb9c9c8836": ("reject", "none", "not_applicable",
        "returns one Boolean membership result rather than count, sum, or selected elements"),
    "kc_4832b255505000ec819a": ("applicable_candidate", "count", "implicit_integer_range",
        "explicit function counts numbers satisfying a defined predicate in inclusive bounds"),
    "kc_b5e0b9f2a1af1ddb702c": ("applicable_candidate", "sum", "explicit_collection",
        "named function sums BST node values whose keys are within inclusive bounds"),
    "kc_ceda7867e16cb3046bae": ("reject", "none", "not_applicable",
        "standard-I/O range-update/query system rather than one locked function"),
    "kc_f33e3c0b7091d041af56": ("reject", "none", "not_applicable",
        "requires two target functions and mixes validation with summation"),
    "kc_7750d345557c5a9f0aa7": ("reject", "none", "not_applicable",
        "tests whether a derived aggregate falls in a range; does not filter elements by key"),
    "kc_5add8303acbb22259c5f": ("reject", "none", "not_applicable",
        "returns per-city averages, outside the locked count/sum/select output"),
    "kc_27bd0e71847897f96d2a": ("reject", "none", "not_applicable",
        "multi-operation mutable class and processor contract"),
    "kc_fe76e803a689fd9d6185": ("reject", "none", "not_applicable",
        "asks existence of an interval of a given length, not filtering by value bounds"),
    "kc_ebf36eb08b77759f85dc": ("reject", "none", "not_applicable",
        "standard-I/O multi-operation update/prefix-query system"),
    "kc_ae33fb860471e7b79772": ("reject", "none", "not_applicable",
        "BST construction/report task; range language is only a key constraint"),
    "kc_fd4aee9e1f872298933a": ("applicable_candidate", "select", "implicit_integer_range",
        "named function returns all qualifying integers in inclusive bounds"),
    "kc_b77105c58f47d478a681": ("reject", "none", "not_applicable",
        "tree path-string optimization; range language is only a value constraint"),
    "kc_927665c3e7c9bf59a74f": ("applicable_candidate", "select", "implicit_integer_range",
        "explicit function returns all predicate-matching integers in inclusive bounds"),
    "kc_cb3035fb2a42a9617e1f": ("reject", "none", "not_applicable",
        "color interpolation; range language is only component validation"),
    "kc_19c0340ddc16938ae713": ("reject", "none", "not_applicable",
        "always sums a nonempty reordered integer interval, so the locked empty-output case cannot be constructed"),
}


def build(manifest_raw: bytes, shortlist_raw: bytes) -> dict:
    manifest = json.loads(manifest_raw)
    shortlist = json.loads(shortlist_raw)
    families = manifest["families"][20:40]
    if len(families) != 20 or {row["family_id"] for row in families} != set(LABELS):
        raise ValueError("frozen family slice 20:40 no longer matches labels")
    candidates = {row["id"]: row for row in shortlist["rows"]}
    rows = []
    for index, family in enumerate(families, start=20):
        decision, variant, universe_mode, reason = LABELS[family["family_id"]]
        representative = candidates[family["representative_id"]]
        if representative["question_sha256"] != family["representative_question_sha256"]:
            raise ValueError("representative hash mismatch")
        rows.append({
            "review_index": index, "family_id": family["family_id"],
            "subset": family["subset"], "seed_ids": family["seed_ids"],
            "representative_id": family["representative_id"],
            "question_sha256": representative["question_sha256"],
            "contract_decision": decision, "variant": variant,
            "universe_mode": universe_mode, "reason": reason,
            "oracle_status": "pending", "history_status": "pending",
            "near_duplicate_status": "pending", "sandbox_status": "pending",
        })
    decisions = {name: sum(row["contract_decision"] == name for row in rows)
                 for name in ("applicable_candidate", "near_negative_candidate", "reject")}
    return {
        "schema_version": "p16_kodcode_family_contract_review_v1",
        "family_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "shortlist_sha256": hashlib.sha256(shortlist_raw).hexdigest(),
        "frozen_batch_rule": "families 20:40 in review_order",
        "counts": {"reviewed": len(rows), **decisions, "fully_certified": 0},
        "rows": rows,
        "limitations": "Manual text-contract review; oracle/history/near-duplicate/sandbox gates remain separate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("shortlist", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = build(args.manifest.read_bytes(), args.shortlist.read_bytes())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()

