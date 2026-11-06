"""Record manual contract review for the first 20 frozen KodCode families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# family_id: (decision, variant, universe_mode, reason)
LABELS = {
    "kc_2c3fb3a26c845690b192": ("applicable_candidate", "select", "implicit_integer_range",
        "explicit inclusive bounds and returns all integers satisfying a defined predicate"),
    "kc_4171df85676ca489b198": ("reject", "none", "not_applicable",
        "returns only the first qualifying scalar, not the selected set required by the locked select variant"),
    "kc_7a77cafe00a01d3be0a0": ("applicable_candidate", "count", "implicit_integer_range",
        "direct count over explicit inclusive integer bounds"),
    "kc_59ddcf96bed7d921e327": ("applicable_candidate", "select", "implicit_integer_range",
        "explicit function and inclusive bounds return all qualifying primes"),
    "kc_d6ef57eff1aa834cacf8": ("reject", "none", "not_applicable",
        "standard-I/O multi-query program has no single explicit Python function contract"),
    "kc_045d53717791b2d79a3a": ("reject", "none", "not_applicable",
        "tree row maximum; range language is only a value constraint"),
    "kc_794cb9c01bb5d0496ecd": ("reject", "none", "not_applicable",
        "array rotation; range language is only a constraint"),
    "kc_772255d94506ca5bf925": ("applicable_candidate", "sum", "implicit_integer_range",
        "explicit function sums all primes in inclusive integer bounds"),
    "kc_3fb5ff0a1a9ce8002b13": ("reject", "none", "not_applicable",
        "k-th largest search; range language is only an input constraint"),
    "kc_2770df63debd7f6420dc": ("reject", "none", "not_applicable",
        "target operation fits but the displayed function scaffold is syntactically malformed"),
    "kc_1b962a30388143c357fb": ("reject", "none", "not_applicable",
        "multi-operation range-update/query API cannot isolate the locked single function"),
    "kc_15ace8ed10efc8a450b9": ("reject", "none", "not_applicable",
        "function name/signature is not explicitly specified in the source contract"),
    "kc_84632a3c91d9e2d89678": ("reject", "none", "not_applicable",
        "standard-I/O sequence of range updates, not a single count/sum/select function"),
    "kc_7ade91a5f19fb0f41769": ("reject", "none", "not_applicable",
        "standard-I/O problem has no explicit Python function signature"),
    "kc_60463e25cc5178ff79d9": ("applicable_candidate", "count", "explicit_collection",
        "explicit function counts list elements within stated inclusive value bounds"),
    "kc_3f82fe63a265a47628eb": ("reject", "none", "not_applicable",
        "odd/even sorting; range language is only a constraint"),
    "kc_ae2c9b33d997f712ac68": ("near_negative_candidate", "none", "not_applicable",
        "structurally matched range selection but the upper endpoint is explicitly exclusive"),
    "kc_a470385b5b81b3e33d06": ("reject", "none", "not_applicable",
        "word-ladder shortest path; range language is only a constraint"),
    "kc_963b8d6cba97b2b685f3": ("applicable_candidate", "select", "implicit_integer_range",
        "explicit function returns all qualifying integers within inclusive bounds"),
    "kc_8008ea2b8901b39bf313": ("reject", "none", "not_applicable",
        "standard-I/O circular range-update/query API is not the locked single function"),
}


def build(manifest_raw: bytes, shortlist_raw: bytes) -> dict:
    manifest = json.loads(manifest_raw)
    shortlist = json.loads(shortlist_raw)
    families = manifest["families"][:20]
    if len(families) != 20 or {row["family_id"] for row in families} != set(LABELS):
        raise ValueError("frozen first-20 family order no longer matches labels")
    candidates = {row["id"]: row for row in shortlist["rows"]}
    rows = []
    for index, family in enumerate(families):
        decision, variant, universe_mode, reason = LABELS[family["family_id"]]
        representative = candidates[family["representative_id"]]
        if representative["question_sha256"] != family["representative_question_sha256"]:
            raise ValueError("representative hash mismatch")
        rows.append({
            "review_index": index,
            "family_id": family["family_id"],
            "subset": family["subset"],
            "seed_ids": family["seed_ids"],
            "representative_id": family["representative_id"],
            "question_sha256": representative["question_sha256"],
            "contract_decision": decision,
            "variant": variant,
            "universe_mode": universe_mode,
            "reason": reason,
            "oracle_status": "pending",
            "history_status": "pending",
            "near_duplicate_status": "pending",
            "sandbox_status": "pending",
        })
    decisions = {name: sum(row["contract_decision"] == name for row in rows)
                 for name in ("applicable_candidate", "near_negative_candidate", "reject")}
    return {
        "schema_version": "p16_kodcode_family_contract_review_v1",
        "family_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "shortlist_sha256": hashlib.sha256(shortlist_raw).hexdigest(),
        "frozen_batch_rule": "first 20 families in review_order",
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

