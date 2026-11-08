"""Record taxonomy-v1 lane review for frozen KodCode families 40 through 59."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# lane, control subtype, adapter kind, deferred status, variant, universe mode, reason
LABELS = {
    "kc_cc4138fbc897ac32b92d": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit tree-reconstruction function is execution-ready and unrelated to inclusive filtering"),
    "kc_8522b42a19a32fd02540": ("direct_candidate", None, None, None, "count", "implicit_integer_range",
        "explicit function counts palindromic integers in stated inclusive bounds"),
    "kc_43c66b79b701cebff7b7": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit bounded word-reversal function is execution-ready but exercises an unrelated transformation"),
    "kc_17d7f3316e6cfe70cf12": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit pair-sum function is execution-ready and unrelated to inclusive filtering"),
    "kc_3d16fc1170823c201e46": ("hard_reject", None, None, None, "none", "not_applicable",
        "completion target is ambiguous because tree balancing is unfinished while range_sum is already implemented"),
    "kc_6d26680fb623dded46a4": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit list-rotation function is execution-ready and unrelated to inclusive filtering"),
    "kc_d0d7607046e2a6a54a08": ("direct_candidate", None, None, None, "count", "explicit_collection",
        "explicit function counts list elements within inclusive value bounds"),
    "kc_05fce5d2f7f2adc8f949": ("hard_reject", None, None, None, "none", "not_applicable",
        "multiple overlapping bracket intervals do not define one locked L-to-R filter operation"),
    "kc_9ca2ae6b536c8ba0b0bd": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable",
        "mutable update/range-query operation stream is outside the locked single-operation contract"),
    "kc_ecd5a65a02298642ebd7": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit vertical-sum function is execution-ready and unrelated to inclusive filtering"),
    "kc_03d2644446c2ef21c148": ("execution_adapter_candidate", None, "stdio_to_function", None, "count", "implicit_integer_range",
        "single palindrome count over inclusive bounds has one deterministic scalar output"),
    "kc_6f6a3d9d98a02ae4461e": ("hard_reject", None, None, None, "none", "not_applicable",
        "the predicate defining a good number is absent, so the semantic contract is incomplete"),
    "kc_0303b72b8e11c9bf0fa4": ("direct_candidate", None, None, None, "select", "implicit_integer_range",
        "explicit function returns all palindromic integers in inclusive bounds"),
    "kc_8d92073da897e24cba0a": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable",
        "mutable segment-tree update/query API is outside the locked single-operation contract"),
    "kc_92762b87508fa0ee727a": ("direct_candidate", None, None, None, "select", "implicit_integer_range",
        "explicit function returns all self-dividing integers in inclusive bounds"),
    "kc_4bc0c1d8396ddcf6a441": ("direct_candidate", None, None, None, "select", "implicit_integer_range",
        "explicit function returns all primes up to an inclusive upper bound"),
    "kc_e3972b76f7d6b57356bb": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit graph-cycle function is execution-ready and unrelated to inclusive filtering"),
    "kc_8835b86b3c7dd9fb74e2": ("hard_reject", None, None, None, "none", "not_applicable",
        "per-city average dictionary falls outside the locked count, sum, and select outputs"),
    "kc_95acb76b9a6876ad5a32": ("control_candidate", "unrelated", None, None, "none", "not_applicable",
        "explicit linked-list intersection function is execution-ready and unrelated to inclusive filtering"),
    "kc_f3665d5ffdadfe7a04dc": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable",
        "mutable update/sumRange class is outside the locked single-operation contract"),
}


def build(manifest_raw: bytes, shortlist_raw: bytes) -> dict:
    manifest = json.loads(manifest_raw)
    shortlist = json.loads(shortlist_raw)
    families = manifest["families"][40:60]
    if len(families) != 20 or {row["family_id"] for row in families} != set(LABELS):
        raise ValueError("frozen family slice 40:60 no longer matches labels")
    candidates = {row["id"]: row for row in shortlist["rows"]}
    rows = []
    for index, family in enumerate(families, start=40):
        lane, control_subtype, adapter_kind, deferred_status, variant, universe_mode, reason = LABELS[family["family_id"]]
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
            "review_lane": lane,
            "control_subtype": control_subtype,
            "adapter_kind": adapter_kind,
            "deferred_status": deferred_status,
            "variant": variant,
            "universe_mode": universe_mode,
            "reason": reason,
            "oracle_status": "pending",
            "history_status": "pending",
            "near_duplicate_status": "pending",
            "sandbox_status": "pending",
        })
    lane_names = ("direct_candidate", "execution_adapter_candidate", "control_candidate", "hard_reject")
    counts = {name: sum(row["review_lane"] == name for row in rows) for name in lane_names}
    return {
        "schema_version": "p16_kodcode_family_lane_review_v1",
        "taxonomy": "phase16/docs/FAMILY_REVIEW_TAXONOMY_V1.md",
        "family_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "shortlist_sha256": hashlib.sha256(shortlist_raw).hexdigest(),
        "frozen_batch_rule": "families 40:60 in review_order",
        "counts": {
            "reviewed": len(rows),
            **counts,
            "applicable_capacity_candidates": counts["direct_candidate"] + counts["execution_adapter_candidate"],
            "near_negative_controls": sum(row["control_subtype"] == "near_negative" for row in rows),
            "unrelated_controls": sum(row["control_subtype"] == "unrelated" for row in rows),
            "deferred_multiop": sum(row["deferred_status"] == "deferred_multiop" for row in rows),
            "fully_certified": 0,
        },
        "rows": rows,
        "limitations": "Lane review only; adapters are not yet written or hashed and oracle/history/near-duplicate/sandbox gates remain pending.",
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
