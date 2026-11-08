"""Record taxonomy-v1 lane review for frozen KodCode families 60 through 99."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# lane, control subtype, adapter kind, deferred status, variant, universe mode, reason
LABELS = {
    "kc_0470620ac07c970b55a0": ("control_candidate", "unrelated", "stdio_to_function", None, "none", "not_applicable", "single sorting-feasibility task is unrelated to inclusive filtering"),
    "kc_3d2e26cfb7836097b987": ("control_candidate", "unrelated", None, None, "none", "not_applicable", "explicit array-rotation function is unrelated to inclusive filtering"),
    "kc_f3b1481d413ac58e437f": ("hard_reject", None, None, None, "none", "not_applicable", "malformed distance implementation and weak tests leave the route contract unreliable"),
    "kc_5ec6e1cc1c8520b1a385": ("hard_reject", None, None, None, "none", "not_applicable", "nonempty natural-number interval cannot supply the locked empty behavior"),
    "kc_34074136e2431b9c5339": ("control_candidate", "unrelated", None, None, "none", "not_applicable", "explicit FizzBuzz limiter function is unrelated to inclusive filtering"),
    "kc_b2a5b27512a91f3f34a9": ("control_candidate", "unrelated", "stdio_to_function", None, "none", "not_applicable", "single shortest-route task is unrelated to inclusive filtering"),
    "kc_ba86999cff2727647716": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "batched multi-query prime-count program is outside the locked single-operation contract"),
    "kc_cd9732520f6c7dd886e1": ("control_candidate", "unrelated", "stdio_to_function", None, "none", "not_applicable", "single coverage-feasibility task is unrelated to inclusive filtering"),
    "kc_d7dba027e2b9ffce3e58": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable multi-command flag system is outside the locked single-operation contract"),
    "kc_145bc153f7c995ba1596": ("hard_reject", None, None, None, "none", "not_applicable", "select output uses a NO sentinel instead of the locked empty-list behavior"),
    "kc_082c2dbb0c2818531c0c": ("hard_reject", None, None, None, "none", "not_applicable", "random multidimensional generation and aggregate sum do not define one L-to-R filter"),
    "kc_103302259abc5220bc24": ("hard_reject", None, None, None, "none", "not_applicable", "argmax after time filtering is outside the locked count, sum, and select outputs"),
    "kc_3ac29613400ba7df5936": ("hard_reject", None, None, None, "none", "not_applicable", "top-n aggregation after filtering is outside the locked select operation"),
    "kc_243ae5461e6e5fa1295c": ("direct_candidate", None, None, None, "count", "implicit_integer_range", "explicit function counts primes in inclusive integer bounds"),
    "kc_6d7182765073c9fa2600": ("control_candidate", "unrelated", None, None, "none", "not_applicable", "explicit symmetric-tree function is unrelated to inclusive filtering"),
    "kc_ccc289b68ee5f849e321": ("control_candidate", "unrelated", None, None, "none", "not_applicable", "explicit Caesar-cipher function is unrelated to inclusive filtering"),
    "kc_7e0e47af2e3964f8a4d5": ("control_candidate", "unrelated", None, None, "none", "not_applicable", "explicit word-transformation function is unrelated to inclusive filtering"),
    "kc_da784a200b563c1aa987": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "worker and orchestration targets make this a multi-function execution task"),
    "kc_599efae72fa96722dd7b": ("execution_adapter_candidate", None, "single_entrypoint_with_helper", None, "sum", "implicit_integer_range", "prime-sum entry point has one observable scalar output plus a deterministic helper"),
    "kc_d8a15d48f127497282f1": ("execution_adapter_candidate", None, "single_entrypoint_with_helper", None, "select", "implicit_integer_range", "prime-list entry point has one observable list output plus a deterministic helper"),
    "kc_3be46d7d66377cef1738": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable add/count event class is outside the locked single-operation contract"),
    "kc_0067a2dfdae7ab4a32cf": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable set/sum sequence class is outside the locked single-operation contract"),
    "kc_4f6e925db9b35f045e14": ("execution_adapter_candidate", None, "single_entrypoint_with_helper", None, "select", "implicit_integer_range", "prime-list target is explicit and only needs its required primality helper"),
    "kc_634b099d69d7ff54b409": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable tree update/range-query class is outside the locked single-operation contract"),
    "kc_0dd9ba6c8153b7033c8a": ("hard_reject", None, None, None, "none", "not_applicable", "half-open segment task returns a multi-field dictionary outside the locked variants"),
    "kc_adb306911d6ee0e6316d": ("hard_reject", None, None, None, "none", "not_applicable", "the source asks for explanation of completed code rather than an implementation contract"),
    "kc_2a29dfacdadbaba321b4": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "two conversion targets are outside the locked single-operation contract"),
    "kc_f4a3010fc295ac0f70fb": ("hard_reject", None, None, None, "none", "not_applicable", "earliest/latest/count dictionary is a multi-output contract outside the locked variants"),
    "kc_c80311df889bf80d9fff": ("control_candidate", "unrelated", None, None, "none", "not_applicable", "explicit balanced-tree function is unrelated to inclusive filtering"),
    "kc_90163f4547c88b20101b": ("control_candidate", "unrelated", "stdio_to_function", None, "none", "not_applicable", "single route-distance task is unrelated to inclusive filtering"),
    "kc_e75a89f5b8d3a4192ecb": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable insert/delete/range-sum structure is outside the locked single-operation contract"),
    "kc_87e558d9d7a433648a53": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "two required prime-sum entry points include batched ranges"),
    "kc_b79a5420cd0a76f91393": ("control_candidate", "unrelated", "stdio_to_function", None, "none", "not_applicable", "single linked-list reversal task is unrelated to inclusive filtering"),
    "kc_3bb267de7ac705e0b804": ("hard_reject", None, None, None, "none", "not_applicable", "required sorting conflicts with the locked select variant's input-order output"),
    "kc_ca436fe5392a085d0d71": ("control_candidate", "unrelated", "stdio_to_function", None, "none", "not_applicable", "single partition-sum decision task is unrelated to inclusive filtering"),
    "kc_30f4a4c2c83a4efba6e3": ("direct_candidate", None, None, None, "count", "implicit_integer_range", "explicit function counts palindromic integers in inclusive bounds"),
    "kc_b9e5da36b3f2bef29b07": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable update/sumRange class is outside the locked single-operation contract"),
    "kc_a3c4b7f0397b21ec19e8": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "mutable update/range-query operation stream is outside the locked single-operation contract"),
    "kc_6187d0cf7d1fb7163130": ("hard_reject", None, None, "deferred_multiop", "none", "not_applicable", "multiple construction and query targets return a count/sum tuple rather than one locked output"),
    "kc_d430c79c1129fee30820": ("execution_adapter_candidate", None, "canonical_function_wrapper", None, "sum", "implicit_integer_range", "single odd-number sum has explicit inputs and output but omits the callable name"),
}


def build(manifest_raw: bytes, shortlist_raw: bytes) -> dict:
    manifest = json.loads(manifest_raw)
    shortlist = json.loads(shortlist_raw)
    families = manifest["families"][60:100]
    if len(families) != 40 or {row["family_id"] for row in families} != set(LABELS):
        raise ValueError("frozen family slice 60:100 no longer matches labels")
    candidates = {row["id"]: row for row in shortlist["rows"]}
    rows = []
    for index, family in enumerate(families, start=60):
        lane, control_subtype, adapter_kind, deferred_status, variant, universe_mode, reason = LABELS[family["family_id"]]
        representative = candidates[family["representative_id"]]
        if representative["question_sha256"] != family["representative_question_sha256"]:
            raise ValueError("representative hash mismatch")
        rows.append({
            "review_index": index, "family_id": family["family_id"],
            "subset": family["subset"], "seed_ids": family["seed_ids"],
            "representative_id": family["representative_id"],
            "question_sha256": representative["question_sha256"],
            "review_lane": lane, "control_subtype": control_subtype,
            "adapter_kind": adapter_kind, "deferred_status": deferred_status,
            "variant": variant, "universe_mode": universe_mode, "reason": reason,
            "oracle_status": "pending", "history_status": "pending",
            "near_duplicate_status": "pending", "sandbox_status": "pending",
        })
    lane_names = ("direct_candidate", "execution_adapter_candidate", "control_candidate", "hard_reject")
    counts = {name: sum(row["review_lane"] == name for row in rows) for name in lane_names}
    return {
        "schema_version": "p16_kodcode_family_lane_review_v1",
        "taxonomy": "phase16/docs/FAMILY_REVIEW_TAXONOMY_V1.md",
        "family_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "shortlist_sha256": hashlib.sha256(shortlist_raw).hexdigest(),
        "frozen_batch_rule": "families 60:100 in review_order",
        "counts": {"reviewed": len(rows), **counts,
            "applicable_capacity_candidates": counts["direct_candidate"] + counts["execution_adapter_candidate"],
            "near_negative_controls": sum(row["control_subtype"] == "near_negative" for row in rows),
            "unrelated_controls": sum(row["control_subtype"] == "unrelated" for row in rows),
            "deferred_multiop": sum(row["deferred_status"] == "deferred_multiop" for row in rows),
            "fully_certified": 0},
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
