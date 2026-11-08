"""Relabel the first 40 frozen KodCode family reviews under taxonomy v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# Overrides are intentionally keyed by immutable family ID. Rows not listed
# retain the mechanical mapping from their original contract decision.
OVERRIDES = {
    # Batch 1: clean unrelated controls.
    "kc_045d53717791b2d79a3a": ("control_candidate", "unrelated", None,
        "explicit largest-values function is execution-ready but exercises an unrelated tree traversal skill"),
    "kc_794cb9c01bb5d0496ecd": ("control_candidate", "unrelated", None,
        "explicit array-shift function is execution-ready and unrelated to inclusive filtering"),
    "kc_3fb5ff0a1a9ce8002b13": ("control_candidate", "unrelated", "stdio_to_function",
        "single deterministic k-th-largest task can be mechanically wrapped and exercises an unrelated BST skill"),
    # Batch 1: positive contracts needing only a mechanical interface adapter.
    "kc_2770df63debd7f6420dc": ("execution_adapter_candidate", None, "scaffold_syntax_only",
        "count contract and signature are explicit; the scaffold has one trailing punctuation blemish"),
    "kc_15ace8ed10efc8a450b9": ("execution_adapter_candidate", None, "canonical_function_wrapper",
        "single range-sum BST operation has explicit inputs and output but omits the callable name"),
    "kc_7ade91a5f19fb0f41769": ("execution_adapter_candidate", None, "stdio_to_function",
        "single uniform-digit count over inclusive bounds has one deterministic scalar output"),
    # Batch 1: excluded multi-operation/multi-query queue.
    "kc_d6ef57eff1aa834cacf8": ("hard_reject", None, "deferred_multiop",
        "multi-query standard-I/O range-sum program is outside the locked single-operation contract"),
    "kc_1b962a30388143c357fb": ("hard_reject", None, "deferred_multiop",
        "mutable range-update/query system is outside the locked single-operation contract"),
    "kc_84632a3c91d9e2d89678": ("hard_reject", None, "deferred_multiop",
        "multi-operation range-toggle program is outside the locked single-operation contract"),
    "kc_8008ea2b8901b39bf313": ("hard_reject", None, "deferred_multiop",
        "circular range-update/query system is outside the locked single-operation contract"),
    # Batch 2: clean unrelated controls.
    "kc_9f0e1beea36207a478f0": ("control_candidate", "unrelated", None,
        "explicit largest-BST-subtree function is execution-ready but unrelated to C-upper-inclusive"),
    "kc_cb3035fb2a42a9617e1f": ("control_candidate", "unrelated", None,
        "explicit color-transition function is execution-ready but unrelated to inclusive filtering"),
    # Batch 2: one positive task with a helper plus one observable entry point.
    "kc_927665c3e7c9bf59a74f": ("execution_adapter_candidate", None, "single_entrypoint_with_helper",
        "Armstrong listing has one observable list entry point plus a deterministic helper"),
    # Batch 2: excluded multi-operation queue.
    "kc_c0371fdb56f10d96eb0c": ("hard_reject", None, "deferred_multiop",
        "multi-method employee database is outside the locked single-operation contract"),
    "kc_ceda7867e16cb3046bae": ("hard_reject", None, "deferred_multiop",
        "range-update/query program is outside the locked single-operation contract"),
    "kc_f33e3c0b7091d041af56": ("hard_reject", None, "deferred_multiop",
        "two required target functions are outside the locked single-operation contract"),
    "kc_27bd0e71847897f96d2a": ("hard_reject", None, "deferred_multiop",
        "mutable multi-method laboratory API is outside the locked single-operation contract"),
    "kc_ebf36eb08b77759f85dc": ("hard_reject", None, "deferred_multiop",
        "range-update/prefix-query system is outside the locked single-operation contract"),
}


def base_lane(decision: str) -> tuple[str, str | None, str | None]:
    if decision == "applicable_candidate":
        return "direct_candidate", None, None
    if decision == "near_negative_candidate":
        return "control_candidate", "near_negative", None
    if decision == "reject":
        return "hard_reject", None, None
    raise ValueError(f"unknown original contract decision: {decision}")


def build(manifest_raw: bytes, batch_raws: list[bytes]) -> dict:
    manifest = json.loads(manifest_raw)
    batches = [json.loads(raw) for raw in batch_raws]
    source_rows = [row for batch in batches for row in batch["rows"]]
    if len(source_rows) != 40:
        raise ValueError("expected exactly 40 source review rows")
    by_index = {row["review_index"]: row for row in source_rows}
    if set(by_index) != set(range(40)):
        raise ValueError("source reviews do not cover frozen indices 0:40")

    rows = []
    for index, family in enumerate(manifest["families"][:40]):
        source = by_index[index]
        if source["family_id"] != family["family_id"]:
            raise ValueError(f"family mismatch at frozen index {index}")
        lane, control_subtype, adapter_kind = base_lane(source["contract_decision"])
        reason = source["reason"]
        deferred_status = None
        if family["family_id"] in OVERRIDES:
            lane, control_subtype, special, reason = OVERRIDES[family["family_id"]]
            if special == "deferred_multiop":
                deferred_status = special
                adapter_kind = None
            else:
                adapter_kind = special
        rows.append({
            "review_index": index,
            "family_id": family["family_id"],
            "subset": family["subset"],
            "seed_ids": family["seed_ids"],
            "representative_id": family["representative_id"],
            "question_sha256": source["question_sha256"],
            "original_contract_decision": source["contract_decision"],
            "review_lane": lane,
            "control_subtype": control_subtype,
            "adapter_kind": adapter_kind,
            "deferred_status": deferred_status,
            "reason": reason,
            "oracle_status": source["oracle_status"],
            "history_status": source["history_status"],
            "near_duplicate_status": source["near_duplicate_status"],
            "sandbox_status": source["sandbox_status"],
        })

    lane_names = ("direct_candidate", "execution_adapter_candidate",
                  "control_candidate", "hard_reject")
    lane_counts = {name: sum(row["review_lane"] == name for row in rows)
                   for name in lane_names}
    adapter_kinds = sorted({row["adapter_kind"] for row in rows if row["adapter_kind"]})
    return {
        "schema_version": "p16_kodcode_family_lane_review_v1",
        "taxonomy": "phase16/docs/FAMILY_REVIEW_TAXONOMY_V1.md",
        "family_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_review_sha256": [hashlib.sha256(raw).hexdigest() for raw in batch_raws],
        "frozen_batch_rule": "families 0:40 in review_order",
        "counts": {
            "reviewed": len(rows),
            **lane_counts,
            "applicable_capacity_candidates": lane_counts["direct_candidate"] + lane_counts["execution_adapter_candidate"],
            "near_negative_controls": sum(row["control_subtype"] == "near_negative" for row in rows),
            "unrelated_controls": sum(row["control_subtype"] == "unrelated" for row in rows),
            "deferred_multiop": sum(row["deferred_status"] == "deferred_multiop" for row in rows),
            "fully_certified": 0,
        },
        "adapter_kinds_observed": adapter_kinds,
        "rows": rows,
        "limitations": "Lane review only; adapters are not yet written or hashed and oracle/history/near-duplicate/sandbox gates remain pending.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("batch1", type=Path)
    parser.add_argument("batch2", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = build(args.manifest.read_bytes(), [args.batch1.read_bytes(), args.batch2.read_bytes()])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
