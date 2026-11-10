"""Reproduce oracle-ready A0 residuals in the frozen CPU sandbox."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from colab_a0_discovery import canonical_sha256, sandbox_execute, sandbox_preflight
from track_a0_cpu_admission import build_tracker, sha256_file

import audit_a0_cpu_review_batch1 as b1
import audit_a0_cpu_review_batch2 as b2
import audit_a0_cpu_review_batch3 as b3
import audit_a0_cpu_review_batch4 as b4
import audit_a0_cpu_review_batch5 as b5
import audit_a0_cpu_review_batch6 as b6
import audit_a0_cpu_review_batch7 as b7
import audit_a0_cpu_review_batch8 as b8


SCHEMA_VERSION = "p16_a0_cpu_sandbox_closure_v1"


def _single(cases):
    return [(case_id, [copy.deepcopy(value)]) for case_id, value in cases]


def _args(cases):
    return [(case_id, list(copy.deepcopy(values))) for case_id, values in cases]


SPECS = {
    0: (b1.REMOVE_PARENTHESES_CASES, b1.remove_parentheses_oracle, _single),
    6: (b2.TAB_CASES, b2.transpose_grid_oracle, _args),
    10: (b2.WEAPON_CASES, b2.identify_weapon_oracle, _single),
    12: (b2.PAIR_CASES, b2.closest_pair_oracle, _single),
    21: (b3.CASE_ID_CASES, b3.case_id_oracle, _single),
    22: (b3.DIFFERENCE_CASES, b3.difference_oracle, _single),
    23: (b4.SCRAP_CASES, b4.calculate_scrap_oracle, _args),
    28: (b4.RACE_CASES, b4.race_oracle, _args),
    29: (b4.DAY_PLAN_CASES, b4.day_plan_oracle, _args),
    30: (b5.KA_CASES, b5.ka_cipher_oracle, _single),
    31: (b5.FISHEX_CASES, b5.fishex_oracle, _single),
    39: (b6.DIAMOND_CASES, b6.diamond_oracle, _args),
    40: (b6.GORDON_CASES, b6.gordon_oracle, _args),
    41: (b6.FINDING_K_CASES, b6.finding_k_oracle, _args),
    42: (b6.CHANGE_CASE_CASES, b6.change_case_oracle, _args),
    43: (b7.BASE_FINDER_CASES, b7.base_finder_oracle, _args),
    44: (b7.SEARCH_SUBSTR_CASES, b7.search_substr_oracle, _args),
    47: (b7.ARCHERS_READY_CASES, b7.archers_ready_oracle, _args),
    48: (b7.COMBS_CASES, b7.combs_oracle, _args),
    51: (b7.WHERES_WALLY_CASES, b7.wheres_wally_oracle, _args),
    56: (b8.UNCOLLAPSE_CASES, b8.uncollapse_oracle, _args),
    58: (b8.STRING_MERGE_CASES, b8.string_merge_oracle, _args),
    60: (b8.CHECKCHOOSE_CASES, b8.checkchoose_oracle, _args),
    61: (b8.CUSTOM_FIB_CASES, b8.custom_fib_oracle, _args),
}


def _json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _load_completions(run_dir: Path) -> dict[int, dict]:
    rows = {}
    for shard in range(4):
        path = run_dir / f"a0_shard_{shard:02d}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "completion":
                rows[row["position"]] = row
    if len(rows) != 400:
        raise RuntimeError("expected 400 frozen completions")
    return rows


def build(inventory: dict, reviews: list[dict], completions: dict[int, dict],
          *, execute=sandbox_execute, preflight=sandbox_preflight,
          source_files: dict[str, str] | None = None) -> dict:
    tracker = build_tracker(inventory, reviews)
    positions = tracker["positions"]["oracle_ready"]
    if positions != sorted(SPECS):
        raise RuntimeError("oracle-ready positions differ from frozen closure specs")
    by_inventory = {row["position"]: row for row in inventory["records"]}
    preflight()
    records = []
    for position in positions:
        frozen = by_inventory[position]
        completion = completions[position]
        for key in ("item_id", "family_id", "prompt_sha256",
                    "recovered_output_sha256"):
            if completion.get(key) != frozen.get(key):
                raise RuntimeError(f"completion {key} mismatch at {position}")
        if completion.get("recovery_status") != "accepted":
            raise RuntimeError(f"frozen recovery not accepted at {position}")
        cases, oracle, normalizer = SPECS[position]
        normalized = normalizer(cases)
        test_inputs = [arguments for _, arguments in normalized]
        expected = [_json_value(oracle(*copy.deepcopy(arguments)))
                    for arguments in test_inputs]
        observation = execute(completion["recovered_output"], {
            "function_name": frozen["function_name"],
            "test_inputs": test_inputs,
            "test_outputs": expected,
        })
        reproduced = (observation.get("status") != "completed"
                      or not observation.get("all_correct", False))
        records.append({
            "position": position,
            "item_id": frozen["item_id"],
            "family_id": frozen["family_id"],
            "recovered_output_sha256": frozen["recovered_output_sha256"],
            "frozen_recovery_status": "accepted",
            "independent_case_count": len(test_inputs),
            "independent_cases_sha256": canonical_sha256(normalized),
            "independent_expected_sha256": canonical_sha256(expected),
            "sandbox_observation": observation,
            "residual_reproduced_after_recovery": reproduced,
            "sandbox_gate": "passed_residual_reproduced" if reproduced
                            else "failed_residual_not_reproduced",
            "correction_admitted": False,
        })
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "review_chain_tip_sha256": reviews[-1]["review_sha256"],
            "source_file_sha256": dict(sorted((source_files or {}).items())),
        },
        "counts": {
            "oracle_ready_checked": len(records),
            "residual_reproduced": sum(
                row["residual_reproduced_after_recovery"] for row in records),
            "residual_not_reproduced": sum(
                not row["residual_reproduced_after_recovery"] for row in records),
            "correction_admitted": 0,
        },
        "records": records,
        "claims": {
            "model_generation": False,
            "gpu_used": False,
            "published_test_values_used": False,
            "dataset_reference_solutions_used": False,
            "human_approval_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
        },
    }
    document["closure_sha256"] = canonical_sha256(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    reviews = [json.loads(path.read_text(encoding="utf-8")) for path in args.reviews]
    files = {args.inventory.name: sha256_file(args.inventory)}
    files.update({path.name: sha256_file(path) for path in args.reviews})
    report = build(inventory, reviews, _load_completions(args.run),
                   source_files=files)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
