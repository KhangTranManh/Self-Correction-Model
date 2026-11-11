"""Close remaining A0 sandbox gates in individually resumable CPU batches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import audit_a0_cpu_review_batch9 as batch9
import audit_a0_cpu_review_batch10 as batch10
import audit_a0_cpu_review_batch11 as batch11
import audit_a0_cpu_review_batch16 as batch16
import audit_a0_cpu_review_group_b as group_b
import audit_a0_cpu_review_group_c as group_c
import audit_a0_cpu_review_group_d as group_d
import audit_a0_cpu_review_group_e as group_e
import audit_a0_cpu_review_group_f as group_f
import audit_a0_cpu_review_group_g as group_g
from audit_a0_cpu_review_batch6 import _evidence, canonical_sha256
from colab_a0_discovery import sandbox_execute, sandbox_preflight
from rescore_a0_discovery import load_lock, verified_completions, verify_manifest
from track_a0_cpu_admission import build_tracker, sha256_file


def definitions():
    registry = dict(zip((62, 64, 66, 68), batch9.DEFINITIONS))
    registry.update(batch10.DEFINITIONS)
    registry.update(batch11.DEFINITIONS)
    registry.update(batch16.DEFINITIONS)
    registry.update({position: tuple(value[1:])
                     for position, value in group_b.DEFINITIONS.items()
                     if value[0] == "passed"})
    for group in (group_c, group_d, group_e, group_f, group_g):
        overlap = set(registry) & set(group.DEFINITIONS)
        if overlap:
            raise RuntimeError(f"duplicate definitions: {sorted(overlap)}")
        registry.update(group.DEFINITIONS)
    return registry


def prepare(inventory, reviews, completions, previous_closure):
    tracker = build_tracker(inventory, reviews)
    old = dict(previous_closure)
    claimed = old.pop("closure_sha256", None)
    if claimed != canonical_sha256(old):
        raise RuntimeError("previous closure hash mismatch")
    if previous_closure["source"]["inventory_sha256"] != inventory["inventory_sha256"]:
        raise RuntimeError("previous closure inventory mismatch")
    if previous_closure["source"]["review_chain_tip_sha256"] != reviews[7]["review_sha256"]:
        raise RuntimeError("previous closure review tip mismatch")
    previous_positions = [row["position"] for row in previous_closure["records"]]
    ready = tracker["positions"]["oracle_ready"]
    if len(previous_positions) != 24 or previous_positions != ready[:24]:
        raise RuntimeError("previous closure positions mismatch")
    registry = definitions()
    remaining = ready[24:]
    if sorted(registry) != remaining:
        raise RuntimeError("remaining oracle registry mismatch")
    latest = {row["position"]: row for review in reviews for row in review["rows"]}
    by_inventory = {row["position"]: row for row in inventory["records"]}
    by_completion = {row["position"]: row for row in completions}
    prepared = []
    for position in remaining:
        frozen = by_inventory[position]
        completion = by_completion[position]
        for key in ("item_id", "family_id", "prompt_sha256", "recovered_output_sha256"):
            if completion.get(key) != frozen.get(key):
                raise RuntimeError(f"completion {key} mismatch at {position}")
        code = completion.get("recovered_output")
        if not isinstance(code, str) or hashlib.sha256(code.encode()).hexdigest() != frozen["recovered_output_sha256"]:
            raise RuntimeError(f"recovered output hash mismatch at {position}")
        if completion.get("recovery_status") != "accepted":
            raise RuntimeError(f"recovery not accepted at {position}")
        cases, oracle, reference, mutant, _ = registry[position]
        evidence = _evidence(cases, oracle, reference, mutant)
        if evidence != latest[position]["independent_oracle"]:
            raise RuntimeError(f"oracle evidence mismatch at {position}")
        inputs = [list(copy.deepcopy(arguments)) for _, arguments in cases]
        expected = [json.loads(json.dumps(oracle(*copy.deepcopy(args)), ensure_ascii=False))
                    for args in inputs]
        prepared.append((frozen, code, inputs, expected, evidence))
    return tracker, prepared


def build_batch(inventory, reviews, completions, previous_closure, *, start, count,
                execute=sandbox_execute, preflight=sandbox_preflight, source_files=None):
    tracker, prepared = prepare(inventory, reviews, completions, previous_closure)
    if start < 0 or count < 1 or start + count > len(prepared):
        raise ValueError("batch bounds outside remaining queue")
    selected = prepared[start:start + count]
    preflight()
    records = []
    for frozen, code, inputs, expected, evidence in selected:
        observation = execute(code, {"function_name": frozen["function_name"],
                                     "test_inputs": inputs, "test_outputs": expected},
                              exact_expected=True)
        if observation.get("status") != "completed":
            verdict = "execution_or_scaffold_unresolved"
        elif observation.get("test_count") != len(inputs) or not isinstance(observation.get("match_count"), int):
            raise RuntimeError(f"invalid sandbox observation at {frozen['position']}")
        elif observation["match_count"] == len(inputs) and observation.get("all_correct") is True:
            verdict = "residual_not_reproduced"
        elif 0 <= observation["match_count"] < len(inputs) and observation.get("all_correct") is False:
            verdict = "substantive_residual_reproduced"
        else:
            raise RuntimeError(f"inconsistent sandbox observation at {frozen['position']}")
        records.append({"position": frozen["position"], "item_id": frozen["item_id"],
                        "family_id": frozen["family_id"],
                        "recovered_output_sha256": frozen["recovered_output_sha256"],
                        "independent_cases_sha256": evidence["cases_sha256"],
                        "independent_expected_sha256": canonical_sha256(expected),
                        "sandbox_observation": observation, "verdict": verdict,
                        "correction_admitted": False})
    report = {
        "schema_version": "p16_a0_cpu_sandbox_closure_batch_v2",
        "source": {"inventory_sha256": inventory["inventory_sha256"],
                   "review_chain_tip_sha256": tracker["source"]["review_chain_tip_sha256"],
                   "previous_closure_sha256": previous_closure["closure_sha256"],
                   "source_file_sha256": dict(sorted((source_files or {}).items()))},
        "offset": start, "total_remaining": len(prepared), "records": records,
        "counts": {name: sum(row["verdict"] == name for row in records) for name in
                   ("substantive_residual_reproduced", "residual_not_reproduced",
                    "execution_or_scaffold_unresolved")},
        "claims": {"model_generation": False, "gpu_used": False,
                   "correction_pool_certified": False, "gpu_training_authorized": False},
    }
    report["closure_sha256"] = canonical_sha256(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-closure", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    reviews = [json.loads(path.read_text(encoding="utf-8")) for path in args.reviews]
    previous = json.loads(args.previous_closure.read_text(encoding="utf-8"))
    manifest = verify_manifest(args.run, args.manifest)
    lock = load_lock(args.run)
    if manifest.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("manifest and lock differ")
    completions, _ = verified_completions(args.run, lock)
    paths = [args.inventory, args.manifest, args.previous_closure, *args.reviews]
    hashes = {path.name: sha256_file(path) for path in paths}
    report = build_batch(inventory, reviews, completions, previous,
                         start=args.start, count=args.count, source_files=hashes)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
