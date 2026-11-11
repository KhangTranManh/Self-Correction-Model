"""Flag input-shape differences between frozen independent and discovery cases.

This triage does not decide contract validity and does not use discovery output
values. Published inputs supply only broad top-level type/arity signatures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from close_a0_cpu_sandbox_gates import SPECS
from close_a0_cpu_sandbox_v2 import prepare
from prepare_a0_discovery import canonical_sha256, sha256_file
from rescore_a0_discovery import load_lock, verified_completions, verify_manifest


def shape(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def signature(values):
    return tuple(shape(value) for value in values)


def audit(inventory, bank, reviews, completions, previous, *, source_files):
    tracker, later = prepare(inventory, reviews, completions, previous)
    bank_body = dict(bank)
    if bank_body.pop("bank_sha256", None) != canonical_sha256(bank_body):
        raise RuntimeError("bank hash mismatch")
    old_by_position = {row["position"]: row for row in previous["records"]}
    old_cases = {}
    for position, (cases, _, normalize) in SPECS.items():
        normalized = normalize(cases)
        if canonical_sha256(normalized) != old_by_position[position]["independent_cases_sha256"]:
            raise RuntimeError(f"previous independent cases differ at {position}")
        old_cases[position] = [args for _, args in normalized]
    cases_by_position = {frozen["position"]: inputs
                         for frozen, _, inputs, _, _ in later}
    cases_by_position.update(old_cases)
    ready = tracker["positions"]["oracle_ready"]
    if sorted(cases_by_position) != ready:
        raise RuntimeError("independent case coverage mismatch")
    inventory_by_position = {row["position"]: row for row in inventory["records"]}
    tasks_by_id = {task["item_id"]: task for task in bank["tasks"]}
    flags = []
    for position in ready:
        frozen = inventory_by_position[position]
        task = tasks_by_id[frozen["item_id"]]
        if task["prompt_sha256"] != frozen["prompt_sha256"]:
            raise RuntimeError(f"prompt mismatch at {position}")
        observed = {signature(args) for args in task["test_inputs"]}
        independent = {signature(args) for args in cases_by_position[position]}
        if not independent <= observed:
            flags.append({"position": position, "item_id": task["item_id"],
                          "function_name": task["function_name"],
                          "independent_signatures": sorted(independent),
                          "discovery_signatures": sorted(observed),
                          "verdict": "manual_contract_review_required"})
    report = {
        "schema_version": "p16_a0_oracle_input_shape_triage_v1",
        "source": {"inventory_sha256": inventory["inventory_sha256"],
                   "review_chain_tip_sha256": tracker["source"]["review_chain_tip_sha256"],
                   "source_file_sha256": dict(sorted(source_files.items()))},
        "records_checked": len(ready), "flags": flags,
        "limitations": ["discovery input type and arity are screening hints, not a contract or oracle",
                        "equal broad signatures do not establish valid inputs or correct outputs",
                        "variable-arity or mixed-domain cases may legitimately differ"],
        "claims": {"candidate_code_executed": False, "published_expected_values_used": False,
                   "oracle_validity_certified": False, "gpu_training_authorized": False},
    }
    report["audit_sha256"] = canonical_sha256(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    manifest = verify_manifest(args.run, args.manifest)
    lock = load_lock(args.run)
    if manifest.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("manifest and lock differ")
    completions, _ = verified_completions(args.run, lock)
    paths = [args.inventory, args.bank, args.manifest, args.previous_closure, *args.reviews]
    report = audit(read(args.inventory), read(args.bank), [read(p) for p in args.reviews],
                   completions, read(args.previous_closure),
                   source_files={p.name: sha256_file(p) for p in paths})
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"checked {report['records_checked']}; flagged {len(report['flags'])}")
    print(f"audit_sha256: {report['audit_sha256']}")


if __name__ == "__main__":
    main()
