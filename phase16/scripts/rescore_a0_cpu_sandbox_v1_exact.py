"""Rerun the first 24 frozen sandbox records with exact expected equality."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from close_a0_cpu_sandbox_gates import build
from colab_a0_discovery import canonical_sha256, sandbox_execute
from rescore_a0_discovery import load_lock, verified_completions, verify_manifest
from track_a0_cpu_admission import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    reviews = [json.loads(p.read_text(encoding="utf-8")) for p in args.reviews]
    previous = json.loads(args.previous_closure.read_text(encoding="utf-8"))
    body = dict(previous)
    claimed = body.pop("closure_sha256", None)
    if claimed != canonical_sha256(body) or len(reviews) != 8:
        raise RuntimeError("previous closure or review chain invalid")
    if (previous["source"]["inventory_sha256"] != inventory["inventory_sha256"] or
            previous["source"]["review_chain_tip_sha256"] != reviews[-1]["review_sha256"]):
        raise RuntimeError("previous closure source mismatch")
    manifest = verify_manifest(args.run, args.manifest)
    lock = load_lock(args.run)
    if manifest.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("manifest and lock differ")
    completions, _ = verified_completions(args.run, lock)
    by_position = {r["position"]: r for r in completions}
    for row in previous["records"]:
        code = by_position[row["position"]]["recovered_output"]
        if hashlib.sha256(code.encode()).hexdigest() != row["recovered_output_sha256"]:
            raise RuntimeError(f"recovered output hash mismatch at {row['position']}")
    exact = lambda code, task: sandbox_execute(code, task, exact_expected=True)
    raw = build(inventory, reviews, by_position, execute=exact)
    if len(raw["records"]) != len(previous["records"]):
        raise RuntimeError("closure record count mismatch")
    records = []
    for new, old in zip(raw["records"], previous["records"]):
        for key in ("position", "item_id", "family_id", "recovered_output_sha256",
                    "independent_case_count", "independent_cases_sha256", "independent_expected_sha256"):
            if new[key] != old[key]:
                raise RuntimeError(f"previous closure {key} mismatch at {new['position']}")
        observation = new["sandbox_observation"]
        if observation.get("status") != "completed":
            verdict = "execution_or_scaffold_unresolved"
        elif observation.get("all_correct") is True:
            verdict = "residual_not_reproduced"
        else:
            verdict = "substantive_residual_reproduced"
        records.append({"position": new["position"], "item_id": new["item_id"],
                        "family_id": new["family_id"], "recovered_output_sha256": new["recovered_output_sha256"],
                        "independent_cases_sha256": new["independent_cases_sha256"],
                        "independent_expected_sha256": new["independent_expected_sha256"],
                        "sandbox_observation": observation, "verdict": verdict,
                        "correction_admitted": False})
    names = ("substantive_residual_reproduced", "residual_not_reproduced", "execution_or_scaffold_unresolved")
    report = {"schema_version": "p16_a0_cpu_sandbox_v1_exact_rescore_v1",
              "source": {"previous_closure_sha256": previous["closure_sha256"],
                         "manifest_file_sha256": sha256_file(args.manifest)},
              "records": records,
              "counts": {name: sum(row["verdict"] == name for row in records) for name in names},
              "claims": {"gpu_used": False, "correction_pool_certified": False,
                         "gpu_training_authorized": False}}
    report["rescore_sha256"] = canonical_sha256(report)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
