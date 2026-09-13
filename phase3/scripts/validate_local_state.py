"""Validate the closed Phase 3 package, datasets, reports, and adapters.

This audit is read-only and never loads environment files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "phase3" / "configs" / "experiments.yaml"
CATALOG_PATH = PROJECT_ROOT / "phase3" / "data" / "catalog.json"


def jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-adapters", action="store_true", help="Hash each 323 MB adapter and compare it with the registry")
    args = parser.parse_args()
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    phase = registry.get("phase", {})
    phase_closed = phase.get("status") == "closed"
    training_disabled = phase.get("new_training_allowed") is False
    checks.append({
        "artifact": "phase_registry",
        "closed": phase_closed,
        "new_training_allowed": phase.get("new_training_allowed"),
        "final_router": phase.get("final_router"),
    })
    if not phase_closed or not training_disabled:
        failures.append("registry does not enforce the closed Phase 3 state")

    if not CATALOG_PATH.exists():
        failures.append(f"missing data catalog: {CATALOG_PATH}")
    else:
        catalog = load_json(CATALOG_PATH)
        catalog_valid = (
            catalog.get("schema_version") == "phase3_closed_data_catalog_v1"
            and catalog.get("phase_status") == "closed"
            and catalog.get("artifact_count") == len(catalog.get("artifacts", []))
        )
        checks.append({
            "artifact": "data_catalog",
            "present": True,
            "valid": catalog_valid,
            "entries": catalog.get("artifact_count"),
        })
        if not catalog_valid:
            failures.append("closed data catalog invariants failed")

    frozen = PROJECT_ROOT / registry["frozen_evaluation"]["manifest"]
    expected_frozen = int(registry["frozen_evaluation"]["rows"])
    if not frozen.exists():
        failures.append(f"missing frozen manifest: {frozen}")
    else:
        observed = jsonl_rows(frozen)
        checks.append({"artifact": "frozen_manifest", "rows": observed, "expected": expected_frozen})
        if observed != expected_frozen:
            failures.append(f"frozen row count {observed} != {expected_frozen}")

    semantic_summary_path = PROJECT_ROOT / "phase3/data/semantic_model_dpo/semantic_dpo_summary.json"
    if semantic_summary_path.exists():
        semantic = load_json(semantic_summary_path)
        expected = {
            "pairs": 100,
            "preference_rows": 200,
            "keep": 100,
            "revise": 100,
        }
        observed = {
            "pairs": semantic.get("pairs"),
            "preference_rows": semantic.get("preference_rows"),
            "keep": semantic.get("label_distribution", {}).get("KEEP"),
            "revise": semantic.get("label_distribution", {}).get("REVISE"),
        }
        checks.append({"artifact": "semantic_model_dpo", **observed})
        if observed != expected or not semantic.get("validation", {}).get("all_passed"):
            failures.append(f"semantic dataset invariant mismatch: {observed}")
    else:
        failures.append(f"missing semantic summary: {semantic_summary_path}")

    for key, experiment in registry["experiments"].items():
        item = {"artifact": key, "status": experiment["status"]}
        adapter = experiment.get("adapter")
        if adapter:
            adapter_model = PROJECT_ROOT / adapter / "adapter_model.safetensors"
            item["adapter_present"] = adapter_model.exists()
            if not adapter_model.exists():
                failures.append(f"missing adapter: {adapter_model}")
            elif args.hash_adapters:
                observed_hash = sha256(adapter_model)
                item["adapter_sha256"] = observed_hash
                item["hash_matches"] = observed_hash == experiment.get("adapter_sha256")
                if not item["hash_matches"]:
                    failures.append(f"adapter hash mismatch: {key}")
        frozen_run = experiment.get("frozen_run")
        if frozen_run:
            summary = PROJECT_ROOT / frozen_run / "two_stage_summary.json"
            item["frozen_summary_present"] = summary.exists()
            if not summary.exists():
                failures.append(f"missing frozen summary: {summary}")
        checks.append(item)

    required_reports = (
        "phase3/README.md",
        "phase3/docs/FINAL_REPORT.md",
        "phase3/docs/RESULTS.md",
        "phase3/runs/dpo_pilot_v1/dpo_pilot_report.md",
        "phase3/runs/dpo_semantic_v2/semantic_v2_report.md",
        "outputs/phase3_dpo_semantic_v2/representation_probe/probe_report.md",
    )
    for relative in required_reports:
        exists = (PROJECT_ROOT / relative).exists()
        checks.append({"artifact": relative, "present": exists})
        if not exists:
            failures.append(f"missing report: {relative}")

    removed_queue_artifacts = (
        "phase3/data/router_recovery_v2/full_code_scan/apps_generation_manifest.jsonl",
        "phase3/data/router_recovery_v2/full_code_scan/mbpp_generation_manifest.jsonl",
        "phase3/data/router_recovery_v2/full_code_scan/launch_order.json",
    )
    for relative in removed_queue_artifacts:
        absent = not (PROJECT_ROOT / relative).exists()
        checks.append({"artifact": relative, "removed_on_closure": absent})
        if not absent:
            failures.append(f"unexecuted bulk queue remains after closure: {relative}")

    result = {
        "schema_version": "phase3_local_state_audit_v2",
        "registry": str(REGISTRY_PATH),
        "hash_adapters": args.hash_adapters,
        "checks": checks,
        "failures": failures,
        "all_passed": not failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
