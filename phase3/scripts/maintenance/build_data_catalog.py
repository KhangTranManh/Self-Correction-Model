"""Build the compact machine-readable catalog for the closed Phase 3 data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
OUTPUT = ROOT / "data" / "catalog.json"

ARTIFACTS = (
    ("source_pool", "data/source/problems.jsonl", "source", "canonical"),
    ("base_attempts", "data/attempts/base/raw_attempts.jsonl", "immutable_attempts", "canonical"),
    ("v1_attempts", "data/attempts/self_correction_v1/raw_attempts.jsonl", "immutable_attempts", "canonical"),
    ("apps_sources", "data/apps_pilot/candidates/code_candidates.jsonl", "source", "canonical"),
    ("apps_base_attempts", "data/apps_pilot/attempts/base/raw_attempts.jsonl", "immutable_attempts", "canonical"),
    ("apps_v1_attempts", "data/apps_pilot/attempts/self_correction_v1/raw_attempts.jsonl", "immutable_attempts", "canonical"),
    ("unified_inventory", "data/behavior/unified_source_inventory.jsonl", "provenance_index", "canonical"),
    ("decision_only_all", "data/decision_only/decision_only_dataset.jsonl", "router_dataset", "canonical"),
    ("decision_only_train", "data/decision_only/decision_only_train.jsonl", "router_train", "canonical"),
    ("decision_only_dev", "data/decision_only/decision_only_dev.jsonl", "router_dev", "canonical"),
    ("decision_only_test", "data/decision_only/decision_only_test.jsonl", "router_test", "canonical"),
    ("revised_router", "data/revised_router/revised_router_dataset.jsonl", "router_dataset", "historical_negative"),
    ("contrastive_pairs", "data/contrastive_pairs/contrastive_pairs.jsonl", "paired_dataset", "historical_negative"),
    ("dpo_pilot", "data/dpo_pilot/dpo_preferences.jsonl", "preference_dataset", "historical_negative"),
    ("semantic_dpo_pairs", "data/semantic_model_dpo/semantic_pairs.jsonl", "paired_dataset", "historical_negative"),
    ("semantic_dpo", "data/semantic_model_dpo/dpo_preferences.jsonl", "preference_dataset", "historical_negative"),
    ("frozen_200", "data/two_stage_selective_repair/frozen_eval.jsonl", "protected_evaluation", "canonical_frozen"),
    ("router_calibration", "data/router_calibration_v1/calibration_all.jsonl", "calibration_dataset", "completed_gate_failed"),
    ("router_recovery_pairs", "data/router_recovery_v2/current/verified_pairs.partial.jsonl", "paired_dataset", "incomplete_closed"),
    ("router_recovery_behavior", "data/router_recovery_v2/current/development_all.partial.jsonl", "router_dataset", "incomplete_closed"),
    ("representation_probe", "runs/representation_probe/probe_dataset.jsonl", "protected_probe", "canonical_frozen"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_profile(path: Path) -> dict[str, Any]:
    rows = 0
    sources: set[str] = set()
    labels: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            source = row.get("source_id", row.get("id"))
            if source is not None:
                sources.add(str(source))
            if row.get("label") is not None:
                labels[str(row["label"])] += 1
            if row.get("dataset") is not None:
                datasets[str(row["dataset"])] += 1
            if row.get("domain") is not None:
                domains[str(row["domain"])] += 1
    return {
        "rows": rows,
        "unique_sources": len(sources),
        "labels": dict(sorted(labels.items())),
        "datasets": dict(sorted(datasets.items())),
        "domains": dict(sorted(domains.items())),
    }


def main() -> None:
    artifacts = []
    for name, relative, role, lifecycle in ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Catalog artifact is missing: {path}")
        artifacts.append({
            "name": name,
            "path": f"phase3/{relative}",
            "role": role,
            "lifecycle": lifecycle,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            **jsonl_profile(path),
        })

    rollups = {}
    for directory in sorted(path for path in (ROOT / "data").iterdir() if path.is_dir()):
        files = [path for path in directory.rglob("*") if path.is_file()]
        rollups[directory.name] = {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }

    payload = {
        "schema_version": "phase3_closed_data_catalog_v1",
        "phase_status": "closed",
        "closed_on": "2026-09-13",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "directory_rollups": rollups,
        "rules": {
            "canonical_files_are_immutable": True,
            "protected_evaluation_is_never_training_data": True,
            "incomplete_closed_data_requires_a_new_phase_to_resume": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
