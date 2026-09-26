"""Freeze Phase 8 protocol, generation, probe, and analysis code hashes once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "phase8/data/execution_lock_v1.json"
FILES = (
    "phase1/src/core/prompts.py",
    "phase1/src/data/verifiers/math.py",
    "phase5/configs/review_protocol_v1.yaml",
    "phase5/data/splits/v1/train.jsonl",
    "phase5/data/splits/v1/development.jsonl",
    "phase5/scripts/extract_prehint_activations.py",
    "phase7/scripts/paired_prompts.py",
    "phase7/scripts/materialize_v2_local.py",
    "phase7/data/split_v1/protected_ids.json",
    "outputs/phase7_initials_v1/initial_rollouts.jsonl",
    "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl",
    "phase8/data/model_lineage_v1.json",
    "phase8/docs/PREREGISTRATION.md",
    "phase8/scripts/collect_first_pass_vllm.py",
    "phase8/scripts/reproduce_phase7_initials.py",
    "phase8/scripts/freeze_distractors.py",
    "phase8/scripts/collect_three_arms_vllm.py",
    "phase8/scripts/fit_locked_probe.py",
    "phase8/scripts/score_probe.py",
    "phase8/scripts/analyze_protected.py",
    "phase8/scripts/mark_phase5_lineage.py",
    "phase8/scripts/verify_adapters.py",
    "phase8/scripts/verify_execution_lock.py",
    "phase8/scripts/ops/run_remote.sh",
)


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("Refusing to replace existing execution lock")
    files = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
             for relative in FILES}
    lock = {
        "schema_version": "phase8_execution_lock_v1",
        "status": "frozen_before_three_arm_probe_and_protected_analysis",
        "timing_note": "First-pass collection had started; no first-answer contents, probe scores, or protected labels were inspected before this lock.",
        "files": files,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"locked_files": len(files), "lock_path": str(LOCK)}, indent=2))


if __name__ == "__main__":
    main()
