"""Freeze Phase 8 protocol, generation, probe, and analysis code hashes once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "phase8/data/execution_lock_v2.json"
FILES = (
    "phase1/src/core/prompts.py",
    "phase1/src/data/verifiers/math.py",
    "phase5/configs/review_protocol_v1.yaml",
    "phase5/configs/experiments.yaml",
    "outputs/phase5_gpu_vllm/probe_v1/selection/original_solver_probe.joblib",
    "outputs/phase5_gpu_vllm/probe_v1/selection/original_solver_probe_selection.json",
    "outputs/phase5_gpu_vllm/probe_v1/selection/warmstart_v2_probe.joblib",
    "outputs/phase5_gpu_vllm/probe_v1/selection/warmstart_v2_probe_selection.json",
    "outputs/phase5_gpu_vllm/probe_v1/selection/correction_sft_v3_probe.joblib",
    "outputs/phase5_gpu_vllm/probe_v1/selection/correction_sft_v3_probe_selection.json",
    "phase7/configs/blind_resolve_v1.yaml",
    "phase7/configs/experiments.yaml",
    "phase7/data/candidates_v1/candidate_problems.jsonl",
    "phase7/scripts/collect_initials_4bit.py",
    "phase7/scripts/paired_prompts.py",
    "phase7/scripts/materialize_v2_local.py",
    "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl",
    "phase8/data/model_lineage_v1.json",
    "phase8/docs/PREREGISTRATION.md",
    "phase8/scripts/batched.py",
    "phase8/scripts/fetch_adapters.py",
    "phase8/scripts/regen_phase7_donors.py",
    "phase8/scripts/collect_first_pass_vllm.py",
    "phase8/scripts/freeze_distractors.py",
    "phase8/scripts/collect_three_arms_vllm.py",
    "phase8/scripts/score_probe.py",
    "phase8/scripts/analyze_protected.py",
    "phase8/scripts/verify_adapters.py",
    "phase8/scripts/verify_execution_lock.py",
    "phase8/scripts/ops/run_remote.sh",
)
AMENDMENT = (
    "Amendment v2 (2026-09-30), made before any v2 generation and without any Phase 8 "
    "answer, score, or label inspected: (1) the RTX 3090 host and all Phase 7/8 raw "
    "outputs were lost, so all Phase 8 generation restarts on one Tesla V100-SXM2-32GB; "
    "(2) requests are batched (64 per vLLM call) with unchanged prompts, per-request seeds, "
    "decoding, and 768-token cap; (3) the Phase 7 donor pool is regenerated with the frozen "
    "Phase 7 initial protocol; (4) the exact Phase 5 probes were recovered with matching "
    "SHA-256 and are used instead of a rebuilt probe; (5) the historical Phase 7 replay "
    "(Step 1 exploratory) cannot run without the lost historical text and is dropped."
)


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("Refusing to replace existing execution lock")
    files = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
             for relative in FILES}
    lock = {
        "schema_version": "phase8_execution_lock_v2",
        "status": "frozen_before_any_v2_generation",
        "supersedes": "phase8/data/execution_lock_v1.json",
        "amendment": AMENDMENT,
        "files": files,
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"locked_files": len(files), "lock_path": str(LOCK)}, indent=2))


if __name__ == "__main__":
    main()
