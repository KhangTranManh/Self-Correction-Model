"""Freeze (--freeze) or verify the Phase 9 execution lock.

Text files are hashed after CRLF -> LF normalization, so the lock verifies
identically on a Windows checkout (core.autocrlf) and on the Linux GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "phase9/data/execution_lock_v1.json"
TEXT = {".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".md"}
FILES = (
    "phase1/src/core/prompts.py",
    "phase1/src/data/verifiers/math.py",
    "phase5/configs/review_protocol_v1.yaml",
    "outputs/phase5_gpu_vllm/probe_v1/selection/original_solver_probe.joblib",
    "outputs/phase5_gpu_vllm/probe_v1/selection/original_solver_probe_selection.json",
    "outputs/phase5_gpu_vllm/probe_v1/selection/warmstart_v2_probe.joblib",
    "outputs/phase5_gpu_vllm/probe_v1/selection/warmstart_v2_probe_selection.json",
    "outputs/phase5_gpu_vllm/probe_v1/selection/correction_sft_v3_probe.joblib",
    "outputs/phase5_gpu_vllm/probe_v1/selection/correction_sft_v3_probe_selection.json",
    "phase7/scripts/paired_prompts.py",
    "phase8/data/model_lineage_v1.json",
    "phase8/scripts/batched.py",
    "phase9/data/source_pool_v1/development.jsonl",
    "phase9/data/source_pool_v1/protected.jsonl",
    "phase9/data/source_pool_v1/report.json",
    "phase9/docs/PREREGISTRATION.md",
    "phase9/scripts/answers.py",
    "phase9/scripts/collect_first.py",
    "phase9/scripts/collect_checkpoint.py",
    "phase9/scripts/score_probe.py",
    "phase9/scripts/analyze.py",
    "phase9/scripts/verify_lock.py",
    "phase9/scripts/ops/run_remote.sh",
)


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in TEXT:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if LOCK.exists():
            raise RuntimeError("Refusing to replace the Phase 9 execution lock")
        lock = {"schema_version": "phase9_execution_lock_v1",
                "status": "frozen_before_any_phase9_generation",
                "hashing": "sha256 of CRLF->LF normalized bytes for text files",
                "files": {rel: digest(ROOT / rel) for rel in FILES}}
        LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for rel, expected in lock["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"Phase 9 execution lock mismatch: {rel}")
    print(json.dumps({"status": "verified", "files": len(lock["files"]),
                      "lock_sha256": digest(LOCK)}))


if __name__ == "__main__":
    main()
