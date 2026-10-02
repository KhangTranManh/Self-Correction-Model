"""Freeze (--freeze) or verify the Phase 10 execution lock.

Text files are hashed after CRLF -> LF normalization, so the lock verifies
identically on a Windows checkout (core.autocrlf) and on the Linux GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "phase10/data/execution_lock_v1.json"
TEXT = {".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".md"}
FILES = (
    "phase1/src/core/prompts.py",
    "phase1/src/data/verifiers/math.py",
    "phase7/scripts/paired_prompts.py",
    "phase8/scripts/batched.py",
    "phase9/scripts/answers.py",
    "phase9/scripts/analyze.py",
    "phase9/scripts/collect_checkpoint.py",
    "phase9/scripts/collect_first.py",
    "phase10/data/sources_v1/train_pool.jsonl",
    "phase10/data/sources_v1/holdout.jsonl",
    "phase10/data/sources_v1/report.json",
    "phase10/docs/PREREGISTRATION.md",
    "phase10/scripts/generate.py",
    "phase10/scripts/build_data.py",
    "phase10/scripts/train_judge.py",
    "phase10/scripts/analyze.py",
    "phase10/scripts/verify_lock.py",
    "phase10/scripts/ops/run_remote.sh",
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
            raise RuntimeError("Refusing to replace the Phase 10 execution lock")
        lock = {"schema_version": "phase10_execution_lock_v1",
                "status": "frozen_before_any_phase10_generation",
                "hashing": "sha256 of CRLF->LF normalized bytes for text files",
                "files": {rel: digest(ROOT / rel) for rel in FILES}}
        LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for rel, expected in lock["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"Phase 10 execution lock mismatch: {rel}")
    print(json.dumps({"status": "verified", "files": len(lock["files"]),
                      "lock_sha256": digest(LOCK)}))


if __name__ == "__main__":
    main()
