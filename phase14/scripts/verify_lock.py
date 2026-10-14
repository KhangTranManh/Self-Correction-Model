"""Freeze (--freeze) or verify a Phase 14 execution lock.

Two locks, both frozen before the generation they govern:
  pilot  preregistration, config, raw manifest, sources, every Phase 14 script,
         and every earlier-phase file whose code or prompt Phase 14 imports
  main   everything in the pilot lock plus data/main_size.json and the pilot
         report that fixed it
Text files are hashed after CRLF -> LF normalization, so the lock verifies
identically on a Windows checkout and on the Linux GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = {".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".md", ".xml"}
SHARED = (  # earlier-phase code and prompts imported unchanged
    "phase1/src/core/prompts.py",
    "phase1/src/core/schema.py",
    "phase1/src/data/verifiers/math.py",
    "phase4/lib/rollout.py",
    "phase5/scripts/prepare_candidates.py",
    "phase7/scripts/paired_prompts.py",
    "phase8/scripts/batched.py",
    "phase9/scripts/answers.py",
    "phase9/scripts/collect_checkpoint.py",
    "phase9/scripts/prepare_sources.py",
    "phase13/scripts/constrained.py",
    "phase13/scripts/prepare_sources.py",
    "phase12/scripts/generate.py",        # SVAMP holdout loader
    "phase12/data/sources_v1/holdout.jsonl",
    "phase12/data/sources_v1/report.json",
    "phase13/scripts/generate.py",        # GSM8K holdout and bridge attempts loader
    "phase13/data/sources_v1/holdout.jsonl",
    "phase13/data/sources_v1/report.json",
)
PHASE14 = (
    "phase14/docs/PREREGISTRATION.md",
    "phase14/data/raw/manifest.json",
    "phase14/data/raw/ASDiv.xml",
    "phase14/data/raw/math_test.jsonl",
    "phase14/data/sources_v1/report.json",
    "phase14/data/sources_v1/asdiv_pilot.jsonl",
    "phase14/data/sources_v1/asdiv_main_pool.jsonl",
    "phase14/data/sources_v1/math13_pilot.jsonl",
    "phase14/data/sources_v1/math13_main_pool.jsonl",
    "phase14/scripts/config.py",
    "phase14/scripts/fetch_raw.py",
    "phase14/scripts/prepare_sources.py",
    "phase14/scripts/answers.py",
    "phase14/scripts/generate.py",
    "phase14/scripts/metrics.py",
    "phase14/scripts/pilot_report.py",
    "phase14/scripts/analyze.py",
    "phase14/scripts/smoke_model.py",
    "phase14/scripts/smoke_hidden.py",
    "phase14/scripts/extract_hidden.py",
    "phase14/scripts/probe_analysis.py",
    "phase14/scripts/report_tables.py",
    "phase14/scripts/regression.py",
    "phase14/scripts/verify_lock.py",
    "phase14/scripts/ops/run_remote.sh",
    "phase14/scripts/ops/run_sequential.sh",
)
MAIN_EXTRA = ("phase14/data/main_size.json", "outputs/phase14_v1/pilot/pilot_report.json")
COMPLETION_EXTRA = ("phase14/scripts/ops/run_complete.sh", "phase14/scripts/ops/fetch_model.py",
                    "phase14/scripts/ops/gen_batch.py", "phase14/scripts/ops/run_tail.sh")
LOCKS = {"completion": ROOT / "phase14/data/execution_lock_completion.json",  # after the opening (amendment)
         "pilot": ROOT / "phase14/data/execution_lock_pilot.json",
         "main": ROOT / "phase14/data/execution_lock_main.json"}


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in TEXT:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=tuple(LOCKS))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    lock_path = LOCKS[args.stage]
    files = (*SHARED, *PHASE14, *(MAIN_EXTRA if args.stage in ("main", "completion") else ()),
             *(COMPLETION_EXTRA if args.stage == "completion" else ()))
    if args.freeze:
        if lock_path.exists():
            raise RuntimeError(f"Refusing to replace {lock_path.name}")
        lock = {"schema_version": f"phase14_execution_lock_{args.stage}_v1",
                "status": f"frozen_before_any_phase14_{args.stage}_generation",
                "hashing": "sha256 of CRLF->LF normalized bytes for text files",
                "files": {rel: digest(ROOT / rel) for rel in files}}
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for rel, expected in lock["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f"Phase 14 {args.stage} lock mismatch: {rel}")
    print(json.dumps({"status": "verified", "stage": args.stage, "files": len(lock["files"]),
                      "lock_sha256": digest(lock_path)}))


if __name__ == "__main__":
    main()
