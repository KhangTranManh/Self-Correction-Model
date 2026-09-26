"""Hash-lock exact Phase 7 paired generation inputs after balanced selection."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase7.scripts.collect_initials_4bit import sha256, write_atomic

LOCK = ROOT / "phase7/data/protocol/paired_generation_v1_lock.json"


def main() -> None:
    files = [
        "phase7/configs/blind_resolve_v1.yaml",
        "phase7/configs/experiments.yaml",
        "phase7/scripts/collect_paired_vllm.py",
        "phase7/scripts/paired_prompts.py",
        "phase7/scripts/analyze_paired.py",
        "phase7/scripts/collect_initials_4bit.py",
        "phase7/data/candidates_v1/candidate_problems.jsonl",
        "phase7/data/split_v1/split_report.json",
        "phase7/data/split_v1/development_ids.json",
        "phase7/data/split_v1/protected_ids.json",
        "phase1/src/core/prompts.py",
        "outputs/phase7_initials_v1/summary.json",
        "outputs/phase7_initials_v1/initial_rollouts.jsonl",
        "models/phase7_v2_merged_fp16/phase7_lineage.json",
        "outputs/phase4_exploration_warmstart_v2/final_adapter/adapter_model.safetensors",
        "outputs/phase4_correction_sft_v3/final_adapter/adapter_model.safetensors",
    ]
    merged = ROOT / "models/phase7_v2_merged_fp16"
    shards = sorted(merged.glob("*.safetensors"))
    if not shards:
        raise ValueError("V2 merged model shards are missing")
    files.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in shards)
    if not all((ROOT / item).exists() for item in files):
        raise FileNotFoundError("Required paired-generation input is missing")
    split = json.loads((ROOT / "phase7/data/split_v1/split_report.json").read_text())
    if split["development_count"] != 40 or split["protected_count"] != 160:
        raise ValueError("Balanced split size mismatch")
    lock = {
        "schema_version": "phase7_paired_generation_lock_v1",
        "purpose": "same frozen inputs for both arms and all three checkpoints",
        "files": {item: sha256(ROOT / item) for item in files},
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        old = json.loads(LOCK.read_text(encoding="utf-8"))
        if old != lock:
            raise ValueError("Existing paired protocol lock differs")
    else:
        write_atomic(LOCK, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "locked", "files": len(files),
                      "sha256": sha256(LOCK)}, indent=2))


if __name__ == "__main__":
    main()
