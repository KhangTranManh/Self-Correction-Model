"""Freeze an unused source pool for the Phase 6 harness confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/harness_confirmation_v1.yaml"
OUTPUT = ROOT / "phase6/data/harness_confirmation_source_pool_v1.jsonl"
LOCK = ROOT / "phase6/data/harness_confirmation_source_pool_v1_lock.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rank(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|phase6_harness_confirmation_v1|{problem_id}".encode()).hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise RuntimeError("Source pool already exists; refusing to overwrite")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    candidates_path = ROOT / config["scope"]["source_pool"]
    generated_path = ROOT / config["scope"]["exclude_initial_rollouts"]
    candidates = read_jsonl(candidates_path)
    generated = read_jsonl(generated_path)
    excluded = {row["problem_id"] for row in generated}
    split_paths = [ROOT / "phase5/data/splits/v1" / f"{name}.jsonl"
                   for name in config["scope"]["excluded_phase5_splits"]]
    phase6_paths = [ROOT / "phase6/data" / f"{name}.jsonl"
                    for name in config["scope"]["excluded_phase6_manifests"]]
    for path in split_paths + phase6_paths:
        excluded.update(row["problem_id"] for row in read_jsonl(path))
    eligible = [row for row in candidates if row["id"] not in excluded]
    eligible.sort(key=lambda row: (rank(row["id"], int(config["seed"])), row["id"]))
    selected = eligible[:int(config["scope"]["source_pool_rows"])]
    if len(selected) != int(config["scope"]["source_pool_rows"]):
        raise RuntimeError("Insufficient fresh sources")
    output_rows = [{**row, "problem_id": row["id"], "source_rank_sha256": rank(row["id"], int(config["seed"]))}
                   for row in selected]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in output_rows), encoding="utf-8", newline="\n")
    paths = [CONFIG, candidates_path, generated_path, *split_paths, *phase6_paths, OUTPUT]
    lock = {
        "schema_version": "phase6_harness_confirmation_source_pool_lock_v1",
        "status": "frozen_before_initial_generation",
        "files": {path.relative_to(ROOT).as_posix(): {"sha256_lf": sha256_lf(path)} for path in paths},
        "source_pool_rows": len(output_rows),
        "fresh_after_exclusions": len(eligible),
        "excluded_source_count": len(excluded),
        "protected_data_used": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

