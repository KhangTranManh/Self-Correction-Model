"""Select the balanced final holdout deterministically from frozen initial rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/harness_confirmation_v1.yaml"
SOURCE_LOCK = ROOT / "phase6/data/harness_confirmation_source_pool_v1_lock.json"
# `run_harness_vllm.py` names its initial output by checkpoint.  The initial
# rollout is deliberately performed only with original_solver, so this is the
# immutable source for balancing the fresh holdout.
INITIALS = ROOT / "outputs/phase6_harness_confirmation_v1/initial/initial_original_solver.jsonl"
OUTPUT = ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl"
LOCK = ROOT / "phase6/data/harness_confirmation_holdout_v1_lock.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise RuntimeError("Final holdout already exists; refusing to overwrite")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    for relative, expected in source_lock["files"].items():
        if sha256_lf(ROOT / relative) != expected["sha256_lf"]:
            raise RuntimeError(f"Frozen source input changed: {relative}")
    rows = read_jsonl(INITIALS)
    if len(rows) != int(config["scope"]["source_pool_rows"]):
        raise RuntimeError("Initial rollout count does not equal frozen source pool")
    selected: list[dict] = []
    per_state = int(config["scope"]["final_holdout_rows"]) // 2
    for state in (True, False):
        group = [row for row in rows if bool(row["initial_correct"]) is state]
        group.sort(key=lambda row: (row["source_rank_sha256"], row["problem_id"]))
        if len(group) < per_state:
            raise RuntimeError(f"Insufficient initial_correct={state}: {len(group)}")
        selected.extend(group[:per_state])
    selected.sort(key=lambda row: (row["source_rank_sha256"], row["problem_id"]))
    if sum(bool(row["initial_correct"]) for row in selected) != per_state or len(selected) != 2 * per_state:
        raise RuntimeError("Final holdout composition mismatch")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in selected), encoding="utf-8", newline="\n")
    lock = {
        "schema_version": "phase6_harness_confirmation_holdout_lock_v1",
        "status": "frozen_before_confidence_or_recheck",
        "files": {
            CONFIG.relative_to(ROOT).as_posix(): {"sha256_lf": sha256_lf(CONFIG)},
            SOURCE_LOCK.relative_to(ROOT).as_posix(): {"sha256_lf": sha256_lf(SOURCE_LOCK)},
            INITIALS.relative_to(ROOT).as_posix(): {"sha256_lf": sha256_lf(INITIALS)},
            OUTPUT.relative_to(ROOT).as_posix(): {"sha256_lf": sha256_lf(OUTPUT)},
        },
        "rows": len(selected), "initially_correct": per_state, "initially_wrong": per_state,
        "protected_data_used": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
