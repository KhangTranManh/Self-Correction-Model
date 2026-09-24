"""Freeze the eight-source Phase 5 GPU smoke manifest (CPU only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "phase5/data/splits/v1/train.jsonl"
OUTPUT = ROOT / "phase5/data/protocol/smoke_v1.jsonl"


def rank(source_id: str) -> str:
    return hashlib.sha256(f"phase5_smoke_v1|{source_id}".encode()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen smoke manifest: {OUTPUT}")
    with TRAIN.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    correct = sorted((row for row in rows if row["initial_correct"]),
                     key=lambda row: rank(row["problem_id"]))[:4]
    wrong = sorted((row for row in rows if not row["initial_correct"]),
                   key=lambda row: rank(row["problem_id"]))[:4]
    selected = sorted(correct + wrong, key=lambda row: rank(row["problem_id"]))
    if len(selected) != 8 or sum(row["initial_correct"] for row in selected) != 4:
        raise AssertionError("Smoke manifest is not 4 correct / 4 wrong")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected
    ).encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload)
    print(json.dumps({
        "output": str(OUTPUT), "rows": 8, "correct": 4, "wrong": 4,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
