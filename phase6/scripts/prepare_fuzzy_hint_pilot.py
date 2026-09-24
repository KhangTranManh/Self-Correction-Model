"""Freeze a disjoint balanced development pilot for fuzzy correctness hints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/fuzzy_hint_pilot_v1.yaml"
SOURCE = ROOT / "phase5/data/splits/v1/development.jsonl"
EXCLUDE = ROOT / "phase6/data/pilot_v1.jsonl"
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
OUTPUT = ROOT / "phase6/data/fuzzy_hint_pilot_v1.jsonl"
LOCK = ROOT / "phase6/data/fuzzy_hint_pilot_v1_lock.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rank(row: dict, seed: int) -> str:
    value = f"{seed}|phase6_fuzzy_hint_pilot_v1|{row['problem_id']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise RuntimeError("Pilot or lock already exists; refusing to overwrite")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = read_jsonl(SOURCE)
    excluded = {row["problem_id"] for row in read_jsonl(EXCLUDE)}
    eligible = [row for row in rows if row["problem_id"] not in excluded]
    selected: list[dict] = []
    per_state = int(config["scope"]["sources"]) // 2
    for state in (True, False):
        group = [row for row in eligible if bool(row["initial_correct"]) is state]
        group.sort(key=lambda row: (rank(row, int(config["seed"])), row["problem_id"]))
        selected.extend(group[:per_state])
    selected.sort(key=lambda row: (rank(row, int(config["seed"])), row["problem_id"]))
    if len(selected) != 8 or sum(bool(row["initial_correct"]) for row in selected) != 4:
        raise RuntimeError("Pilot composition is not 4 correct / 4 wrong")
    if {row["problem_id"] for row in selected} & excluded:
        raise RuntimeError("Pilot overlaps the prior Phase 6 pilot")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in selected), encoding="utf-8", newline="\n")
    files = {}
    for path in (CONFIG, SOURCE, EXCLUDE, REGISTRY, OUTPUT):
        files[path.relative_to(ROOT).as_posix()] = {"sha256_lf": sha256_lf(path)}
    lock = {
        "schema_version": "phase6_fuzzy_hint_pilot_lock_v1",
        "status": "frozen_before_generation",
        "files": files,
        "sources": len(selected),
        "initially_correct": 4,
        "initially_wrong": 4,
        "excluded_prior_pilot_sources": len(excluded),
        "prior_pilot_overlap": 0,
        "probability_wrong_percent": config["intervention"]["probability_wrong_percent"],
        "generated_rows_per_checkpoint": len(selected) * len(
            config["intervention"]["probability_wrong_percent"]),
        "protected_data_used": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

