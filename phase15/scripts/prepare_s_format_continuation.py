"""Freeze the unrun P15-A1 S-format episodes for amendment P15-A2."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "outputs/phase15_kaggle_pilot_v1/v2/phase15_pilot_v1/audit.jsonl"
SCREEN = ROOT / "phase15/data/pilot_v1/screen.jsonl"
RESERVE = ROOT / "phase15/data/pilot_v1/reserve.jsonl"
OUT = ROOT / "phase15/kaggle/pilot_s_format_continuation_v1/episodes.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    from phase15.scripts.pilot_episodes import build_episodes

    sources = {row["id"]: row for row in [*read_jsonl(SCREEN), *read_jsonl(RESERVE)]}
    records = read_jsonl(AUDIT)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_type[row["type"]].append(row)

    solve_pool = []
    for row in by_type["screen_solution"]:
        if int(row["sample"]) != 0:
            continue
        source = sources[row["problem_id"]]
        solve_pool.append({
            **source,
            "correct": bool(row["correct"]),
            "errors": list(row["errors"]),
            "output": row["output"],
        })

    episodes = build_episodes(solve_pool, "S-format", limit=40)
    if len(episodes) != 40:
        raise AssertionError(f"expected 40 reconstructed episodes, got {len(episodes)}")
    completed = [row["episode_id"] for row in by_type["episode_complete"]]
    expected_completed = [episode["episode_id"] for episode in episodes[:31]]
    if completed != expected_completed:
        raise AssertionError("completed IDs are not the first 31 reconstructed S-format episodes")
    frozen = episodes
    if [row["episode_id"] for row in frozen] != [f"s-format_{index:02d}" for index in range(40)]:
        raise AssertionError("unexpected frozen episode IDs")

    all_items = [item["id"] for episode in episodes for item in [episode["teaching"], *episode["probes"], episode["control"]]]
    if len(all_items) != len(set(all_items)):
        raise AssertionError("episode reconstruction is not source-disjoint")

    payload = {
        "schema": "phase15_s_format_completion_manifest_v1",
        "amendment": "P15-A2",
        "prior_audit": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "prior_audit_sha256": digest(AUDIT),
        "screen_sha256": digest(SCREEN),
        "reserve_sha256": digest(RESERVE),
        "reconstructed_capacity": len(episodes),
        "completed_episode_ids": completed,
        "continuation_episode_ids": [row["episode_id"] for row in frozen],
        "episodes": frozen,
    }
    if args.stdout:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "out": str(OUT),
        "sha256": digest(OUT),
        "completed": len(completed),
        "frozen": len(frozen),
    }, indent=2))


if __name__ == "__main__":
    main()
