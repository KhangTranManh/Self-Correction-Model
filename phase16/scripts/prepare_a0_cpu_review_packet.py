"""Print the next frozen A0 CPU review records with contract and saved output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from close_a0_cpu_sandbox_gates import _load_completions
from track_a0_cpu_admission import build_tracker


def build_packet(inventory: dict, bank: dict, reviews: list[dict],
                 completions: dict[int, dict], count: int,
                 offset: int = 0) -> dict:
    if count < 1:
        raise RuntimeError("count must be positive")
    if offset < 0:
        raise RuntimeError("offset must be non-negative")
    tracker = build_tracker(inventory, reviews)
    start = tracker["positions"]["next_frozen_position"]
    reviewed = set(tracker["positions"]["reviewed"])
    pending = [row for row in inventory["records"]
               if row["admission_status"] == "pending_independent_oracle_review"
               and row["position"] not in reviewed]
    selected = pending[offset:offset + count]
    if pending and pending[0]["position"] != start:
        raise RuntimeError("tracker next position differs from pending inventory")
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    records = []
    for frozen in selected:
        position = frozen["position"]
        task = by_id[frozen["item_id"]]
        completion = completions[position]
        for key in ("item_id", "family_id", "prompt_sha256",
                    "recovered_output_sha256"):
            if completion.get(key) != frozen.get(key):
                raise RuntimeError(f"completion {key} mismatch at {position}")
        if task["prompt_sha256"] != frozen["prompt_sha256"]:
            raise RuntimeError(f"bank prompt mismatch at {position}")
        records.append({
            "position": position,
            "item_id": frozen["item_id"],
            "function_name": task["function_name"],
            "prompt": task["prompt"],
            "recovery_status": completion["recovery_status"],
            "recovered_output": completion["recovered_output"],
        })
    return {
        "source_review_sha256": reviews[-1]["review_sha256"],
        "start_position": start,
        "offset": offset,
        "requested_count": count,
        "returned_count": len(records),
        "remaining_after_packet": (tracker["funnel"]["unreviewed_queue"]
                                   - offset - len(records)),
        "records": records,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    packet = build_packet(
        json.loads(args.inventory.read_text(encoding="utf-8")),
        json.loads(args.bank.read_text(encoding="utf-8")),
        [json.loads(path.read_text(encoding="utf-8")) for path in args.reviews],
        _load_completions(args.run),
        args.count,
        args.offset,
    )
    print(json.dumps(packet, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
