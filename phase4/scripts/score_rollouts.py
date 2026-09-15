"""Validate already verified Phase 4 transitions and recompute their rewards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phase4.lib.transition import parse_review, transition_reward  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "phase4/configs/transition_rl_v1.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rewards = config["reward"]
    rows = read_jsonl(args.rollouts)
    counts: Counter[str] = Counter()
    total_reward = 0.0
    mismatches: list[str] = []
    decisions: Counter[str] = Counter()
    initial_correct_count = 0
    final_correct_count = 0
    for row in rows:
        action = parse_review(str(row["review_output"]))
        decisions[action.decision] += 1
        initial_correct_count += int(bool(row["initial_correct"]))
        final_correct_count += int(bool(row["final_correct"]))
        reward, reason = transition_reward(
            initial_correct=bool(row["initial_correct"]),
            final_correct=bool(row["final_correct"]),
            action=action,
            rewards=rewards,
        )
        counts[reason] += 1
        total_reward += reward
        if "reward" in row and float(row["reward"]) != reward:
            mismatches.append(str(row.get("rollout_id", row.get("problem_id"))))
    result = {
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "mean_reward": total_reward / len(rows) if rows else None,
        "initial_accuracy": initial_correct_count / len(rows) if rows else None,
        "final_accuracy": final_correct_count / len(rows) if rows else None,
        "decision_counts": dict(sorted(decisions.items())),
        "contract_valid_rate": (
            (len(rows) - decisions["INVALID"]) / len(rows) if rows else None
        ),
        "recorded_reward_mismatches": mismatches,
        "all_passed": not mismatches,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
