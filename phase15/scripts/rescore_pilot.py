"""Rescore a downloaded Phase 15 pilot audit on CPU.

The Kaggle kernel saves raw completions and enough identifiers to reconstruct
the deterministic episodes.  This command treats those completions as the
source of truth, reapplies the local Phase 14 detector wrappers, and writes a
compact JSON record with integrity checks and arm-level probe/control scores.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))

from phase14.scripts.answers import parse  # noqa: E402
from phase15.scripts.detectors import judge_error, solve_errors  # noqa: E402
from phase15.scripts.pilot_episodes import build_episodes, keep_types  # noqa: E402
from phase15.scripts.pilot_score import (  # noqa: E402
    asks_for_count,
    correct_answer,
    judge_prompt,
    pick,
)


ARMS = ("a0_base", "a1_context", "a2_naive", "a3_self_edit")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_block(rows: list[dict]) -> dict:
    count = len(rows)
    error_free = sum(row["error_free"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    joint = sum(row["error_free"] and row["correct"] for row in rows)
    return {
        "n": count,
        "error_free": error_free,
        "error_free_rate": ratio(error_free, count),
        "correct": correct,
        "correct_rate": ratio(correct, count),
        "joint": joint,
        "joint_rate": ratio(joint, count),
    }


def difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--screen", type=Path, default=ROOT / "phase15/data/pilot_v1/screen.jsonl")
    parser.add_argument("--reserve", type=Path, default=ROOT / "phase15/data/pilot_v1/reserve.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    audit_path = args.audit.resolve()
    summary_path = (args.summary or audit_path.with_name("summary.json")).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = read_jsonl(audit_path)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_type[row["type"]].append(row)

    source_rows = read_jsonl(args.screen) + read_jsonl(args.reserve)
    sources = {row["id"]: row for row in source_rows}
    screen_ids = {row["id"] for row in read_jsonl(args.screen)}

    solution_records: dict[tuple[str, int], dict] = {}
    local_solution: dict[tuple[str, int], dict] = {}
    for row in by_type["screen_solution"]:
        key = (row["problem_id"], int(row["sample"]))
        if key in solution_records:
            raise AssertionError(f"duplicate solution record: {key}")
        source = sources[row["problem_id"]]
        solution_records[key] = row
        local_solution[key] = {
            "errors": solve_errors(row["output"], source["question"], hit_768_cap=bool(row["hit_cap"])),
            "correct": correct_answer(row["output"], source["reference_answer"]),
        }

    judgment_records = {row["problem_id"]: row for row in by_type["screen_judgment"]}
    local_judgment: dict[str, dict] = {}
    for problem_id, row in judgment_records.items():
        shown_a = parse(solution_records[(problem_id, 0)]["output"], "lenient")
        shown_b = parse(solution_records[(problem_id, 1)]["output"], "lenient")
        error = judge_error(row["output"], shown_a, shown_b, hit_token_cap=bool(row["hit_cap"]))
        chosen = pick(row["output"], shown_a, shown_b)
        local_judgment[problem_id] = {"error": error, "correct": chosen == row["correct_side"], "picked": chosen}

    screen_first = [row for row in by_type["screen_solution"] if row["sample"] == 0 and row["problem_id"] in screen_ids]
    local_rates = {
        "S-format": {"count": sum("S-format" in local_solution[(row["problem_id"], 0)]["errors"] for row in screen_first), "eligible": len(screen_first)},
        "S-loop": {"count": sum("S-loop" in local_solution[(row["problem_id"], 0)]["errors"] for row in screen_first), "eligible": len(screen_first)},
        "S-integer": {
            "count": sum("S-integer" in local_solution[(row["problem_id"], 0)]["errors"] for row in screen_first),
            "eligible": sum(asks_for_count(sources[row["problem_id"]]["question"]) for row in screen_first),
        },
        "J-format": {"count": sum(row["error"] == "J-format" for row in local_judgment.values()), "eligible": len(local_judgment)},
        "J-loop": {"count": sum(row["error"] == "J-loop" for row in local_judgment.values()), "eligible": len(local_judgment)},
        "J-resolve": {"count": sum(row["error"] == "J-resolve" for row in local_judgment.values()), "eligible": len(local_judgment)},
        "J-third": {"count": sum(row["error"] == "J-third" for row in local_judgment.values()), "eligible": len(local_judgment)},
    }
    metadata = by_type["metadata"][0]
    spec = metadata["spec"]
    local_kept = keep_types(
        local_rates,
        minimum_rate=float(spec["keep_minimum_rate"]),
        minimum_eligible=int(spec["keep_minimum_eligible"]),
    )

    # Rebuild the exact episode assignment from the classifications saved by
    # the kernel. Local classifications are used only for scoring completions.
    solve_pool = []
    for row in by_type["screen_solution"]:
        if row["sample"] != 0:
            continue
        source = sources[row["problem_id"]]
        solve_pool.append({**source, "correct": row["correct"], "errors": row["errors"], "output": row["output"]})

    judge_pool = []
    for row in by_type["screen_judgment"]:
        problem_id = row["problem_id"]
        source = sources[problem_id]
        first = solution_records[(problem_id, 0)]["output"]
        second = solution_records[(problem_id, 1)]["output"]
        source_text = judge_prompt(source["question"], first, second)
        judge_pool.append({
            "id": f"{problem_id}_judge",
            "question": source["question"],
            "reference_answer": source["reference_answer"],
            "source_text": source_text,
            "errors": [row["error"]] if row["error"] else [],
            "correct": row["correct"],
            "correction": (
                "Choose the shown solution whose final answer is verified. "
                f"Do not invent a third answer.\nVerdict: Solution {row['correct_side']}\n"
                f"ÄĂ¡p sá»‘: {source['reference_answer']}"
            ),
        })

    started = by_type["episode_start"]
    started_types = {row["error_type"] for row in started}
    episode_map: dict[str, dict] = {}
    for error_type in started_types:
        pool = judge_pool if error_type.startswith("J-") else solve_pool
        for episode in build_episodes(pool, error_type, limit=int(spec["max_episodes_per_type"])):
            episode_map[episode["episode_id"]] = episode
    episode_capacity = {
        error_type: len(build_episodes(
            judge_pool if error_type.startswith("J-") else solve_pool,
            error_type,
            limit=int(spec["max_episodes_per_type"]),
        ))
        for error_type in by_type["base_rates"][0]["kept"]
    }

    completion_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    scored = []
    for row in by_type["completion"]:
        episode = episode_map[row["episode_id"]]
        item_roles = {item["id"]: "probe" for item in episode["probes"]}
        item_roles[episode["control"]["id"]] = "control"
        if row["item_id"] not in item_roles:
            raise AssertionError(f"unexpected item {row['item_id']} in {row['episode_id']}")
        completion_ids[(row["episode_id"], row["arm"])].add(row["item_id"])
        item = next(item for item in [*episode["probes"], episode["control"]] if item["id"] == row["item_id"])
        error_type = episode["error_type"]
        if error_type.startswith("S-"):
            errors = solve_errors(row["output"], item["question"], hit_768_cap=bool(row["hit_cap"]))
            correct = correct_answer(row["output"], item["reference_answer"])
        else:
            problem_id = item["id"].removesuffix("_judge")
            shown_a = parse(solution_records[(problem_id, 0)]["output"], "lenient")
            shown_b = parse(solution_records[(problem_id, 1)]["output"], "lenient")
            errors = [kind] if (kind := judge_error(row["output"], shown_a, shown_b, hit_token_cap=bool(row["hit_cap"]))) else []
            correct_side = judgment_records[problem_id]["correct_side"]
            correct = pick(row["output"], shown_a, shown_b) == correct_side
        scored.append({
            "episode_id": row["episode_id"],
            "error_type": error_type,
            "arm": row["arm"],
            "item_id": row["item_id"],
            "role": item_roles[row["item_id"]],
            "errors": errors,
            "error_free": error_type not in errors,
            "correct": correct,
        })

    arm_scores = {}
    for arm in ARMS:
        arm_rows = [row for row in scored if row["arm"] == arm]
        arm_scores[arm] = {
            "episodes": len({row["episode_id"] for row in arm_rows}),
            "probe": score_block([row for row in arm_rows if row["role"] == "probe"]),
            "control": score_block([row for row in arm_rows if row["role"] == "control"]),
        }
    deltas = {}
    for arm in ARMS[1:]:
        arm_rows = [row for row in scored if row["arm"] == arm]
        matched_episodes = {row["episode_id"] for row in arm_rows}
        matched_a0 = {
            role: score_block([
                row for row in scored
                if row["arm"] == "a0_base" and row["episode_id"] in matched_episodes and row["role"] == role
            ])
            for role in ("probe", "control")
        }
        deltas[f"{arm}_minus_a0"] = {
            "matched_episodes": len(matched_episodes),
            "matched_a0": matched_a0,
            "difference": {
                role: {
                    metric: difference(
                        arm_scores[arm][role][f"{metric}_rate"],
                        matched_a0[role][f"{metric}_rate"],
                    )
                    for metric in ("error_free", "correct", "joint")
                }
                for role in ("probe", "control")
            },
        }

    kernel_rates = by_type["base_rates"][0]["rates"]
    classification_differences = {
        "solutions": sum(
            sorted(row["errors"]) != sorted(local_solution[(row["problem_id"], int(row["sample"]))]["errors"])
            for row in by_type["screen_solution"]
        ),
        "solution_correct": sum(
            bool(row["correct"]) != bool(local_solution[(row["problem_id"], int(row["sample"]))]["correct"])
            for row in by_type["screen_solution"]
        ),
        "judgments": sum(row["error"] != local_judgment[row["problem_id"]]["error"] for row in by_type["screen_judgment"]),
        "judgment_correct": sum(bool(row["correct"]) != bool(local_judgment[row["problem_id"]]["correct"]) for row in by_type["screen_judgment"]),
    }

    completed_ids = {row["episode_id"] for row in by_type["episode_complete"]}
    start_ids = {row["episode_id"] for row in started}
    reset_rows = by_type["episode_complete"]
    updates = Counter(row["arm"] for row in by_type["update"])
    self_edits = Counter(row["status"] for row in by_type["self_edit"])
    result = {
        "schema_version": "phase15_pilot_rescore_v1",
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": digest(audit_path),
        "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": digest(summary_path),
        "integrity": {
            "audit_hash_matches_summary": digest(audit_path) == summary["audit_sha256"],
            "record_counts": dict(sorted(Counter(row["type"] for row in records).items())),
            "episodes_started": len(start_ids),
            "episodes_completed": len(completed_ids),
            "started_equal_completed": start_ids == completed_ids,
            "reset_ok": sum(bool(row["reset_ok"]) for row in reset_rows),
            "updates": dict(sorted(updates.items())),
            "self_edits": dict(sorted(self_edits.items())),
            "a3_skipped": len(by_type["a3_skipped"]),
        },
        "kernel_summary": summary,
        "base_rates": {
            "kernel": kernel_rates,
            "local": local_rates,
            "kernel_kept": by_type["base_rates"][0]["kept"],
            "local_kept": local_kept,
            "classification_differences": classification_differences,
        },
        "episode_reconstruction": {
            "started_types": sorted(started_types),
            "started_ids_found": len(start_ids & episode_map.keys()),
            "all_started_ids_found": start_ids <= episode_map.keys(),
            "required_per_type": int(spec["max_episodes_per_type"]),
            "capacity": episode_capacity,
            "capacity_sufficient": {
                name: count >= int(spec["max_episodes_per_type"])
                for name, count in episode_capacity.items()
            },
        },
        "partial": not bool(summary["go_nogo_decided"]),
        "arms": arm_scores,
        "deltas": deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
