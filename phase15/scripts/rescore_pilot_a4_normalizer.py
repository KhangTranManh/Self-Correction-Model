"""Evaluate the frozen P15-A4 deterministic suffix normalizer on P15-A3 A0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "phase15" / "kaggle" / "pilot_s_format_continuation_v1"
sys.path.insert(0, str(HERE))
from pilot_score import (  # noqa: E402
    correct_answer,
    numeric,
    parse_lenient,
    parse_rescued,
    same,
    solve_errors,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(output: str, question: str, hit_cap: bool) -> tuple[str, str]:
    """Return output and status without consulting any reference answer."""
    before_errors = solve_errors(output, question, hit_768_cap=hit_cap)
    if "S-format" not in before_errors:
        return output, "not_applicable"
    rescued = parse_rescued(output)
    if not numeric(rescued):
        return output, "rejected_non_numeric"
    candidate = output.rstrip() + f"\nĐáp số: {rescued}\n"
    parsed = parse_lenient(candidate)
    if not numeric(parsed) or not same(parsed, rescued):
        return output, "rejected_parse_mismatch"
    return candidate, "accepted"


def block(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "format_clean": sum(row["format_clean"] for row in rows),
        "joint": sum(row["correct"] and row["format_clean"] for row in rows),
    }


def paired(before: list[dict], after: list[dict]) -> dict:
    return {
        metric: {
            "fixes": sum(not left[metric] and right[metric] for left, right in zip(before, after)),
            "harms": sum(left[metric] and not right[metric] for left, right in zip(before, after)),
        }
        for metric in ("correct", "format_clean", "joint")
    }


def score(audit_path: Path) -> dict:
    manifest_path = HERE / "episodes.json"
    expected_manifest = "1839aba51f447c4f0da8b1b95a8fea9775072a81736f50f1faf96e418ac1c2e9"
    if digest(manifest_path) != expected_manifest:
        raise AssertionError("Manifest hash changed")
    episodes = {
        episode["episode_id"]: episode
        for episode in json.loads(manifest_path.read_text(encoding="utf-8"))["episodes"]
    }
    records = read_jsonl(audit_path)
    metadata = [row for row in records if row["type"] == "metadata"]
    if len(metadata) != 1 or metadata[0]["amendment"] != "P15-A3":
        raise AssertionError("Expected one P15-A3 metadata row")
    completed = {row["episode_id"] for row in records if row["type"] == "episode_complete"}
    if completed != set(episodes):
        raise AssertionError("P15-A3 audit is not complete")
    completions = [row for row in records if row["type"] == "completion" and row["arm"] == "a0_base"]
    if len(completions) != 120:
        raise AssertionError("Expected 120 A0 completions")

    before: list[dict] = []
    after: list[dict] = []
    statuses: dict[str, int] = {}
    changed_clean = 0
    accepted_incorrect = 0
    seen: set[tuple[str, str]] = set()
    for row in completions:
        episode = episodes[row["episode_id"]]
        items = {item["id"]: (item, "probe") for item in episode["probes"]}
        items[episode["control"]["id"]] = (episode["control"], "control")
        if row["item_id"] not in items:
            raise AssertionError("Unknown item")
        key = (row["episode_id"], row["item_id"])
        if key in seen:
            raise AssertionError("Duplicate A0 completion")
        seen.add(key)
        item, role = items[row["item_id"]]
        errors = solve_errors(row["output"], item["question"], hit_768_cap=bool(row["hit_cap"]))
        left = {
            "role": role,
            "correct": correct_answer(row["output"], item["reference_answer"]),
            "format_clean": "S-format" not in errors,
        }
        left["joint"] = left["correct"] and left["format_clean"]
        repaired, status = normalize(row["output"], item["question"], bool(row["hit_cap"]))
        statuses[status] = statuses.get(status, 0) + 1
        new_errors = solve_errors(repaired, item["question"], hit_768_cap=bool(row["hit_cap"]))
        right = {
            "role": role,
            "correct": correct_answer(repaired, item["reference_answer"]),
            "format_clean": "S-format" not in new_errors,
        }
        right["joint"] = right["correct"] and right["format_clean"]
        changed_clean += int(left["format_clean"] and repaired != row["output"])
        accepted_incorrect += int(status == "accepted" and not right["correct"])
        before.append(left)
        after.append(right)

    result = {
        "schema_version": "phase15_a4_normalizer_rescore_v1",
        "source_audit": str(audit_path),
        "source_audit_sha256": digest(audit_path),
        "completed_episodes": len(completed),
        "transformation_used_reference_answers": False,
        "statuses": dict(sorted(statuses.items())),
        "accepted_but_incorrect": accepted_incorrect,
        "changed_already_clean": changed_clean,
        "roles": {},
    }
    for role in ("probe", "control"):
        left = [row for row in before if row["role"] == role]
        right = [row for row in after if row["role"] == role]
        result["roles"][role] = {
            "before": block(left),
            "after": block(right),
            "paired": paired(left, right),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(score(args.audit.resolve()), ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
