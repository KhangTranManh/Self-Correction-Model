"""Independently validate and score the P15-A3 matched-seed Colab audit."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


HERE = Path(__file__).resolve().parents[1] / "kaggle" / "pilot_s_format_continuation_v1"
sys.path.insert(0, str(HERE))
from pilot_score import correct_answer, solve_errors  # noqa: E402

ARMS = ("a0_base", "a1_context", "a2_naive")


def seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def score(path: Path) -> dict:
    manifest_path = HERE / "episodes.json"
    expected_hash = "1839aba51f447c4f0da8b1b95a8fea9775072a81736f50f1faf96e418ac1c2e9"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_hash:
        raise AssertionError("Manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    metadata = [row for row in rows if row["type"] == "metadata"]
    if len(metadata) != 1 or metadata[0]["manifest_sha256"] != expected_hash:
        raise AssertionError("Missing or mismatched metadata")
    if metadata[0]["shared_seed_rule"] != "seed_for(episode_id + '|a0') + item_index":
        raise AssertionError("Seed rule changed")
    completions = [row for row in rows if row["type"] == "completion"]
    updates = [row for row in rows if row["type"] == "update"]
    completes = [row for row in rows if row["type"] == "episode_complete"]
    completed_ids = {row["episode_id"] for row in completes}
    if len(completes) != len(completed_ids) or not completed_ids <= episodes.keys():
        raise AssertionError("Duplicate or unknown completed episode")
    initial_sha = metadata[0]["initial_adapter_sha256"]
    if any(not row["reset_ok"] or row["reset_sha256"] != initial_sha for row in completes):
        raise AssertionError("Adapter reset failure")
    if len(updates) != len({row["episode_id"] for row in updates}) or any(
        row["episode_id"] not in completed_ids or
        row["initial_adapter_sha256"] != initial_sha or
        row["trained_adapter_sha256"] == initial_sha or
        len(row["losses"]) != 3 for row in updates
    ):
        raise AssertionError("Adapter update failure")

    by_key = {}
    disagreement = 0
    for row in completions:
        eid, arm, item_id = row["episode_id"], row["arm"], row["item_id"]
        if eid not in completed_ids or arm not in ARMS or (eid, arm, item_id) in by_key:
            raise AssertionError("Unexpected or duplicate completion")
        items = [*episodes[eid]["probes"], episodes[eid]["control"]]
        matches = [(index, item) for index, item in enumerate(items) if item["id"] == item_id]
        if len(matches) != 1:
            raise AssertionError("Unexpected item ID")
        index, item = matches[0]
        if row["seed"] != seed_for(eid + "|a0") + index:
            raise AssertionError("Seed mismatch")
        errors = solve_errors(row["output"], item["question"], hit_768_cap=bool(row["hit_cap"]))
        correct = correct_answer(row["output"], item["reference_answer"])
        clean = "S-format" not in errors
        local = {"correct": correct, "format_clean": clean, "joint": correct and clean}
        disagreement += int(bool(row["correct"]) != correct or sorted(row["errors"]) != sorted(errors))
        by_key[(eid, arm, item_id)] = {"role": "control" if index == 2 else "probe", **local}
    for eid in completed_ids:
        for arm in ARMS:
            for item in [*episodes[eid]["probes"], episodes[eid]["control"]]:
                if (eid, arm, item["id"]) not in by_key:
                    raise AssertionError("Missing completion in completed episode")
    if len(completions) != 9 * len(completed_ids):
        raise AssertionError("Unexpected completion count")

    results = {"audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "completed_episodes": len(completed_ids),
               "score_disagreements": disagreement, "arms": {}, "paired": {}}
    for role in ("probe", "control"):
        results["arms"][role] = {}
        for arm in ARMS:
            selected = [value for (eid, a, _), value in by_key.items()
                        if eid in completed_ids and a == arm and value["role"] == role]
            results["arms"][role][arm] = {metric: sum(row[metric] for row in selected)
                                            for metric in ("correct", "format_clean", "joint")}
            results["arms"][role][arm]["n"] = len(selected)
    for arm in ARMS[1:]:
        results["paired"][arm] = {}
        for role in ("probe", "control"):
            pairs = [(by_key[(eid, "a0_base", item["id"])], by_key[(eid, arm, item["id"])])
                     for eid in completed_ids
                     for item in ([*episodes[eid]["probes"]] if role == "probe" else [episodes[eid]["control"]])]
            results["paired"][arm][role] = {
                metric: {"fixes": sum(not base[metric] and test[metric] for base, test in pairs),
                         "harms": sum(base[metric] and not test[metric] for base, test in pairs)}
                for metric in ("correct", "format_clean", "joint")}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.audit), indent=2))
