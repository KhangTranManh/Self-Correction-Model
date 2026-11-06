"""Decompose strict parsing, rescued-answer correctness, and format in P15-A3."""

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
    _to_sympy,
    correct_answer,
    numeric,
    parse_rescued,
    same,
    solve_errors,
)


EXPECTED_AUDIT_SHA256 = "ef2a05b5e4c534451028a2b8cf97fa0787bb381f83171ce0e9b2d9542562bf79"
EXPECTED_MANIFEST_SHA256 = "1839aba51f447c4f0da8b1b95a8fea9775072a81736f50f1faf96e418ac1c2e9"
ARMS = ("a0_base", "a1_context", "a2_naive")
PAIRED_METRICS = (
    "strict_correct",
    "rescued_correct",
    "format_clean",
    "strict_joint",
    "rescued_joint",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_row(output: str, question: str, reference: str, hit_cap: bool) -> dict:
    strict_correct = correct_answer(output, reference)
    rescued = parse_rescued(output)
    try:
        gold = _to_sympy(reference)
    except Exception:
        gold = None
    rescued_correct = numeric(rescued) and same(rescued, gold)
    errors = solve_errors(output, question, hit_768_cap=hit_cap)
    format_clean = "S-format" not in errors
    return {
        "strict_correct": strict_correct,
        "rescued_correct": rescued_correct,
        "format_clean": format_clean,
        "strict_joint": strict_correct and format_clean,
        "rescued_joint": rescued_correct and format_clean,
        "correct_but_strict_unparseable": rescued_correct and not strict_correct,
        "wrong_but_clean": format_clean and not rescued_correct,
        "rescued_numeric": numeric(rescued),
        "all_detector_clean": not errors,
    }


def block(rows: list[dict]) -> dict:
    result = {"n": len(rows)}
    for metric in PAIRED_METRICS:
        result[metric] = sum(row[metric] for row in rows)
    for metric in (
        "correct_but_strict_unparseable",
        "wrong_but_clean",
        "rescued_numeric",
    ):
        result[metric] = sum(row[metric] for row in rows)
    return result


def paired(base: dict[tuple[str, str], dict], arm: dict[tuple[str, str], dict]) -> dict:
    if set(base) != set(arm):
        raise AssertionError("Paired item keys differ across arms")
    result = {}
    for metric in PAIRED_METRICS:
        result[metric] = {
            "fixes": sum(not base[key][metric] and arm[key][metric] for key in base),
            "harms": sum(base[key][metric] and not arm[key][metric] for key in base),
        }
    return result


def score(audit_path: Path) -> dict:
    manifest_path = HERE / "episodes.json"
    audit_sha256 = digest(audit_path)
    manifest_sha256 = digest(manifest_path)
    if audit_sha256 != EXPECTED_AUDIT_SHA256:
        raise AssertionError(f"Audit hash changed: {audit_sha256}")
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise AssertionError(f"Manifest hash changed: {manifest_sha256}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    if len(episodes) != 40:
        raise AssertionError("Expected 40 manifest episodes")

    records = read_jsonl(audit_path)
    metadata = [row for row in records if row.get("type") == "metadata"]
    if len(metadata) != 1 or metadata[0].get("amendment") != "P15-A3":
        raise AssertionError("Expected one P15-A3 metadata row")
    if metadata[0].get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("Audit metadata has an unexpected manifest hash")
    if metadata[0].get("evaluation_split_opened") is not False:
        raise AssertionError("Protected evaluation split state is not closed")

    completed = {row["episode_id"] for row in records if row.get("type") == "episode_complete"}
    if completed != set(episodes):
        raise AssertionError("P15-A3 audit is not complete")

    rows_by_role_arm: dict[str, dict[str, dict[tuple[str, str], dict]]] = {
        role: {arm: {} for arm in ARMS} for role in ("probe", "control")
    }
    saved_correct_label_disagreements = 0
    saved_all_error_joint_label_disagreements = 0
    seeds: dict[tuple[str, str], dict[str, int]] = {}
    completions = [row for row in records if row.get("type") == "completion"]
    if len(completions) != 360:
        raise AssertionError(f"Expected 360 completions, found {len(completions)}")

    for row in completions:
        arm = row.get("arm")
        if arm not in ARMS:
            raise AssertionError(f"Unexpected arm: {arm}")
        episode_id = row["episode_id"]
        episode = episodes.get(episode_id)
        if episode is None:
            raise AssertionError(f"Unknown episode: {episode_id}")
        items = {item["id"]: (item, "probe") for item in episode["probes"]}
        items[episode["control"]["id"]] = (episode["control"], "control")
        item_id = row["item_id"]
        if item_id not in items:
            raise AssertionError(f"Unknown item: {episode_id}/{item_id}")
        item, role = items[item_id]
        key = (episode_id, item_id)
        if key in rows_by_role_arm[role][arm]:
            raise AssertionError(f"Duplicate completion: {key}/{arm}")

        scored = score_row(
            row["output"],
            item["question"],
            item["reference_answer"],
            bool(row["hit_cap"]),
        )
        rows_by_role_arm[role][arm][key] = scored
        seeds.setdefault(key, {})[arm] = row["seed"]
        saved_correct_label_disagreements += int(
            bool(row["correct"]) != scored["strict_correct"]
        )
        saved_all_error_joint_label_disagreements += int(
            bool(row["joint_correct"])
            != (scored["strict_correct"] and scored["all_detector_clean"])
        )

    expected_counts = {"probe": 80, "control": 40}
    for role, count in expected_counts.items():
        for arm in ARMS:
            if len(rows_by_role_arm[role][arm]) != count:
                raise AssertionError(f"Expected {count} {role} rows for {arm}")
    if any(set(arm_seeds) != set(ARMS) or len(set(arm_seeds.values())) != 1 for arm_seeds in seeds.values()):
        raise AssertionError("A0/A1/A2 do not share one recorded seed for every item")
    if len(seeds) != 120:
        raise AssertionError("Expected 120 matched-seed items")

    result = {
        "schema_version": "phase15_a5_answer_decomposition_v1",
        "source_audit": str(audit_path),
        "source_audit_sha256": audit_sha256,
        "manifest_sha256": manifest_sha256,
        "completed_episodes": len(completed),
        "matched_seed_items": len(seeds),
        "saved_correct_label_disagreements": saved_correct_label_disagreements,
        "saved_all_error_joint_label_disagreements": saved_all_error_joint_label_disagreements,
        "evaluation_split_opened": False,
        "arms": {},
        "paired_vs_a0": {},
    }
    for role in ("probe", "control"):
        result["arms"][role] = {
            arm: block(list(rows_by_role_arm[role][arm].values())) for arm in ARMS
        }
    for arm in ("a1_context", "a2_naive"):
        result["paired_vs_a0"][arm] = {}
        for role in ("probe", "control"):
            result["paired_vs_a0"][arm][role] = paired(
                rows_by_role_arm[role]["a0_base"],
                rows_by_role_arm[role][arm],
            )

    a0_probe = result["arms"]["probe"]["a0_base"]["rescued_correct"]
    a2_probe = result["arms"]["probe"]["a2_naive"]["rescued_correct"]
    a0_control = result["arms"]["control"]["a0_base"]["rescued_correct"]
    a2_control = result["arms"]["control"]["a2_naive"]["rescued_correct"]
    probe_pair = result["paired_vs_a0"]["a2_naive"]["probe"]["rescued_correct"]
    conditions = {
        "probe_gain_at_least_4_of_80": a2_probe - a0_probe >= 4,
        "probe_fixes_greater_than_harms": probe_pair["fixes"] > probe_pair["harms"],
        "control_not_worse_than_1_of_40": a2_control - a0_control >= -1,
    }
    result["ia3_gpu_definition_gate"] = {
        "a2_minus_a0_probe": a2_probe - a0_probe,
        "a2_minus_a0_control": a2_control - a0_control,
        "probe_fixes": probe_pair["fixes"],
        "probe_harms": probe_pair["harms"],
        "conditions": conditions,
        "passed": all(conditions.values()),
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
