"""Score a recovery-first micro-update curve from verified Boolean labels.

This CPU-only scorer never executes model output or treats an agent as the
oracle. Inputs must already contain independent full-correct verdicts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any


LEVELS = (8, 16, 32, 64)
ROLE_COUNTS = {"positive": 80, "near_negative": 20, "unrelated": 40}
SEED_COUNT = 3
INTEGRITY_GATES = (
    "reset_ok",
    "hashes_ok",
    "audit_complete",
    "leakage_clear",
    "local_rescore_ok",
)


def _sha256(value: Any, key: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a Boolean")
    return value


def _paired(rows: list[dict[str, Any]]) -> dict[str, int]:
    wins = sum(row["update_correct"] and not row["harness_correct"] for row in rows)
    losses = sum(row["harness_correct"] and not row["update_correct"] for row in rows)
    return {"wins": wins, "losses": losses, "net": wins - losses}


def _validate_items(items: Any) -> tuple[list[dict[str, Any]], tuple[tuple[str, str, str], ...]]:
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("every item must be an object")
        item_id = row.get("id")
        family = row.get("family")
        role = row.get("role")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("item ids must be unique nonempty strings")
        if not isinstance(family, str) or not family:
            raise ValueError("item family must be a nonempty string")
        if role not in ROLE_COUNTS:
            raise ValueError(f"unsupported role: {role}")
        seen.add(item_id)
        _bool(row, "harness_correct")
        _bool(row, "update_correct")
        rows.append(row)
    counts = Counter(row["role"] for row in rows)
    if dict(counts) != ROLE_COUNTS:
        raise ValueError(f"role counts must equal {ROLE_COUNTS}")
    signature = tuple(sorted((row["id"], row["family"], row["role"]) for row in rows))
    return rows, signature


def score(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != "p16_recovery_first_curve_v1":
        raise ValueError("unsupported schema_version")
    pool_sha256 = _sha256(document.get("pool_sha256"), "pool_sha256")
    correction_order = document.get("correction_order")
    if (not isinstance(correction_order, list) or len(correction_order) != LEVELS[-1]
            or any(not isinstance(value, str) or not value for value in correction_order)
            or len(set(correction_order)) != LEVELS[-1]):
        raise ValueError(f"correction_order must contain {LEVELS[-1]} unique nonempty ids")
    levels = document.get("levels")
    if not isinstance(levels, list) or [row.get("n") for row in levels if isinstance(row, dict)] != list(LEVELS):
        raise ValueError(f"levels must be exactly {list(LEVELS)}")

    prior_training: list[str] = []
    common_item_signature: tuple[tuple[str, str, str], ...] | None = None
    reports = []
    for level in levels:
        n = level["n"]
        training_ids = level.get("training_ids")
        if (not isinstance(training_ids, list) or len(training_ids) != n
                or any(not isinstance(value, str) or not value for value in training_ids)
                or len(set(training_ids)) != n):
            raise ValueError(f"level {n} needs {n} unique nonempty training_ids")
        if training_ids[:len(prior_training)] != prior_training:
            raise ValueError("training prefixes must be nested")
        if training_ids != correction_order[:n]:
            raise ValueError("training_ids must match the frozen pool correction_order prefix")
        prior_training = training_ids
        training_ids_sha256 = hashlib.sha256(
            json.dumps(training_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        integrity = level.get("integrity")
        if not isinstance(integrity, dict):
            raise ValueError(f"level {n} needs an integrity object")
        integrity_report = {key: _bool(integrity, key) for key in INTEGRITY_GATES}
        integrity_pass = all(integrity_report.values())

        seeds = level.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != SEED_COUNT:
            raise ValueError(f"level {n} needs exactly {SEED_COUNT} update seeds")
        seed_names: set[str] = set()
        harness_by_id: dict[str, bool] | None = None
        seed_reports = []
        for seed in seeds:
            if not isinstance(seed, dict):
                raise ValueError("seed must be an object")
            seed_name = seed.get("seed")
            if not isinstance(seed_name, str) or not seed_name or seed_name in seed_names:
                raise ValueError("seed names must be unique nonempty strings per level")
            seed_names.add(seed_name)
            items, signature = _validate_items(seed.get("items"))
            if common_item_signature is None:
                common_item_signature = signature
            elif signature != common_item_signature:
                raise ValueError("evaluation item ids, families, and roles must match across levels and seeds")
            current_harness = {row["id"]: row["harness_correct"] for row in items}
            if harness_by_id is None:
                harness_by_id = current_harness
            elif current_harness != harness_by_id:
                raise ValueError("harness verdicts must match across update seeds within a level")

            by_role = {role: [row for row in items if row["role"] == role] for role in ROLE_COUNTS}
            probe = _paired(by_role["positive"])
            win_families = len({
                row["family"] for row in by_role["positive"]
                if row["update_correct"] and not row["harness_correct"]
            })
            harms = {
                role: sum(row["harness_correct"] and not row["update_correct"] for row in by_role[role])
                for role in ("near_negative", "unrelated")
            }
            safety = harms["near_negative"] <= 1 and harms["unrelated"] <= 1
            seed_pass = probe["net"] >= 4 and win_families >= 3 and safety
            seed_reports.append({
                "seed": seed_name,
                "positive_update_vs_harness": probe,
                "positive_win_families": win_families,
                "harness_correct_to_update_wrong": harms,
                "safety_pass": safety,
                "seed_pass": seed_pass,
            })

        nets = [row["positive_update_vs_harness"]["net"] for row in seed_reports]
        level_pass = (
            sum(row["seed_pass"] for row in seed_reports) >= 2
            and min(nets) >= 0
            and all(row["safety_pass"] for row in seed_reports)
            and integrity_pass
        )
        reports.append({
            "n": n,
            "training_ids_sha256": training_ids_sha256,
            "integrity": integrity_report,
            "integrity_pass": integrity_pass,
            "median_positive_net": median(nets),
            "minimum_positive_net": min(nets),
            "passing_seeds": sum(row["seed_pass"] for row in seed_reports),
            "level_pass": level_pass,
            "seeds": seed_reports,
        })

    onset = None
    for current, following in zip(reports, reports[1:]):
        if current["level_pass"] and following["level_pass"]:
            onset = current["n"]
            break
    return {
        "schema_version": "p16_recovery_first_curve_report_v1",
        "verified_boolean_input_only": True,
        "pool_sha256": pool_sha256,
        "evaluation_signature_sha256": hashlib.sha256(
            json.dumps(common_item_signature, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "counts": {"levels": len(reports), "update_seeds_per_level": SEED_COUNT, **ROLE_COUNTS},
        "levels": reports,
        "stable_update_onset_n": onset,
        "largest_level_pass_without_confirmation": reports[-1]["level_pass"] and onset is None,
        "interpretation_limit": (
            "Development resource gate only; a stable onset authorizes a separately frozen execution protocol, "
            "not GPU, deployment, significance, or a learning claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.input.read_text(encoding="utf-8"))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
