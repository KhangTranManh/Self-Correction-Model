"""CPU-only, Boolean-verdict triage for P16-S0 synthetic or audited records.

This scorer never executes candidate code or infers correctness from text.
The input must already contain independently verified full-correct labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


S2_ARMS = ("a0", "a1", "a1_sham", "a2", "a2_sham")


def _require_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a Boolean")
    return value


def _require_id(row: dict[str, Any], seen: set[str]) -> str:
    value = row.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every source task needs a nonempty id")
    if value in seen:
        raise ValueError(f"duplicate source task id: {value}")
    seen.add(value)
    return value


def _paired(rows: list[dict[str, Any]], treatment: str, baseline: str) -> dict[str, int]:
    wins = sum(row[treatment] and not row[baseline] for row in rows)
    losses = sum(row[baseline] and not row[treatment] for row in rows)
    return {"wins": wins, "losses": losses, "net": wins - losses}


def score(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != "p16_signal_only_cpu_v1":
        raise ValueError("unsupported schema_version")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 20:
        raise ValueError("exactly 20 episodes required")

    seen_tasks: set[str] = set()
    seen_episodes: set[str] = set()
    wrong: list[dict[str, Any]] = []
    correct: list[dict[str, Any]] = []
    roles: dict[str, list[dict[str, Any]]] = {
        "positive": [], "near_negative": [], "unrelated": []
    }
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("episode must be an object")
        episode_id = episode.get("id")
        family = episode.get("family")
        if not isinstance(episode_id, str) or not episode_id.strip() or episode_id in seen_episodes:
            raise ValueError("episode ids must be unique nonempty strings")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("episode family must be a nonempty string")
        seen_episodes.add(episode_id)

        for role, expected_a0, keys, target in (
            ("h1_wrong", False, ("a0", "blind", "visible", "guided"), wrong),
            ("h1_correct", True, ("a0", "blind", "visible"), correct),
        ):
            row = episode.get(role)
            if not isinstance(row, dict):
                raise ValueError(f"{role} must be an object")
            _require_id(row, seen_tasks)
            for key in keys:
                _require_bool(row, key)
            if row["a0"] is not expected_a0:
                raise ValueError(f"{role}.a0 has wrong initial state")
            target.append({**row, "episode": episode_id})

        for role, count in (("positive", 2), ("near_negative", 1), ("unrelated", 1)):
            items = episode.get(role)
            if not isinstance(items, list) or len(items) != count:
                raise ValueError(f"{role} must have exactly {count} task(s) per episode")
            for row in items:
                if not isinstance(row, dict):
                    raise ValueError(f"{role} task must be an object")
                _require_id(row, seen_tasks)
                for key in S2_ARMS:
                    _require_bool(row, key)
                roles[role].append({**row, "episode": episode_id})

    autonomous = _paired(wrong, "visible", "blind")
    guided_blind = _paired(wrong, "guided", "blind")
    guided_visible = _paired(wrong, "guided", "visible")
    correct_visible = _paired(correct, "visible", "blind")
    autonomous_win_episodes = len({
        row["episode"] for row in wrong if row["visible"] and not row["blind"]
    })
    guided_win_episodes = len({
        row["episode"] for row in wrong
        if row["guided"] and not row["blind"] and not row["visible"]
    })
    s1_autonomous_hint = (
        autonomous["wins"] >= 4 and autonomous["net"] >= 3
        and autonomous_win_episodes >= 3
    )
    s1_autonomous = s1_autonomous_hint and correct_visible["net"] >= -2
    s1_guided = (
        guided_blind["wins"] >= 4 and guided_blind["net"] >= 3
        and guided_visible["wins"] >= 4 and guided_visible["net"] >= 3
        and guided_win_episodes >= 3
    )

    s2: dict[str, Any] = {}
    for treatment in ("a1", "a2"):
        sham = f"{treatment}_sham"
        gain = _paired(roles["positive"], treatment, "a0")
        specific = _paired(roles["positive"], treatment, sham)
        win_episodes = len({
            row["episode"] for row in roles["positive"]
            if row[treatment] and not row["a0"]
        })
        harms = {
            role: sum(row["a0"] and not row[treatment] for row in roles[role])
            for role in ("near_negative", "unrelated")
        }
        behavioral_hint = (
            gain["wins"] >= 4 and gain["net"] >= 3
            and specific["net"] >= 2 and win_episodes >= 3
        )
        s2[treatment] = {
            "versus_a0": gain,
            "versus_sham": specific,
            "win_episodes": win_episodes,
            "a0_correct_to_wrong": harms,
            "behavioral_hint": behavioral_hint,
            "follow_up_signal": behavioral_hint and all(harm <= 2 for harm in harms.values()),
        }

    return {
        "schema_version": "p16_signal_only_cpu_report_v1",
        "synthetic_or_verified_input_only": True,
        "counts": {"episodes": 20, "h1_wrong": len(wrong), "h1_correct": len(correct),
                   **{role: len(rows) for role, rows in roles.items()}},
        "s1": {
            "visible_vs_blind_on_wrong": autonomous,
            "guided_vs_blind_on_wrong": guided_blind,
            "guided_vs_visible_on_wrong": guided_visible,
            "visible_vs_blind_on_initially_correct": correct_visible,
            "autonomous_win_episodes": autonomous_win_episodes,
            "guided_win_episodes": guided_win_episodes,
            "autonomous_behavioral_hint": s1_autonomous_hint,
            "autonomous_follow_up_signal": s1_autonomous,
            "guided_follow_up_signal": s1_guided,
        },
        "s2": s2,
        "a2_vs_a1_positive_descriptive": _paired(roles["positive"], "a2", "a1"),
        "interpretation_limit": "Triage only; no significance, full application, or causal-use claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with independently verified Boolean verdicts")
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(score(document), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
