"""Build disjoint pilot episodes from screened dev problems.

A type is kept only when its base rate is at least 10% and at least 20 eligible
items were scored. An episode uses one teaching problem, two probes, and one
control. The teaching text must not contain a probe or control gold answer or
an 8-word span of that problem.
"""

from __future__ import annotations

try:
    from phase15.scripts.self_edit_gate import number_values, protected_overlap
except ImportError:  # Kaggle kernel directory
    from self_edit_gate import number_values, protected_overlap


SOLVE_TYPES = ("S-format", "S-loop", "S-integer")
JUDGE_TYPES = ("J-format", "J-loop", "J-resolve", "J-third")


def keep_types(rates: dict[str, dict], *, minimum_rate: float, minimum_eligible: int) -> list[str]:
    kept = []
    for name, row in rates.items():
        eligible = int(row["eligible"])
        if eligible < minimum_eligible:
            continue
        if row["count"] / eligible >= minimum_rate:
            kept.append(name)
    return kept


def _gold_values(reference: str) -> set[str]:
    return number_values(reference)


def leaks(teaching_text: str, others: list[dict]) -> bool:
    if protected_overlap(teaching_text, [row["question"] for row in others]):
        return True
    present = number_values(teaching_text)
    for row in others:
        if present & _gold_values(str(row["reference_answer"])):
            return True
    return False


def build_episodes(records: list[dict], error_type: str, *, limit: int) -> list[dict]:
    """Rows need ``id``, ``question``, ``reference_answer``, ``errors``, and ``correct``.

    A judge row may set ``source_text`` to the full judgment prompt and
    ``correction`` to the verified verdict text.
    """
    erred = [row for row in records if error_type in row.get("errors", [])]
    controls = [row for row in records if row.get("correct") and error_type not in row.get("errors", [])]
    used: set[str] = set()
    episodes = []
    for teaching in erred:
        if len(episodes) >= limit or teaching["id"] in used:
            continue
        candidates = [row for row in erred if row["id"] not in used and row["id"] != teaching["id"]]
        chosen = None
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                probes = [candidates[left], candidates[right]]
                blocked = {teaching["id"], probes[0]["id"], probes[1]["id"]}
                control = next((row for row in controls if row["id"] not in used and row["id"] not in blocked), None)
                if control is None:
                    continue
                target = teaching.get("correction") or (
                    "The verified correction is the behavior, not a new problem.\n"
                    f"Đáp số: {teaching['reference_answer']}"
                )
                teaching_text = f"{teaching.get('source_text', teaching['question'])}\n{target}"
                if leaks(teaching_text, probes + [control]):
                    continue
                chosen = (probes, control, target)
                break
            if chosen is not None:
                break
        if chosen is None:
            used.add(teaching["id"])
            continue
        probes, control, target = chosen
        for row in (teaching, *probes, control):
            used.add(row["id"])
        episodes.append({
            "episode_id": f"{error_type.lower()}_{len(episodes):02d}",
            "error_type": error_type,
            "teaching": teaching,
            "target": target,
            "probes": probes,
            "control": control,
        })
    return episodes
