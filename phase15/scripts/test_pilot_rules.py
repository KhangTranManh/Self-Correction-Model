"""CPU checks for pilot scoring and episode construction. No model call."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))

from phase14.scripts.answers import parse  # noqa: E402
from phase15.scripts.detectors import judge_error as detector_judge  # noqa: E402
from phase15.scripts.detectors import solve_errors as detector_solve  # noqa: E402
from phase15.scripts.pilot_episodes import build_episodes, keep_types, leaks  # noqa: E402
from phase15.scripts.pilot_score import judge_error, solve_errors  # noqa: E402


def scores_agree() -> None:
    shown_a, shown_b = parse("9"), parse("4")
    format_text = "Kiểm tra.\nĐáp số: Kết quả là 9 trứng."
    loop = " ".join(["the model keeps repeating this exact checking sentence once again"] * 3)
    resolve = (
        "Step 1: ignore the two shown solutions and start over.\n"
        "Step 2: add the quantities in a fresh calculation.\n"
        "Step 3: the working gives 424242.\n"
    )
    third = "The other result is 424242."
    cases = (
        (format_text, False, "J-format"),
        (loop, True, "J-loop"),
        (resolve, False, "J-resolve"),
        (third, False, "J-third"),
    )
    for text, capped, expected in cases:
        left = judge_error(text, shown_a, shown_b, hit_token_cap=capped)
        right = detector_judge(text, shown_a, shown_b, hit_token_cap=capped)
        if left != expected or left != right:
            raise AssertionError((expected, left, right))
    if solve_errors(format_text, "How many eggs does each child receive?", hit_768_cap=False) != ["S-format"]:
        raise AssertionError("pilot S-format")
    if detector_solve(format_text, "How many eggs does each child receive?", hit_768_cap=False) != ["S-format"]:
        raise AssertionError("detector S-format")
    counted = solve_errors("Đáp số: 7/2", "How many apples are left?", hit_768_cap=False)
    if counted != ["S-integer"]:
        raise AssertionError(counted)


def episodes_respect_gold() -> None:
    def row(name: str, answer: str, bad: bool) -> dict:
        return {
            "id": name,
            "question": {
                "t": "alpha one two three four five six seven eight",
                "p1": "bravo one two three four five six seven eight",
                "p2": "charlie red blue green orange purple yellow black",
                "c": "delta north south east west river mountain valley",
                "p3": "echo copper silver iron nickel zinc lead tin",
            }[name],
            "reference_answer": answer,
            "errors": ["S-format"] if bad else [],
            "correct": not bad,
        }
    records = [
        row("t", "5", True),
        row("p1", "5", True),
        row("p2", "9", True),
        row("c", "11", False),
        row("p3", "13", True),
    ]
    made = build_episodes(records, "S-format", limit=5)
    if len(made) != 1 or made[0]["probes"][0]["id"] == "p1":
        raise AssertionError([episode["episode_id"] for episode in made])
    if leaks("Đáp số: 11", [records[3]]):
        pass
    else:
        raise AssertionError("control gold was not detected")
    rates = {"S-format": {"count": 2, "eligible": 10}, "S-loop": {"count": 5, "eligible": 19}}
    if keep_types(rates, minimum_rate=0.1, minimum_eligible=20) != []:
        raise AssertionError("small or rare types were kept")
    rates["S-format"]["eligible"] = 20
    if keep_types(rates, minimum_rate=0.1, minimum_eligible=20) != ["S-format"]:
        raise AssertionError("a 10% type on 20 items was dropped")


if __name__ == "__main__":
    scores_agree()
    episodes_respect_gold()
    print("pilot rules ok")
