"""CPU checks for the Phase 15 self-edit gate and candidate detectors.

The gate is applied to the eight saved Kaggle smoke lessons. The detectors are
checked on fixed fixtures and then counted on the saved smoke completions.
Neither result is Phase 15 evidence. No lesson or model output is written.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase15/scripts"))
sys.path.insert(0, str(ROOT / "phase1"))

from phase14.scripts.answers import parse  # noqa: E402
from phase15.scripts.detectors import judge_error, solve_errors  # noqa: E402
from phase15.scripts.self_edit_gate import review  # noqa: E402


AUDIT = ROOT / "outputs/phase15_kaggle_smoke_v1/v2/phase15_smoke_v1/audit.jsonl"
OUT = ROOT / "phase15/data/preparation_v1"
SMOKE_PY = ROOT / "phase15/kaggle/smoke_v1/phase15_kaggle_smoke.py"


def load_smoke():
    spec = importlib.util.spec_from_file_location("phase15_kaggle_smoke", SMOKE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_fixture() -> None:
    clean = (
        "General rule: Check the operation named by the problem before writing the final line.\n\n"
        "Ideal response template:\n"
        "State the operation, compute it, and end in the requested format. Name either shown solution."
    )
    accepted = review(clean, forbidden_numbers=set(), protected_texts=[])
    if accepted["status"] != "accept":
        raise AssertionError(accepted["reasons"])

    marked = (
        "General rule: Recheck equal groups before answering a new problem.\n\n"
        "Ideal response template:\n"
        "1. Name the groups.\n"
        "2. Name the size.\n"
        "3. Multiply them.\n"
    )
    normalized = review(marked, forbidden_numbers={3}, protected_texts=[])
    if normalized["status"] != "normalize" or "3" in normalized["text"]:
        raise AssertionError((normalized["status"], normalized["reasons"], normalized["text"]))
    again = review(normalized["text"], forbidden_numbers={3}, protected_texts=[])
    if again["status"] != "accept":
        raise AssertionError(again["reasons"])

    leaked = (
        "General rule: The product in the teaching problem is 12 and should be copied forward.\n\n"
        "Ideal response template:\n"
        "Write 12 on the final line of every new problem."
    )
    rejected = review(leaked, forbidden_numbers={12}, protected_texts=[])
    if rejected["status"] != "retry" or "forbidden_number" not in rejected["reasons"]:
        raise AssertionError(rejected)

    phrase = "alpha bravo charlie delta echo foxtrot golf hotel"
    copied = (
        "General rule: alpha bravo charlie delta echo foxtrot golf hotel is the protected problem.\n\n"
        "Ideal response template:\n"
        "Restate that protected problem before answering it again."
    )
    overlapped = review(copied, forbidden_numbers=set(), protected_texts=[phrase])
    if overlapped["status"] != "retry" or "protected_ngram" not in overlapped["reasons"]:
        raise AssertionError(overlapped)


def smoke_gate(episodes, rows: list[dict]) -> dict:
    by_id = {episode["episode_id"]: episode for episode in episodes}
    edits = [row for row in rows if row.get("type") == "self_edit"]
    if len(edits) != 8:
        raise AssertionError(f"Expected 8 self-edits, found {len(edits)}")
    counts = Counter()
    details = []
    for row in edits:
        episode = by_id[row["episode_id"]]
        forbidden = set()
        for item in (episode["teaching"], *episode["probes"], *episode["controls"]):
            forbidden.update(item["numbers"])
        protected = [item["prompt"] for item in (*episode["probes"], *episode["controls"])]
        result = review(row["output"], forbidden_numbers=forbidden, protected_texts=protected)
        counts[result["status"]] += 1
        if result["status"] == "retry":
            raise AssertionError((row["episode_id"], result["reasons"]))
        second = review(result["text"], forbidden_numbers=forbidden, protected_texts=protected)
        if second["status"] != "accept":
            raise AssertionError((row["episode_id"], second["reasons"]))
        details.append({
            "episode_id": row["episode_id"],
            "status": result["status"],
            "reasons": result["reasons"],
            "list_markers": result["list_markers"],
            "stripped_verdict": result["stripped_verdict"],
        })
    if counts["normalize"] != 8:
        raise AssertionError(dict(counts))
    return {"self_edits": 8, "by_status": dict(counts), "episodes": details, "scientific_evidence": False}


def detector_fixtures() -> None:
    shown_a, shown_b = parse("9"), parse("4")
    format_text = "Kiểm tra.\nĐáp số: Kết quả là 9 trứng."
    if judge_error(format_text, shown_a, shown_b, hit_token_cap=False) != "J-format":
        raise AssertionError("J-format fixture")
    loop = " ".join(["the model keeps repeating this exact checking sentence once again"] * 3)
    if judge_error(loop, shown_a, shown_b, hit_token_cap=True) != "J-loop":
        raise AssertionError("J-loop fixture")
    resolve = (
        "Step 1: ignore the two shown solutions and start over.\n"
        "Step 2: add the quantities in a fresh calculation.\n"
        "Step 3: the working gives 424242.\n"
    )
    if judge_error(resolve, shown_a, shown_b, hit_token_cap=False) != "J-resolve":
        raise AssertionError("J-resolve fixture")
    third = "The other result is 424242."
    if judge_error(third, shown_a, shown_b, hit_token_cap=False) != "J-third":
        raise AssertionError("J-third fixture")
    valid = "Checked both solutions.\nVerdict: Solution A\nĐáp số: 9"
    if judge_error(valid, shown_a, shown_b, hit_token_cap=False) is not None:
        raise AssertionError("valid judgment was typed as an error")
    if solve_errors(format_text, "How many eggs does each child receive?", hit_768_cap=False) != ["S-format"]:
        raise AssertionError("S-format fixture")
    if "S-loop" not in solve_errors(loop, "Add nine and four.", hit_768_cap=True):
        raise AssertionError("S-loop fixture")
    counted = solve_errors("Đáp số: 7/2", "How many apples are left?", hit_768_cap=False)
    if counted != ["S-integer"]:
        raise AssertionError(counted)
    measured = solve_errors("Đáp số: 7/2", "How many meters of rope remain?", hit_768_cap=False)
    if measured:
        raise AssertionError(measured)


def _shown(item: dict) -> tuple[int, int]:
    """Solution A and Solution B values stored on a synthetic judge item."""
    left, right, gold, wrong = item["numbers"]
    if item["expected"] == "A":
        return gold, wrong
    if item["expected"] == "B":
        return wrong, gold
    raise AssertionError(item["expected"])


def main() -> None:
    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)
    gate_fixture()
    detector_fixtures()
    smoke = load_smoke()
    episodes = smoke.build_episodes()
    rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line.strip()]
    gate_report = smoke_gate(episodes, rows)
    detector_report = _count_smoke(episodes, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "self_edit_gate.json").write_text(
        json.dumps(gate_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (OUT / "detector_dry_run.json").write_text(
        json.dumps(detector_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"gate": gate_report["by_status"], "detectors": detector_report}, indent=2))


def _count_smoke(episodes, rows: list[dict]) -> dict:
    items = {}
    for episode in episodes:
        for item in (*episode["probes"], *episode["controls"]):
            items[item["item_id"]] = item
    judge_counts, solve_counts = Counter(), Counter()
    judge_rows = solve_rows = 0
    for row in rows:
        if row.get("type") != "completion":
            continue
        item = items[row["item_id"]]
        if item["kind"] == "judge":
            judge_rows += 1
            shown_a, shown_b = _shown(item)
            kind = judge_error(
                row["output"], parse(str(shown_a)), parse(str(shown_b)),
                hit_token_cap=bool(row.get("score", {}).get("hit_cap")),
            )
            judge_counts[kind or "none"] += 1
        else:
            solve_rows += 1
            found = solve_errors(
                row["output"], item["prompt"], hit_768_cap=bool(row.get("score", {}).get("hit_cap")))
            solve_counts[",".join(found) or "none"] += 1
    if judge_rows != 64 or solve_rows != 32:
        raise AssertionError((judge_rows, solve_rows))
    remote = ROOT / "outputs/phase14_remote/outputs/phase14_v1"
    return {
        "scientific_evidence": False,
        "saved_outputs": "kaggle smoke completions only",
        "phase14_outputs_present": remote.exists(),
        "fixtures_passed": True,
        "smoke_judge_completions": judge_rows,
        "smoke_judge_types": dict(sorted(judge_counts.items())),
        "smoke_solve_completions": solve_rows,
        "smoke_solve_types": dict(sorted(solve_counts.items())),
    }


if __name__ == "__main__":
    main()
