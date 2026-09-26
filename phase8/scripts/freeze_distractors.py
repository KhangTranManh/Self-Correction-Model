"""Freeze deterministic, length-matched Phase 7 wrong-answer distractors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "phase8/data/distractors_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Refusing to overwrite frozen donor map")
    first = ROOT / "outputs/phase8_first_pass_v1/initial/answers.jsonl"
    first_summary = json.loads(first.with_name("summary.json").read_text(encoding="utf-8"))
    if first_summary["rows"] != 400 or sha256(first) != first_summary["answers_sha256"]:
        raise RuntimeError("First-pass source is incomplete or changed")
    old = ROOT / "outputs/phase7_initials_v1/initial_rollouts.jsonl"
    old_summary = json.loads(old.with_name("summary.json").read_text(encoding="utf-8"))
    if sha256(old) != old_summary["rollouts_sha256"]:
        raise RuntimeError("Historical donor answers changed")
    targets = read_jsonl(first)
    donors = [row for row in read_jsonl(old) if row["initial_correct"] is False]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    donor_lengths = {
        row["problem_id"]: len(tokenizer.encode(row["initial_output"], add_special_tokens=False))
        for row in donors
    }
    assignments = []
    for target in targets:
        pid = target["problem_id"]
        length = len(tokenizer.encode(target["output"], add_special_tokens=False))
        eligible = [row for row in donors if row["problem_id"] != pid and
                    abs(donor_lengths[row["problem_id"]] - length) <= 0.10 * max(1, length)]
        eligible.sort(key=lambda row: hashlib.sha256(
            f"phase8_distractor_v1|{pid}|{row['problem_id']}".encode()).hexdigest())
        chosen = eligible[0] if eligible else None
        assignments.append({
            "problem_id": pid, "target_token_length": length,
            "eligible_donors": len(eligible),
            "donor_problem_id": chosen["problem_id"] if chosen else None,
            "donor_token_length": donor_lengths[chosen["problem_id"]] if chosen else None,
            "donor_output": chosen["initial_output"] if chosen else None,
        })
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "assignments.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in assignments), encoding="utf-8", newline="\n")
    report = {
        "schema_version": "phase8_distractor_map_v1",
        "initial_answers_sha256": sha256(first),
        "historical_donors_sha256": sha256(old),
        "tokenizer_model": MODEL, "tokenizer_revision": REVISION,
        "target_rows": len(targets), "historical_wrong_donors": len(donors),
        "matched": sum(row["donor_problem_id"] is not None for row in assignments),
        "unmatched": sum(row["donor_problem_id"] is None for row in assignments),
        "assignments_sha256": sha256(output),
        "protected_correctness_labels_read": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
