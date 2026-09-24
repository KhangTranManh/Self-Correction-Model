"""Analyze the frozen fuzzy-hint curve against existing Phase 5 baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
DISPLAY = {
    "original_solver": "Original solver",
    "warmstart_v2": "Warm-start V2",
    "correction_sft_v3": "Correction SFT V3",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if row["initial_correct"]]
    wrong = [row for row in rows if not row["initial_correct"]]
    valid = [row for row in rows if row.get("strict_contract_valid")]
    return {
        "rows": len(rows),
        "strict_valid": len(valid),
        "strict_contract_validity": len(valid) / len(rows),
        "wrong_to_correct": sum(row.get("transition") == "wrong_to_correct" for row in wrong),
        "wrong_denominator": len(wrong),
        "correct_to_wrong": sum(row.get("transition") == "correct_to_wrong" for row in correct),
        "correct_denominator": len(correct),
        "correct_invalid": sum(not row.get("strict_contract_valid") for row in correct),
        "final_correct": sum(row.get("final_correct") is True for row in rows),
        "final_accuracy_conservative": sum(row.get("final_correct") is True for row in rows) / len(rows),
        "revise_on_wrong": sum(row.get("decision") == "REVISE" for row in wrong),
        "revise_on_correct": sum(row.get("decision") == "REVISE" for row in correct),
        "decisions": dict(Counter(row.get("decision") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path,
                        default=ROOT / "outputs/phase6_fuzzy_hint_pilot_v1/raw")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase6_fuzzy_hint_pilot_v1")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "phase6/configs/fuzzy_hint_pilot_v1.yaml").read_text(
        encoding="utf-8"))
    lock_path = ROOT / "phase6/data/fuzzy_hint_pilot_v1_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256_lf(ROOT / relative) != expected["sha256_lf"]:
            raise RuntimeError(f"Frozen input changed: {relative}")
    manifest = read_jsonl(ROOT / "phase6/data/fuzzy_hint_pilot_v1.jsonl")
    pilot_ids = {row["problem_id"] for row in manifest}
    probabilities = [int(value) for value in config["intervention"]["probability_wrong_percent"]]
    report: dict[str, Any] = {
        "schema_version": "phase6_fuzzy_hint_report_v1",
        "scope": {"sources": 8, "initially_correct": 4, "initially_wrong": 4,
                  "probabilities_wrong_percent": probabilities,
                  "split": "phase5_development", "protected_data_used": False,
                  "prior_phase6_pilot_overlap": 0},
        "checkpoints": {},
    }
    for checkpoint in CHECKPOINTS:
        generated_path = args.raw_dir / f"{checkpoint}.jsonl"
        generated = read_jsonl(generated_path)
        expected_keys = {(source_id, probability) for source_id in pilot_ids
                         for probability in probabilities}
        observed_keys = {(row["problem_id"], int(row["probability_wrong_percent"]))
                         for row in generated}
        if observed_keys != expected_keys or len(generated) != len(expected_keys):
            raise RuntimeError(f"Incomplete or duplicate generated rows for {checkpoint}")
        phase5 = read_jsonl(ROOT / "outputs/phase5_gpu_vllm/review_v1"
                            / f"{checkpoint}_development.jsonl")
        baselines = {}
        for condition in ("neutral", "status"):
            selected = [row for row in phase5
                        if row["problem_id"] in pilot_ids and row["condition"] == condition]
            if len(selected) != 8:
                raise RuntimeError(f"Missing {condition} baseline for {checkpoint}")
            baselines[condition] = metrics(selected)
        curve = {}
        for probability in probabilities:
            subset = [row for row in generated
                      if int(row["probability_wrong_percent"]) == probability]
            curve[str(probability)] = metrics(subset)
        anchor = curve[str(probabilities[0])]
        eligible = [probability for probability in probabilities[1:]
                    if curve[str(probability)]["wrong_to_correct"] > anchor["wrong_to_correct"]
                    and curve[str(probability)]["correct_to_wrong"] <= anchor["correct_to_wrong"]]
        report["checkpoints"][checkpoint] = {
            "display": DISPLAY[checkpoint],
            "raw_sha256_lf": sha256_lf(generated_path),
            "baselines": baselines,
            "curve": curve,
            "descriptive_operating_point_percent": min(eligible) if eligible else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "# Phase 6 fuzzy-hint curve pilot", "",
        "Exploratory 8-source development pilot (4 initially correct / 4 initially wrong), "
        "disjoint from the prior Phase 6 pilot. The same probability grid was applied "
        "to both classes; the protected set was not used.", "",
        "## Benefit-safety curve", "",
        "| Checkpoint | Condition | Valid | Wrong→correct | Correct→wrong | Revise wrong | Revise correct | Final accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        result = report["checkpoints"][checkpoint]
        for condition, metric in (("Neutral", result["baselines"]["neutral"]),
                                  *[(f"P(wrong)={p}%", result["curve"][str(p)])
                                    for p in probabilities],
                                  ("Truthful status", result["baselines"]["status"])):
            lines.append(
                f"| {DISPLAY[checkpoint]} | {condition} | {metric['strict_valid']}/8 | "
                f"{metric['wrong_to_correct']}/4 | {metric['correct_to_wrong']}/4 | "
                f"{metric['revise_on_wrong']}/4 | {metric['revise_on_correct']}/4 | "
                f"{100*metric['final_accuracy_conservative']:.1f}% |"
            )
    lines += ["", "## Descriptive operating point", ""]
    for checkpoint in CHECKPOINTS:
        point = report["checkpoints"][checkpoint]["descriptive_operating_point_percent"]
        rendered = f"{point}%" if point is not None else "none observed"
        lines.append(f"- {DISPLAY[checkpoint]}: {rendered}.")
    lines += [
        "", "The operating point is the lowest level above 50% with more fixes and no "
        "additional verified correct→wrong transition relative to 50%. Four rows per "
        "class are far too few to estimate a deployable threshold; ties and invalid "
        "outputs are reported rather than hidden.", "",
        "## Interpretation boundary", "",
        "This measures sensitivity to an externally supplied belief, not autonomous "
        "error detection. A rising fix curve accompanied by rising revision on correct "
        "answers indicates harness dependence and sycophancy rather than safer self-correction.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({checkpoint: report["checkpoints"][checkpoint][
        "descriptive_operating_point_percent"] for checkpoint in CHECKPOINTS}, indent=2))


if __name__ == "__main__":
    main()

