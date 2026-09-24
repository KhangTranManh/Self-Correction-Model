"""Analyze Phase 6 confidence/probe alignment and rationale-review behavior."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402

CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
DISPLAY = {
    "original_solver": "Original solver",
    "warmstart_v2": "Warm-start V2",
    "correction_sft_v3": "Correction SFT V3",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def correlation(x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"coefficient": None, "p_value": None}
    result = spearmanr(x, y)
    return {"coefficient": float(result.statistic), "p_value": float(result.pvalue)}


def pearson(x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"coefficient": None, "p_value": None}
    result = pearsonr(x, y)
    return {"coefficient": float(result.statistic), "p_value": float(result.pvalue)}


def behavior_metrics(rows: list[dict[str, Any]], *,
                     final_key: str = "final_correct",
                     score_valid_key: str | None = None) -> dict[str, Any]:
    wrong = [row for row in rows if not row["initial_correct"]]
    correct = [row for row in rows if row["initial_correct"]]
    valid = [row for row in rows if row.get("strict_contract_valid")]
    def score_valid(row: dict[str, Any]) -> bool:
        return bool(row.get(score_valid_key)) if score_valid_key else row.get(final_key) is not None
    return {
        "sources": len(rows),
        "strict_valid": len(valid),
        "strict_contract_validity": len(valid) / len(rows),
        "wrong_to_correct": sum(row.get(final_key) is True for row in wrong),
        "wrong_denominator": len(wrong),
        "correct_to_wrong_verified": sum(row.get(final_key) is False and score_valid(row)
                                         for row in correct),
        "correct_invalid": sum(not score_valid(row) for row in correct),
        "correct_denominator": len(correct),
        "final_correct": sum(row.get(final_key) is True for row in rows),
        "final_accuracy_conservative": sum(row.get(final_key) is True for row in rows) / len(rows),
        "decisions": dict(Counter(row.get("decision") for row in valid)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", type=Path,
                        default=ROOT / "outputs/phase6_verbalization_pilot_v1/raw")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase6_verbalization_pilot_v1")
    args = parser.parse_args()
    manifest = read_jsonl(ROOT / "phase6/data/pilot_v1.jsonl")
    pilot_ids = {row["problem_id"] for row in manifest}
    labels = {row["problem_id"]: int(not row["initial_correct"]) for row in manifest}
    source_by_id = {row["problem_id"]: row for row in manifest}
    verifier = MathVerifier()
    report: dict[str, Any] = {
        "schema_version": "phase6_verbalization_pilot_report_v1",
        "scope": {"sources": 16, "initially_correct": 8, "initially_wrong": 8,
                  "split": "phase5_development", "protected_data_used": False},
        "checkpoints": {},
    }

    for checkpoint in CHECKPOINTS:
        generated = read_jsonl(args.pilot_output / f"{checkpoint}.jsonl")
        confidence = {row["problem_id"]: row for row in generated
                      if row["kind"] == "confidence"}
        rationale = [row for row in generated if row["kind"] == "rationale_review"]
        if set(confidence) != pilot_ids or {row["problem_id"] for row in rationale} != pilot_ids:
            raise RuntimeError(f"Incomplete generated rows for {checkpoint}")
        for row in rationale:
            source = source_by_id[row["problem_id"]]
            result = verifier.verify(
                Problem(id=source["problem_id"], domain="math", question=source["question"],
                        reference_answer=source["reference_answer"]),
                row["raw_model_output"],
            )
            row["freeform_final_correct"] = bool(result.passed)
            row["freeform_verifier_detail"] = result.detail
            row["freeform_final_parse_valid"] = not result.detail.startswith(
                ("Khong trich xuat", "Loi parse")
            )

        selection_path = (ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection"
                          / f"{checkpoint}_probe_selection.json")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_layer = int(selection["selected"]["layer"])
        classifier = joblib.load(ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection"
                                 / f"{checkpoint}_probe.joblib")
        archive = np.load(ROOT / "outputs/phase5_gpu_vllm/probe_v1/activations"
                          / f"{checkpoint}_development.npz", allow_pickle=False)
        layers = [int(value) for value in archive["layers"]]
        activations = archive["activations"][:, layers.index(selected_layer), :].astype(np.float32)
        probabilities = classifier.predict_proba(activations)[:, 1]
        probe_by_id = {str(problem_id): float(probability)
                       for problem_id, probability in zip(archive["problem_ids"], probabilities)}

        valid_confidence = [confidence[row["problem_id"]] for row in manifest
                            if confidence[row["problem_id"]]["strict_contract_valid"]]
        ids = [row["problem_id"] for row in valid_confidence]
        verbal_wrong = [1.0 - float(confidence[source_id]["confidence_correct"]) / 100.0
                        for source_id in ids]
        probe_wrong = [probe_by_id[source_id] for source_id in ids]
        truth = [labels[source_id] for source_id in ids]
        confidence_metrics = {
            "valid": len(ids), "total": len(manifest),
            "verbal_wrong_mean": float(np.mean(verbal_wrong)) if ids else None,
            "probe_wrong_mean": float(np.mean(probe_wrong)) if ids else None,
            "verbal_vs_probe_spearman": correlation(verbal_wrong, probe_wrong),
            "verbal_vs_probe_pearson": pearson(verbal_wrong, probe_wrong),
            "verbal_vs_truth_auc": float(roc_auc_score(truth, verbal_wrong)) if len(set(truth)) == 2 else None,
            "probe_vs_truth_auc": float(roc_auc_score(truth, probe_wrong)) if len(set(truth)) == 2 else None,
            "verbal_brier": float(brier_score_loss(truth, verbal_wrong)) if ids else None,
            "probe_brier": float(brier_score_loss(truth, probe_wrong)) if ids else None,
            "mean_absolute_verbal_probe_gap": float(np.mean(np.abs(np.array(verbal_wrong) - np.array(probe_wrong)))) if ids else None,
            "rows": [{"problem_id": source_id, "initial_wrong": bool(labels[source_id]),
                      "verbal_wrong_probability": verbal_wrong[index],
                      "probe_wrong_probability": probe_wrong[index]}
                     for index, source_id in enumerate(ids)],
        }

        direct_path = (ROOT / "outputs/phase5_gpu_vllm/review_v1"
                       / f"{checkpoint}_development.jsonl")
        direct = [row for row in read_jsonl(direct_path) if row["problem_id"] in pilot_ids]
        neutral = [row for row in direct if row["condition"] == "neutral"]
        status = [row for row in direct if row["condition"] == "status"]
        behavior = {
            "neutral_direct": behavior_metrics(neutral),
            "neutral_rationale": behavior_metrics(
                rationale, final_key="freeform_final_correct",
                score_valid_key="freeform_final_parse_valid",
            ),
            "status_direct": behavior_metrics(status),
        }
        behavior["rationale_minus_neutral_fixes"] = (
            behavior["neutral_rationale"]["wrong_to_correct"]
            - behavior["neutral_direct"]["wrong_to_correct"]
        )
        behavior["status_minus_rationale_fixes"] = (
            behavior["status_direct"]["wrong_to_correct"]
            - behavior["neutral_rationale"]["wrong_to_correct"]
        )
        report["checkpoints"][checkpoint] = {
            "display": DISPLAY[checkpoint], "selected_probe_layer": selected_layer,
            "confidence": confidence_metrics, "behavior": behavior,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")
    lines = [
        "# Phase 6 verbalization-path pilot", "",
        "Exploratory 16-source Phase 5 development subset (8 correct / 8 wrong). "
        "The protected set was not used; no weights or probes were fitted.", "",
        "## Confidence versus frozen probe", "",
        "| Checkpoint | Valid | Spearman ρ (p) | Verbal AUC | Probe AUC | Verbal Brier | Probe Brier | Mean abs gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        c = report["checkpoints"][checkpoint]["confidence"]
        s = c["verbal_vs_probe_spearman"]
        corr = "NA" if s["coefficient"] is None else f"{s['coefficient']:.3f} ({s['p_value']:.3f})"
        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.3f}"
        lines.append(f"| {DISPLAY[checkpoint]} | {c['valid']}/16 | {corr} | "
                     f"{fmt(c['verbal_vs_truth_auc'])} | {fmt(c['probe_vs_truth_auc'])} | "
                     f"{fmt(c['verbal_brier'])} | {fmt(c['probe_brier'])} | "
                     f"{fmt(c['mean_absolute_verbal_probe_gap'])} |")
    lines += ["", "## Neutral rationale behavior", "",
              "Rationale final accuracy is verifier-scored from the complete raw model output even when the requested XML decision contract is invalid.", "",
              "| Checkpoint | Condition | Valid | Wrong→correct | Correct→wrong | Correct invalid | Final accuracy |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        behavior = report["checkpoints"][checkpoint]["behavior"]
        for key, label in (("neutral_direct", "Neutral direct"),
                           ("neutral_rationale", "Neutral + rationale"),
                           ("status_direct", "Status direct")):
            row = behavior[key]
            lines.append(f"| {DISPLAY[checkpoint]} | {label} | {row['strict_valid']}/16 | "
                         f"{row['wrong_to_correct']}/8 | {row['correct_to_wrong_verified']}/8 | "
                         f"{row['correct_invalid']}/8 | {100*row['final_accuracy_conservative']:.1f}% |")
    lines += [
        "", "## Interpretation boundary", "",
        "Confidence/probe association on 16 development sources is a pathway diagnostic, not causal proof. "
        "A weak association may reflect quantized confidence language, prompt sensitivity, or low power in addition to a verbalization gap.",
        "",
        "Generated `<review>` text is model-visible output retained for audit. It is not hidden chain-of-thought and is not training data.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({"json": str(json_path), "markdown": str(args.output_dir / 'report.md')}, indent=2))


if __name__ == "__main__":
    main()
