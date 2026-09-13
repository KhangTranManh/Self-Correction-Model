"""Assemble the canonical report for the ordered five-stage experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def probe_metrics(path: Path, label: str) -> dict[str, Any]:
    data = load(path)[label]
    row = next(item for item in data["layers"] if item["layer"] == data["selected_layer"])
    test = row["test"]
    return {
        "selected_layer": data["selected_layer"],
        "balanced_accuracy": test["balanced_accuracy"],
        "keep_recall": test["keep_recall"],
        "revise_recall": test["revise_recall"],
        "roc_auc": test["roc_auc"],
        "code_balanced_accuracy": row["test_subgroups"]["domain"]["code"]["balanced_accuracy"],
        "math_balanced_accuracy": row["test_subgroups"]["domain"]["math"]["balanced_accuracy"],
    }


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="outputs/phase3_five_stage_pipeline")
    args = parser.parse_args()
    root = Path(args.project_root).resolve(); output = (root / args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    stages = output / "stages"
    stage1 = load(stages / "01_frozen_classifier/report.json")
    stage2 = load(stages / "02_decision_token/report.json")
    stage3 = load(stages / "03_shortcut_audit/report.json")
    stage4 = load(stages / "04_same_origin/report.json")
    stage4_data = load(stages / "04_same_origin/data/summary.json")
    stage4_mix = load(stages / "04_same_origin/mixed_data/summary.json")
    stage5 = load(stages / "05_reward_grpo/report.json")
    probe_root = output / "representation_probe/comparisons"
    probes = {
        name: probe_metrics(probe_root / name / "probe_results_by_layer.json", name)
        for name in ("decision_token_v2", "same_origin_v1", "grpo_v1")
    }
    baseline_probe = probe_metrics(probe_root / "decision_token_v2/probe_results_by_layer.json", "decision_only_v1")
    behavior = {
        "decision_only_v1": stage2["baseline"],
        "decision_token_v2": stage2["candidate_metrics"],
        "same_origin_v1": stage4["candidate_metrics"],
        "grpo_v1": stage5["candidate_metrics"],
    }
    report = {
        "schema_version": "phase3_five_stage_pipeline_v1",
        "canonical_checkpoint_before_and_after": "Decision-Only V1",
        "stage_status": {
            "01_frozen_classifier": "pass",
            "02_decision_token_qlora": "fail_not_promoted",
            "03_shortcut_audit": "automated_pass_manual_queue_preserved",
            "04_same_origin_hard_negative": "data_valid_training_fail_not_promoted",
            "05_verifier_reward_grpo": "fail_not_promoted",
        },
        "stage1_classifier": stage1["frozen_test"],
        "behavioral_frozen_200": behavior,
        "shortcut_audit": stage3,
        "same_origin_data": {"generation_candidates": 762, "verified_pairs": stage4_data["accepted_pairs"], "mix": stage4_mix},
        "representation_probe": {"decision_only_v1": baseline_probe, **probes},
        "conclusion": "Only the frozen classifier improved. Every weight-updating pilot increased KEEP bias and reduced REVISE detection; their hidden-state probes also declined, so this is not merely a decoding-policy effect.",
        "promotion": {"promoted_llm_adapter": None, "retained_canonical_router": "Decision-Only V1", "optional_external_router": "Stage 01 frozen classifier"},
    }
    followup_path = output / "frozen_representation_followup_report.json"
    followup = load(followup_path) if followup_path.exists() else None
    if followup:
        report["post_pipeline_frozen_200_followup"] = {
            "verdict": followup["verdict"],
            "precommitted_external_test": followup["precommitted_external_test"],
            "rank_balanced_diagnostic": followup["rank_balanced_diagnostic"],
            "causal_intervention": followup["causal_intervention"],
            "external_classifier_deployable": followup["promotion"]["external_classifier_deployable"],
            "weight_update_authorized": followup["promotion"]["weight_update_authorized"],
        }
        report["conclusion"] = (
            "The frozen classifier preserves a useful external ranking signal but its threshold "
            "does not transfer to frozen-200, and causal readout was not established. Keep "
            "Decision-Only V1 canonical; do not deploy the classifier or update weights until a "
            "new source-disjoint deployment-matched calibration set exists."
        )
        report["promotion"]["optional_external_router"] = "not_deployable_pending_external_calibration"
    (output / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict_text = (
        "Stage 01 passed its original 52-row gate, but its later frozen-200 external test exposed "
        "non-portable calibration. Decision-Only V1 remains canonical; no follow-up artifact is deployable."
        if followup else
        "Only Stage 01 passed. Decision-Only V1 remains the canonical LLM router; the frozen linear classifier is the only improved router candidate."
    )
    lines = [
        "# Phase 3 five-stage router experiment", "", "## Verdict", "",
        verdict_text, "",
        "## Frozen behavioral benchmark (200 rows)", "",
        "| Model | Balanced accuracy | KEEP recall | REVISE recall | Code accuracy | Math accuracy | Result |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    labels = {"decision_only_v1":"Decision-Only V1", "decision_token_v2":"Stage 02 decision-token", "same_origin_v1":"Stage 04 same-origin mix", "grpo_v1":"Stage 05 reward pilot"}
    results = {"decision_only_v1":"baseline", "decision_token_v2":"FAIL", "same_origin_v1":"FAIL", "grpo_v1":"FAIL"}
    comparisons = {"decision_only_v1": stage2, "decision_token_v2": stage2, "same_origin_v1": stage4, "grpo_v1": stage5}
    for name, values in behavior.items():
        comp = comparisons[name]; slices = comp["slices"] if name != "decision_only_v1" else stage2["slices"]
        if name == "decision_only_v1": code = slices["domain"]["code"]["baseline_accuracy"]; math = slices["domain"]["math"]["baseline_accuracy"]
        else: code = slices["domain"]["code"]["candidate_accuracy"]; math = slices["domain"]["math"]["candidate_accuracy"]
        lines.append(f"| {labels[name]} | {pct(values['balanced_accuracy'])} | {pct(values['keep_recall'])} | {pct(values['revise_recall'])} | {pct(code)} | {pct(math)} | {results[name]} |")
    lines += ["", "## Stage 01 frozen classifier", "", f"Balanced accuracy {pct(stage1['frozen_test']['balanced_accuracy'])}, KEEP recall {pct(stage1['frozen_test']['keep_recall'])}, REVISE recall {pct(stage1['frozen_test']['revise_recall'])}. This is the only stage that improved error sensitivity without KEEP collapse.", "", "## Stage 03 shortcut audit", "", f"Grouped answer-style TF-IDF BA {pct(stage3['metrics']['char_tfidf']['balanced_accuracy'])}; surface-only BA {pct(stage3['metrics']['surface_only']['balanced_accuracy'])}; origin-only BA {pct(stage3['metrics']['origin_only']['balanced_accuracy'])}. No strong shortcut was detected.", "", "## Stage 04 data construction", "", f"Generated 762 candidates and retained {stage4_data['accepted_pairs']} same-model-origin pairs after fresh verification. The final 200-row mix contains {stage4_mix['same_origin_pairs']} same-origin and {stage4_mix['cross_origin_fallback_pairs']} fallback problem pairs, balanced 100 KEEP/100 REVISE.", "", "## Representation probe", "", "| Model | Probe BA | KEEP recall | REVISE recall | ROC-AUC | Code BA | Math BA |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (("Decision-Only V1", baseline_probe), ("Stage 02", probes["decision_token_v2"]), ("Stage 04", probes["same_origin_v1"]), ("Stage 05", probes["grpo_v1"])):
        lines.append(f"| {name} | {pct(values['balanced_accuracy'])} | {pct(values['keep_recall'])} | {pct(values['revise_recall'])} | {pct(values['roc_auc'])} | {pct(values['code_balanced_accuracy'])} | {pct(values['math_balanced_accuracy'])} |")
    if followup:
        fixed = followup["precommitted_external_test"]
        rank = followup["rank_balanced_diagnostic"]
        lines += [
            "", "## Post-pipeline frozen-200 follow-up", "",
            f"The precommitted classifier threshold does not transfer: BA {pct(fixed['balanced_accuracy'])}, KEEP recall {pct(fixed['keep_recall'])}, REVISE recall {pct(fixed['revise_recall'])}. Its ROC-AUC remains {pct(fixed['roc_auc'])}, indicating ranking information under severe score shift.", "",
            f"A 50/50 rank-balanced diagnostic reaches BA {pct(rank['balanced_accuracy'])}, but assumes known benchmark prevalence and is not deployment calibration. Linear concatenation and a small MLP generalize worse; layer-28 and layer-21 interventions fail the causal readout gate. See `frozen_representation_followup_report.md`.",
        ]
    lines += ["", "## Research conclusion", "", "All three weight updates moved behavior toward KEEP and reduced frozen REVISE recall. The frozen classifier retains a useful ranking signal on frozen-200, but its absolute threshold collapses and residual intervention does not establish causal use. Keep Decision-Only V1 canonical; do not deploy the classifier or update weights until a new source-disjoint, deployment-matched calibration set exists.", ""]
    (output / "pipeline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
