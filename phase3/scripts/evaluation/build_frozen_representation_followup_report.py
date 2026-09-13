"""Build the canonical report for the frozen-200 representation follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", default="outputs/phase3_five_stage_pipeline")
    args = parser.parse_args()
    root = Path(args.pipeline_dir).resolve()
    external = load(root / "frozen_200_classifier_eval/report.json")
    variants = load(root / "external_probe_variants/report.json")
    stability = load(root / "external_probe_variants/stability_report.json")
    alignment = load(root / "frozen_200_classifier_eval/alignment_report.json")
    intervention_last = load(root / "probe_direction_intervention/report.json")
    intervention_21 = load(root / "probe_direction_intervention_layer21/report.json")
    variant_map = {row["name"]: row for row in variants["variants"]}

    report = {
        "schema_version": "phase3_frozen_representation_followup_v1",
        "verdict": "decodable_ranking_signal_but_uncalibrated_and_not_causally_validated",
        "data_integrity": {
            "training_probe_rows": variants["training_rows"],
            "external_frozen_rows": variants["external_rows"],
            "source_overlap": variants["source_overlap"],
            "external_balance": {"KEEP": 100, "REVISE": 100},
        },
        "precommitted_external_test": external["metrics"],
        "precommitted_external_bootstrap_95": external["bootstrap_95"],
        "repeated_source_disjoint_validation": stability["summary"],
        "rank_balanced_diagnostic": alignment["rank_balanced"],
        "native_behavior": alignment["native"],
        "policy_alignment": {
            "score_auc_expected_revise": alignment["score_auc_for_expected_revise"],
            "score_auc_native_revise": alignment["score_auc_for_native_revise"],
            "native_rank_agreement": alignment["native_rank_agreement"],
            "net_correct_gain_from_rank": alignment["net_correct_gain_from_rank"],
        },
        "capacity_variants": {
            name: {
                "external_fixed": row["external_fixed_threshold"],
                "external_rank_balanced": row["external_rank_balanced_diagnostic"],
                "selected": row["selection"],
            }
            for name, row in variant_map.items()
        },
        "causal_intervention": {
            "layer_28_gate": intervention_last["causal_readout_gate"],
            "layer_28_slope": intervention_last["mean_log_odds_slope"],
            "layer_21_gate": intervention_21["causal_readout_gate"],
            "layer_21_slope": intervention_21["mean_log_odds_slope"],
        },
        "promotion": {
            "external_classifier_deployable": False,
            "weight_update_authorized": False,
            "canonical_router": "Decision-Only V1",
        },
        "next_gate": {
            "action": "collect a new representative, source-disjoint calibration set; do not tune on frozen-200",
            "recommended_rows": "100-200, balanced audit plus natural-prevalence view",
            "requirements": [
                "same dataset/domain/template mix as intended deployment",
                "fresh verifier labels",
                "no overlap with probe-256 or frozen-200",
                "pre-register threshold rule and KEEP floor",
            ],
        },
    }
    (root / "frozen_representation_followup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fixed = external["metrics"]
    rank = alignment["rank_balanced"]
    lines = [
        "# Frozen-200 representation follow-up", "", "## Verdict", "",
        "Decision-Only V1 contains a repeatable, approximately linear ranking signal for correctness, but the saved probability threshold does not transfer to the frozen-200 distribution. The probe direction also failed the causal readout diagnostic. Do not deploy this classifier or update LLM weights yet.", "",
        "## Main results", "",
        "| Test | BA | KEEP recall | REVISE recall | ROC-AUC | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
        f"| Native Decision-Only V1 | {pct(alignment['native']['balanced_accuracy'])} | 81.0% | 49.0% | n/a | Canonical behavior |",
        f"| Precommitted classifier threshold | {pct(fixed['balanced_accuracy'])} | {pct(fixed['keep_recall'])} | {pct(fixed['revise_recall'])} | {pct(fixed['roc_auc'])} | Threshold collapse |",
        f"| Rank-balanced diagnostic | {pct(rank['balanced_accuracy'])} | 67.0% | 67.0% | {pct(fixed['roc_auc'])} | Uses known 50/50 prevalence; not deployment-ready |", "",
        f"External ROC-AUC bootstrap 95% interval: {external['bootstrap_95']['roc_auc']['lower_95']:.3f}-{external['bootstrap_95']['roc_auc']['upper_95']:.3f}. Math AUC is {external['subgroups']['domain']['math']['roc_auc']:.3f}; code AUC is {external['subgroups']['domain']['code']['roc_auc']:.3f}.", "",
        "## Stability and capacity", "",
        f"Across 100 repeated source-disjoint folds, mean BA is {stability['summary']['balanced_accuracy']['mean']:.3f} and mean ROC-AUC is {stability['summary']['roc_auc']['mean']:.3f}. However, mean KEEP recall is only {stability['summary']['keep_recall']['mean']:.3f}, confirming threshold instability.", "",
        "Four-layer concatenation and the small MLP both generalize worse than the selected single-layer linear ranking. More classifier capacity is not the current bottleneck.", "",
        "## Policy and causal diagnostics", "",
        f"The hidden score predicts expected REVISE with AUC {alignment['score_auc_for_expected_revise']:.3f} and native REVISE with AUC {alignment['score_auc_for_native_revise']:.3f}. Native/rank agreement is {alignment['native_rank_agreement']:.1%}; rank balancing improves only {alignment['net_correct_gain_from_rank']:+d} net rows.", "",
        f"Residual intervention failed at both layer 28 and layer 21 (layer-21 slope {intervention_21['mean_log_odds_slope']:.6f}). The probe direction is decodable, but this experiment does not show that it is a causal decision feature.", "",
        "## Decision", "",
        "Keep Decision-Only V1 canonical. Do not train auxiliary LoRA yet. First collect 100-200 new, source-disjoint calibration rows matching the intended deployment distribution, with a pre-registered threshold rule and KEEP-recall floor. Frozen-200 remains test-only.", "",
    ]
    (root / "frozen_representation_followup_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
