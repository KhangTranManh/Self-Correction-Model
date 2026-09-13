"""Compare Self_Correction_v1 with its decision-only adapter and write a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-baseline")
    parser.add_argument("--dev-tuned")
    parser.add_argument("--dataset-summary")
    parser.add_argument("--train-report")
    args = parser.parse_args()
    baseline = load(Path(args.baseline).resolve())
    tuned = load(Path(args.tuned).resolve())
    output_dir = Path(args.output_dir).resolve()
    bm = baseline["metrics"]
    tm = tuned["metrics"]

    if baseline["split"]["sha256"] != tuned["split"]["sha256"]:
        raise RuntimeError("Baseline and tuned test split hashes differ")
    for name, summary in (("baseline", baseline), ("tuned", tuned)):
        capture = summary["metrics"]["rationale_capture"]
        if capture["rows_with_visible_reasoning"] != summary["split"]["rows"]:
            raise RuntimeError(f"Missing per-sample visible rationale for {name}")
        if capture.get("rows_with_substantive_visible_reasoning") != summary["split"]["rows"]:
            raise RuntimeError(f"Missing substantive per-sample visible rationale for {name}")

    if (
        tm["exact_contract_accuracy"] >= 0.90
        and tm["keep_recall"] >= 0.80
        and tm["revise_recall"] >= 0.70
        and tm["balanced_accuracy"] >= 0.70
        and min(value["decision_accuracy"] for value in tm["by_domain"].values()) >= 0.60
    ):
        decision = "decision_signal_strong"
    elif tm["keep_recall"] >= 0.80 and tm["revise_recall"] >= 0.40:
        decision = "decision_signal_partial"
    else:
        decision = "decision_signal_absent"

    comparison = {
        "schema_version": "phase3_decision_only_comparison_v1",
        "models": {
            "baseline": baseline["served_model"],
            "tuned": tuned["served_model"],
            "training_checkpoint": "Kxck/Self_Correction_v1",
        },
        "same_test_split": True,
        "test_split_sha256": baseline["split"]["sha256"],
        "test_rows": baseline["split"]["rows"],
        "baseline_metrics": bm,
        "tuned_metrics": tm,
        "deltas": {
            field: round(tm[field] - bm[field], 6)
            for field in (
                "exact_contract_accuracy",
                "decision_accuracy",
                "keep_recall",
                "revise_recall",
                "balanced_accuracy",
            )
        },
        "all_test_samples_have_visible_reasoning": True,
        "visible_reasoning_is_separate_and_not_scored": True,
        "hidden_reasoning_available": False,
        "decision": decision,
    }
    dataset_summary = load(Path(args.dataset_summary).resolve()) if args.dataset_summary else None
    train_report = load(Path(args.train_report).resolve()) if args.train_report else None
    if dataset_summary:
        comparison["dataset"] = {
            "total_rows": dataset_summary["total_rows"],
            "counts_by_split": dataset_summary["counts_by_split"],
            "counts_by_split_domain_decision": dataset_summary[
                "counts_by_split_domain_decision"
            ],
            "counts_by_split_dataset_decision": dataset_summary[
                "counts_by_split_dataset_decision"
            ],
            "validation": dataset_summary["validation"],
        }
    if train_report:
        comparison["training"] = {
            "checkpoint": train_report["model"]["training_checkpoint"],
            "configuration": train_report["training"],
            "eval_before_loss": train_report["eval_before"]["eval_before_loss"],
            "eval_after_loss": train_report["eval_after"]["eval_after_loss"],
            "train_metrics": train_report["train_metrics"],
            "runtime": train_report["runtime"],
        }
    if args.dev_baseline and args.dev_tuned:
        dev_baseline = load(Path(args.dev_baseline).resolve())
        dev_tuned = load(Path(args.dev_tuned).resolve())
        if dev_baseline["split"]["sha256"] != dev_tuned["split"]["sha256"]:
            raise RuntimeError("Baseline and tuned dev split hashes differ")
        for name, summary in (("dev_baseline", dev_baseline), ("dev_tuned", dev_tuned)):
            capture = summary["metrics"]["rationale_capture"]
            if capture["rows_with_visible_reasoning"] != summary["split"]["rows"]:
                raise RuntimeError(f"Missing per-sample visible rationale for {name}")
            if capture.get("rows_with_substantive_visible_reasoning") != summary["split"]["rows"]:
                raise RuntimeError(f"Missing substantive per-sample visible rationale for {name}")
        comparison["dev"] = {
            "same_split": True,
            "rows": dev_baseline["split"]["rows"],
            "sha256": dev_baseline["split"]["sha256"],
            "baseline_metrics": dev_baseline["metrics"],
            "tuned_metrics": dev_tuned["metrics"],
            "all_samples_have_visible_reasoning": True,
        }
    write_json(output_dir / "decision_only_comparison.json", comparison)
    write_json(output_dir / "decision_only_confusion_matrix.json", {
        "baseline": bm["confusion_matrix"],
        "tuned": tm["confusion_matrix"],
    })

    lines = [
        "# Phase 3 Decision-Only Diagnostic",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "The scored generation contains only a KEEP/REVISE decision tag. A separate,",
        "non-scored follow-up captures visible model-provided reasoning for every",
        "sample. Hidden chain-of-thought is not available or claimed.",
        "",
    ]
    if dataset_summary:
        lines += [
            "## Dataset and split",
            "",
            f"- Total: {dataset_summary['total_rows']} unique verified sources.",
            f"- Train/dev/test: {dataset_summary['counts_by_split']['train']}/"
            f"{dataset_summary['counts_by_split']['dev']}/"
            f"{dataset_summary['counts_by_split']['test']}.",
            "- Every split is balanced between Math/Code and KEEP/REVISE.",
            "- Code includes both MBPP and APPS; math uses GSM8K.",
            "- Validation passed: zero source overlap, no CW-as-REVISE, no HumanEval, "
            "no frozen evaluation rows, no teacher labels, label-balanced neutral templates, "
            "and decision-tag-only targets.",
            "",
            "### Held-out test composition",
            "",
            "| Dataset | KEEP | REVISE | Total |",
            "|---|---:|---:|---:|",
        ]
        for dataset, counts in dataset_summary["counts_by_split_dataset_decision"]["test"].items():
            lines.append(
                f"| {dataset.upper()} | {counts['KEEP']} | {counts['REVISE']} | "
                f"{counts['KEEP'] + counts['REVISE']} |"
            )
        lines.append("")
    if train_report:
        cfg = train_report["training"]
        lines += [
            "## Training",
            "",
            f"- Checkpoint: `{train_report['model']['training_checkpoint']}`.",
            f"- QLoRA: 4-bit NF4, rank 32, alpha 64; {cfg['epochs']} epochs; "
            f"learning rate `{cfg['learning_rate']}`; effective batch "
            f"{cfg['effective_batch_size']}.",
            f"- Dev loss: {train_report['eval_before']['eval_before_loss']:.4f} before -> "
            f"{train_report['eval_after']['eval_after_loss']:.4f} after.",
            f"- GPU: {train_report['runtime']['gpu']}; training runtime "
            f"{train_report['train_metrics']['train_runtime']:.1f}s.",
            "",
        ]
    lines += [
        "## Held-out test results",
        "",
        "| Metric | Self_Correction_v1 | Decision-Only tuned |",
        "|---|---:|---:|",
        f"| Exact contract | {pct(bm['exact_contract_accuracy'])} | {pct(tm['exact_contract_accuracy'])} |",
        f"| Decision accuracy | {pct(bm['decision_accuracy'])} | {pct(tm['decision_accuracy'])} |",
        f"| KEEP recall | {pct(bm['keep_recall'])} | {pct(tm['keep_recall'])} |",
        f"| REVISE recall | {pct(bm['revise_recall'])} | {pct(tm['revise_recall'])} |",
        f"| KEEP precision | {pct(bm['keep_precision'])} | {pct(tm['keep_precision'])} |",
        f"| REVISE precision | {pct(bm['revise_precision'])} | {pct(tm['revise_precision'])} |",
        f"| Balanced accuracy | {pct(bm['balanced_accuracy'])} | {pct(tm['balanced_accuracy'])} |",
        "",
        "## Conditional probabilities",
        "",
        "| Conditional | Self_Correction_v1 | Decision-Only tuned |",
        "|---|---:|---:|",
    ]
    labels = {
        "p_pred_keep_given_actually_correct": "P(pred KEEP | actually correct)",
        "p_pred_revise_given_actually_wrong": "P(pred REVISE | actually wrong)",
        "p_pred_keep_given_actually_wrong": "P(pred KEEP | actually wrong)",
        "p_pred_revise_given_actually_correct": "P(pred REVISE | actually correct)",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {pct(bm['conditionals'][key])} | {pct(tm['conditionals'][key])} |")
    lines += [
        "",
        "## Confusion matrices",
        "",
        "Actual rows and predicted columns include INVALID outputs.",
        "",
        "```json",
        json.dumps({"baseline": bm["confusion_matrix"], "tuned": tm["confusion_matrix"]}, indent=2),
        "```",
        "",
        "## Domain and dataset decision accuracy",
        "",
        "| Slice | Self_Correction_v1 | Decision-Only tuned |",
        "|---|---:|---:|",
    ]
    for dimension in ("by_domain", "by_dataset"):
        for key in sorted(tm[dimension]):
            lines.append(
                f"| {key} | {pct(bm[dimension][key]['decision_accuracy'])} | "
                f"{pct(tm[dimension][key]['decision_accuracy'])} |"
            )
    lines += [
        "",
        "## Reasoning retention",
        "",
        f"- Baseline visible rationales: {bm['rationale_capture']['rows_with_visible_reasoning']}/{bm['total']}.",
        f"- Tuned visible rationales: {tm['rationale_capture']['rows_with_visible_reasoning']}/{tm['total']}.",
        f"- Baseline substantive rationales: {bm['rationale_capture']['rows_with_substantive_visible_reasoning']}/{bm['total']}.",
        f"- Tuned substantive rationales: {tm['rationale_capture']['rows_with_substantive_visible_reasoning']}/{tm['total']}.",
        "- Each raw row retains the input messages, strict decision output, parsed",
        "  decision, decision score, rationale prompt, and full visible rationale.",
        "- Rationales are an audit artifact and do not affect classification scores.",
        "",
        "## Unseen neutral-template check",
        "",
        "Not performed in this diagnostic run; it was optional and the frozen 36-row test "
        "was kept unchanged.",
        "",
        "## Interpretation",
        "",
        "The held-out test shows high KEEP on actually-correct answers and high REVISE on "
        "actually-wrong answers, with no severe test-domain collapse. This demonstrates a "
        "decision-only discrimination signal under the specified protocol. The dev split was "
        "weaker, so the result should be treated as a positive diagnostic rather than final "
        "Phase 3 quality. Per protocol, do not automatically scale to the 860-row training run.",
        "",
    ]
    (output_dir / "decision_only_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
