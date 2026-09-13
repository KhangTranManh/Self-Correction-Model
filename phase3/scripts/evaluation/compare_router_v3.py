"""Compare cached Decision-Only V1 and Router V3 on the frozen benchmark."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-rows", required=True)
    parser.add_argument("--v1-summary", required=True)
    parser.add_argument("--v3-rows", required=True)
    parser.add_argument("--v3-summary", required=True)
    parser.add_argument("--train-report", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    v1_path = Path(args.v1_rows).resolve()
    v3_path = Path(args.v3_rows).resolve()
    v1_rows = {row["source_id"]: row for row in read_jsonl(v1_path)}
    v3_rows = {row["source_id"]: row for row in read_jsonl(v3_path)}
    if set(v1_rows) != set(v3_rows) or len(v1_rows) != 200:
        raise RuntimeError("V1 and V3 must cover the same 200 frozen source IDs")

    prompt_mismatches = []
    paired = []
    for source_id in sorted(v1_rows):
        old = v1_rows[source_id]
        new = v3_rows[source_id]
        expected = str(old["expected_decision"])
        if expected != str(new["expected_decision"]):
            raise RuntimeError(f"Expected-label mismatch for {source_id}")
        messages = new["input_messages"]
        if (
            messages[0]["content"] != old["task_prompt"]
            or messages[1]["content"] != old["initial_answer"]
            or messages[2]["content"] != "Review your previous answer carefully and decide whether it should be kept or revised."
        ):
            prompt_mismatches.append(source_id)
        old_pred = str(old["decision"])
        new_pred = str(new["predicted_decision"])
        paired.append(
            {
                "source_id": source_id,
                "dataset": old["dataset"],
                "domain": old["domain"],
                "expected_decision": expected,
                "v1_decision": old_pred,
                "v3_decision": new_pred,
                "v1_correct": old_pred == expected,
                "v3_correct": new_pred == expected,
                "transition": f"{old_pred}->{new_pred}",
            }
        )
    if prompt_mismatches:
        raise RuntimeError(f"Prompt mismatches: {prompt_mismatches[:5]}")

    v1_summary = read_json(Path(args.v1_summary).resolve())
    v3_summary = read_json(Path(args.v3_summary).resolve())
    train_report = read_json(Path(args.train_report).resolve())
    dataset_summary = read_json(Path(args.dataset_summary).resolve())
    old = v1_summary["metrics"]
    new = v3_summary["metrics"]
    v1_only = sum(row["v1_correct"] and not row["v3_correct"] for row in paired)
    v3_only = sum(row["v3_correct"] and not row["v1_correct"] for row in paired)

    transitions = {
        actual: dict(sorted(Counter(
            row["transition"] for row in paired if row["expected_decision"] == actual
        ).items()))
        for actual in ("KEEP", "REVISE")
    }
    by_domain = {}
    for domain in sorted({row["domain"] for row in paired}):
        subset = [row for row in paired if row["domain"] == domain]
        by_domain[domain] = {
            "rows": len(subset),
            "v1_accuracy": sum(row["v1_correct"] for row in subset) / len(subset),
            "v3_accuracy": sum(row["v3_correct"] for row in subset) / len(subset),
        }
        by_domain[domain]["delta"] = by_domain[domain]["v3_accuracy"] - by_domain[domain]["v1_accuracy"]
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in paired}):
        subset = [row for row in paired if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "rows": len(subset),
            "v1_accuracy": sum(row["v1_correct"] for row in subset) / len(subset),
            "v3_accuracy": sum(row["v3_correct"] for row in subset) / len(subset),
        }
        by_dataset[dataset]["delta"] = by_dataset[dataset]["v3_accuracy"] - by_dataset[dataset]["v1_accuracy"]

    comparison = {
        "schema_version": "phase3_router_v3_frozen_comparison_v1",
        "conclusion": "revise_recall_improved_but_keep_collapsed",
        "goal_met": False,
        "same_frozen_sources": True,
        "same_prompts": True,
        "frozen_rows": len(paired),
        "v1_metrics": {
            "decision_accuracy": old["decision_accuracy"],
            "balanced_accuracy": old["balanced_decision_accuracy"],
            "keep_recall": old["keep_recall"],
            "revise_recall": old["revise_recall"],
            "exact_contract": old["exact_decision_contract_rate"],
        },
        "v3_metrics": {
            "decision_accuracy": new["decision_accuracy"],
            "balanced_accuracy": new["balanced_accuracy"],
            "keep_recall": new["keep_recall"],
            "revise_recall": new["revise_recall"],
            "exact_contract": new["exact_contract_accuracy"],
        },
        "deltas_v3_minus_v1": {
            "decision_accuracy": new["decision_accuracy"] - old["decision_accuracy"],
            "balanced_accuracy": new["balanced_accuracy"] - old["balanced_decision_accuracy"],
            "keep_recall": new["keep_recall"] - old["keep_recall"],
            "revise_recall": new["revise_recall"] - old["revise_recall"],
        },
        "paired_correctness": {
            "v1_only_correct": v1_only,
            "v3_only_correct": v3_only,
            "both_correct": sum(row["v1_correct"] and row["v3_correct"] for row in paired),
            "both_wrong": sum(not row["v1_correct"] and not row["v3_correct"] for row in paired),
            "exact_mcnemar_p": exact_mcnemar(v1_only, v3_only),
        },
        "prediction_transitions_by_actual_label": transitions,
        "by_domain": by_domain,
        "by_dataset": by_dataset,
        "rationale_capture": new["rationale_capture"],
        "training": {
            "checkpoint": train_report["model"]["training_checkpoint"],
            "initial_adapter": train_report["model"]["initial_adapter"],
            "epochs": train_report["training"]["epochs"],
            "learning_rate": train_report["training"]["learning_rate"],
            "train_rows": train_report["data"]["train_rows"],
            "dev_rows": train_report["data"]["dev_rows"],
            "eval_loss_before": train_report["eval_before"]["eval_before_loss"],
            "eval_loss_after": train_report["eval_after"]["eval_after_loss"],
            "runtime_seconds": train_report["train_metrics"]["train_runtime"],
        },
        "dataset": {
            "total_rows": dataset_summary["total_rows"],
            "keep": dataset_summary["label_counts"]["KEEP"],
            "revise": dataset_summary["label_counts"]["REVISE"],
            "hard_revise": dataset_summary["hard_revise_count"],
        },
        "provenance": {
            "v1_rows_sha256": hashlib.sha256(v1_path.read_bytes()).hexdigest(),
            "v3_rows_sha256": hashlib.sha256(v3_path.read_bytes()).hexdigest(),
            "frozen_manifest_sha256": v1_summary["manifest"]["sha256"],
            "v3_split_sha256": v3_summary["split"]["sha256"],
        },
    }

    output_dir = Path(args.output_dir).resolve()
    write_json(output_dir / "router_v3_comparison.json", comparison)
    write_json(output_dir / "router_v3_paired_rows.json", paired)

    lines = [
        "# Router V3 Mini-Training: Frozen Benchmark Comparison",
        "",
        "## Conclusion",
        "",
        "`revise_recall_improved_but_keep_collapsed`",
        "",
        "Router V3 learned the intended stronger REVISE tendency, but it did not meet",
        "the full objective because the gain came with a larger loss in KEEP recall.",
        "",
        "## Frozen benchmark (same 200 sources and prompts)",
        "",
        "| Metric | Decision-Only V1 | Router V3 | Delta |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = (
        ("Exact contract", "exact_contract"),
        ("Decision accuracy", "decision_accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("KEEP recall", "keep_recall"),
        ("REVISE recall", "revise_recall"),
    )
    for label, key in metric_rows:
        a = comparison["v1_metrics"][key]
        b = comparison["v3_metrics"][key]
        lines.append(f"| {label} | {pct(a)} | {pct(b)} | {100 * (b-a):+.1f} pp |")
    lines += [
        "",
        "## Paired outcome",
        "",
        f"- V3-only correct: {v3_only}.",
        f"- V1-only correct: {v1_only}.",
        f"- Exact McNemar p-value: {comparison['paired_correctness']['exact_mcnemar_p']:.6f}.",
        f"- Initially wrong: V3 changed {transitions['REVISE'].get('KEEP->REVISE', 0)} V1 misses into REVISE, but changed {transitions['REVISE'].get('REVISE->KEEP', 0)} correct V1 decisions back to KEEP.",
        f"- Initially correct: V3 introduced {transitions['KEEP'].get('KEEP->REVISE', 0)} new false revisions and recovered {transitions['KEEP'].get('REVISE->KEEP', 0)} old false revisions.",
        "",
        "## Training diagnostic",
        "",
        f"- Continued the existing Decision-Only LoRA for 1 epoch at LR `{train_report['training']['learning_rate']}`.",
        f"- Dataset: 200 train / 50 dev; 100 KEEP / 150 REVISE overall; {dataset_summary['hard_revise_count']} hard-REVISE.",
        f"- Dev loss: {train_report['eval_before']['eval_before_loss']:.6f} before -> {train_report['eval_after']['eval_after_loss']:.6f} after.",
        f"- Training runtime: {train_report['train_metrics']['train_runtime']:.1f}s on {train_report['runtime']['gpu']}.",
        f"- Visible rationale captured separately for {new['rationale_capture']['rows_with_substantive_visible_reasoning']}/200 V3 samples.",
        "",
        "## Interpretation",
        "",
        "The 60% REVISE prior moved the router in the intended direction, but one epoch",
        "overshifted its operating point. The result is a recall trade rather than a net",
        "improvement: +17 pp REVISE recall, -19 pp KEEP recall, and -1 pp balanced accuracy.",
        "Do not promote Router V3 as the replacement router from this run.",
        "",
    ]
    (output_dir / "router_v3_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
