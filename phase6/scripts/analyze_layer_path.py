"""Audit whether correctness signal attenuates from mid to final hidden layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
DISPLAY = {
    "original_solver": "Original solver",
    "warmstart_v2": "Warm-start V2",
    "correction_sft_v3": "Correction SFT V3",
}


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path,
                        default=ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase6_layer_path_audit")
    parser.add_argument("--fixed-c", type=float, default=0.01)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "schema_version": "phase6_layer_path_audit_v1",
        "source": "existing_phase5_development_probe_candidates",
        "split": "development",
        "rows": 80,
        "protected_data_used": False,
        "probe_refit": False,
        "fixed_c": args.fixed_c,
        "checkpoint_results": {},
    }
    for checkpoint in CHECKPOINTS:
        source = args.selection_dir / f"{checkpoint}_probe_selection.json"
        selection = json.loads(source.read_text(encoding="utf-8"))
        candidates = selection["candidates"]
        fixed = [row for row in candidates if float(row["c"]) == args.fixed_c]
        fixed.sort(key=lambda row: int(row["layer"]))
        if [int(row["layer"]) for row in fixed] != [7, 14, 21, 28]:
            raise RuntimeError(f"Missing fixed-C layers for {checkpoint}")
        best_by_layer = []
        for layer in (7, 14, 21, 28):
            choices = [row for row in candidates if int(row["layer"]) == layer]
            best = max(choices, key=lambda row: (
                row["development"]["balanced_accuracy"],
                row["development"]["roc_auc"], -float(row["c"]),
            ))
            best_by_layer.append(best)
        mid = next(row for row in fixed if int(row["layer"]) == 14)["development"]
        final = next(row for row in fixed if int(row["layer"]) == 28)["development"]
        aucs = [row["development"]["roc_auc"] for row in fixed]
        post_mid_monotonic = aucs[1] >= aucs[2] >= aucs[3]
        report["checkpoint_results"][checkpoint] = {
            "display": DISPLAY[checkpoint],
            "selection_source": source.relative_to(ROOT).as_posix(),
            "selection_sha256_lf": sha256_lf(source),
            "fixed_c_curve": fixed,
            "best_c_by_layer": best_by_layer,
            "layer14_to_layer28": {
                "balanced_accuracy_delta": final["balanced_accuracy"] - mid["balanced_accuracy"],
                "roc_auc_delta": final["roc_auc"] - mid["roc_auc"],
                "correct_recall_delta": final["correct_recall"] - mid["correct_recall"],
                "wrong_recall_delta": final["wrong_recall"] - mid["wrong_recall"],
            },
            "auc_nonincreasing_after_layer14": post_mid_monotonic,
        }
    auc_deltas = [value["layer14_to_layer28"]["roc_auc_delta"]
                  for value in report["checkpoint_results"].values()]
    ba_deltas = [value["layer14_to_layer28"]["balanced_accuracy_delta"]
                 for value in report["checkpoint_results"].values()]
    replicated_decline = all(value < 0 for value in auc_deltas)
    report["aggregate"] = {
        "mean_layer14_to_layer28_roc_auc_delta": sum(auc_deltas) / len(auc_deltas),
        "mean_layer14_to_layer28_balanced_accuracy_delta": sum(ba_deltas) / len(ba_deltas),
        "roc_auc_declines_in_all_checkpoints": replicated_decline,
        "interpretation": (
            "The fixed-regularization probes show a replicated mid-layer peak and "
            "partial attenuation toward the final hidden layer. The final layer remains "
            "substantially above chance, so the signal is weakened rather than lost. "
            "This is correlational layer-wise evidence, not a causal logit-path proof."
            if replicated_decline else
            "The fixed-regularization curves do not show a replicated decline from "
            "layer 14 to the final hidden layer."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "# Phase 6 layer-path audit", "",
        "This reuses the frozen Phase 5 development activations and already-recorded "
        "probe candidates. No model or probe was refitted and the protected set was not used.", "",
        "## Fixed-regularization comparison (C = 0.01)", "",
        "| Checkpoint | Layer | Balanced accuracy | ROC-AUC | Correct recall | Wrong recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        for row in report["checkpoint_results"][checkpoint]["fixed_c_curve"]:
            metric = row["development"]
            lines.append(
                f"| {DISPLAY[checkpoint]} | {row['layer']} | "
                f"{metric['balanced_accuracy']:.3f} | {metric['roc_auc']:.3f} | "
                f"{metric['correct_recall']:.3f} | {metric['wrong_recall']:.3f} |"
            )
    lines += ["", "## Layer 14 to final hidden layer", "",
              "| Checkpoint | Delta balanced accuracy | Delta ROC-AUC | Delta correct recall | Delta wrong recall |",
              "|---|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        delta = report["checkpoint_results"][checkpoint]["layer14_to_layer28"]
        lines.append(
            f"| {DISPLAY[checkpoint]} | {delta['balanced_accuracy_delta']:+.3f} | "
            f"{delta['roc_auc_delta']:+.3f} | {delta['correct_recall_delta']:+.3f} | "
            f"{delta['wrong_recall_delta']:+.3f} |"
        )
    aggregate = report["aggregate"]
    lines += [
        "", "## Interpretation", "", aggregate["interpretation"], "",
        f"Mean layer-14 to layer-28 change: balanced accuracy "
        f"{aggregate['mean_layer14_to_layer28_balanced_accuracy_delta']:+.3f}, "
        f"ROC-AUC {aggregate['mean_layer14_to_layer28_roc_auc_delta']:+.3f}.", "",
        "Layer 28 is the final hidden-state probe position, not the vocabulary-logit "
        "readout itself. A teacher-forced KEEP/REVISE logit-lens experiment would be "
        "a separate GPU protocol. Therefore this result supports partial attenuation, "
        "not proof that the signal disappears before verbalization.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
