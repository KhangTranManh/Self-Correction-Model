"""Build the final Phase 5 behavioral/probe report from backed-up raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
CONDITIONS = ("neutral", "status")
DISPLAY = {
    "original_solver": "Original solver",
    "warmstart_v2": "Warm-start V2",
    "correction_sft_v3": "Correction SFT V3",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def bootstrap_delta(a: np.ndarray, b: np.ndarray, namespace: str,
                    repetitions: int = 10000) -> dict[str, float]:
    if a.shape != b.shape or not len(a):
        raise RuntimeError(f"Invalid paired arrays: {namespace}")
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        count = min(500, repetitions - start)
        indices = rng.integers(0, len(a), size=(count, len(a)))
        values[start:start + count] = (a[indices] - b[indices]).mean(axis=1)
    point = float(a.mean() - b.mean())
    return {
        "point": point,
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "outputs/phase5_gpu_vllm")
    parser.add_argument("--output-json", type=Path,
                        default=ROOT / "outputs/phase5_gpu_vllm/phase5_final_report.json")
    parser.add_argument("--output-md", type=Path,
                        default=ROOT / "outputs/phase5_gpu_vllm/phase5_final_report.md")
    parser.add_argument("--receipt", type=Path,
                        default=ROOT / "phase5/data/protocol/protected_opening_v1_receipt.json")
    args = parser.parse_args()

    review_dir = args.root / "protected_v1/reviews"
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    behavioral: dict[str, Any] = {}
    source_maps: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    evidence: dict[str, str] = {}
    for checkpoint in CHECKPOINTS:
        path = review_dir / f"{checkpoint}_protected_test.jsonl"
        rows = read_jsonl(path)
        evidence[path.relative_to(args.root).as_posix()] = sha256(path)
        records[checkpoint] = {condition: [r for r in rows if r["condition"] == condition]
                               for condition in CONDITIONS}
        source_maps[checkpoint] = {
            condition: {r["problem_id"]: r for r in records[checkpoint][condition]}
            for condition in CONDITIONS
        }
        behavioral[checkpoint] = {}
        for condition in CONDITIONS:
            subset = records[checkpoint][condition]
            correct = [r for r in subset if r["initial_correct"]]
            wrong = [r for r in subset if not r["initial_correct"]]
            if len(subset) != 160 or len(correct) != 80 or len(wrong) != 80:
                raise RuntimeError(f"Protected composition mismatch: {checkpoint}/{condition}")
            behavioral[checkpoint][condition] = {
                "rows": len(subset),
                "strict_contract_validity": sum(r["strict_contract_valid"] for r in subset) / len(subset),
                "final_accuracy_conservative": sum(r["final_correct"] is True for r in subset) / len(subset),
                "wrong_to_correct_rate": sum(r["final_correct"] is True for r in wrong) / len(wrong),
                "correct_to_wrong_rate_verified": sum(r["final_correct"] is False for r in correct) / len(correct),
                "correct_invalid_rate": sum(r["final_correct"] is None for r in correct) / len(correct),
                "keep_rate_all": sum(r["decision"] == "KEEP" for r in subset) / len(subset),
                "revise_rate_all": sum(r["decision"] == "REVISE" for r in subset) / len(subset),
            }

    within_checkpoint = {}
    assisted_gate = {}
    for checkpoint in CHECKPOINTS:
        neutral = source_maps[checkpoint]["neutral"]
        status = source_maps[checkpoint]["status"]
        ids = sorted(neutral)
        wrong_ids = [i for i in ids if not neutral[i]["initial_correct"]]
        correct_ids = [i for i in ids if neutral[i]["initial_correct"]]
        delta = {
            "final_accuracy_status_minus_neutral": bootstrap_delta(
                np.asarray([status[i]["final_correct"] is True for i in ids], float),
                np.asarray([neutral[i]["final_correct"] is True for i in ids], float),
                checkpoint + ":final"),
            "wrong_to_correct_status_minus_neutral": bootstrap_delta(
                np.asarray([status[i]["final_correct"] is True for i in wrong_ids], float),
                np.asarray([neutral[i]["final_correct"] is True for i in wrong_ids], float),
                checkpoint + ":w2c"),
            "correct_to_wrong_status_minus_neutral": bootstrap_delta(
                np.asarray([status[i]["final_correct"] is False for i in correct_ids], float),
                np.asarray([neutral[i]["final_correct"] is False for i in correct_ids], float),
                checkpoint + ":c2w"),
        }
        within_checkpoint[checkpoint] = delta
        assisted_gate[checkpoint] = {
            "w2c_lower_ci_gt_zero": delta["wrong_to_correct_status_minus_neutral"]["ci95_lower"] > 0,
            "correct_to_wrong_increase_le_2_5pp": delta["correct_to_wrong_status_minus_neutral"]["point"] <= 0.025,
            "status_contract_validity_ge_95pct": behavioral[checkpoint]["status"]["strict_contract_validity"] >= 0.95,
        }
        assisted_gate[checkpoint]["all_components_pass"] = all(assisted_gate[checkpoint].values())

    checkpoint_deltas: dict[str, Any] = {}
    for condition in CONDITIONS:
        checkpoint_deltas[condition] = {}
        for left, right in (("warmstart_v2", "original_solver"),
                            ("correction_sft_v3", "original_solver"),
                            ("correction_sft_v3", "warmstart_v2")):
            lmap, rmap = source_maps[left][condition], source_maps[right][condition]
            ids = sorted(lmap)
            wrong_ids = [i for i in ids if not lmap[i]["initial_correct"]]
            correct_ids = [i for i in ids if lmap[i]["initial_correct"]]
            key = f"{left}_minus_{right}"
            checkpoint_deltas[condition][key] = {
                "final_accuracy": bootstrap_delta(
                    np.asarray([lmap[i]["final_correct"] is True for i in ids], float),
                    np.asarray([rmap[i]["final_correct"] is True for i in ids], float),
                    condition + ":" + key + ":final"),
                "wrong_to_correct": bootstrap_delta(
                    np.asarray([lmap[i]["final_correct"] is True for i in wrong_ids], float),
                    np.asarray([rmap[i]["final_correct"] is True for i in wrong_ids], float),
                    condition + ":" + key + ":w2c"),
                "correct_to_wrong": bootstrap_delta(
                    np.asarray([lmap[i]["final_correct"] is False for i in correct_ids], float),
                    np.asarray([rmap[i]["final_correct"] is False for i in correct_ids], float),
                    condition + ":" + key + ":c2w"),
            }

    probe_path = args.root / "protected_v1/protected_probe_report.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    evidence[probe_path.relative_to(args.root).as_posix()] = sha256(probe_path)
    controls_path = args.root / "probe_v1/probe_controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    evidence[controls_path.relative_to(args.root).as_posix()] = sha256(controls_path)

    report = {
        "schema_version": "phase5_final_report_v1",
        "status": "protected_evaluation_complete",
        "backend": {"reviews": "vllm_0_7_0_fp16", "probe": "transformers_fp16_forward_only"},
        "protected_open_count": 1,
        "behavioral_protected": behavioral,
        "paired_status_minus_neutral_bootstrap": within_checkpoint,
        "assisted_repair_gate": assisted_gate,
        "paired_checkpoint_bootstrap": checkpoint_deltas,
        "probe_development_controls": controls,
        "probe_protected": probe,
        "evidence_sha256": evidence,
        "interpretation": {
            "autonomous_neutral": "No checkpoint shows a safe autonomous repair gain; V3 fixes some wrong answers but causes large correct-to-wrong harm.",
            "guided_status": "V2 and V3 improve assisted repair without observed status-condition harms; status discloses correctness and is not autonomous detection.",
            "representation": "Correctness is linearly decodable before the hint in all checkpoints. V3 shifts protected recall toward wrong answers, but does not convert that signal into a safe neutral policy.",
            "promotion": "none",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 5 final report",
        "",
        "Protected evaluation was opened once after the review protocol, probe layer/C choices, and controls were locked. Reviews used vLLM 0.7.0 FP16; probes used frozen Transformers FP16 forward passes.",
        "",
        "## Protected behavioral results",
        "",
        "| Checkpoint | Condition | Contract | Final accuracy | Wrong→correct | Correct→wrong | Correct invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        for condition in CONDITIONS:
            row = behavioral[checkpoint][condition]
            lines.append(
                f"| {DISPLAY[checkpoint]} | {condition} | {pct(row['strict_contract_validity'])} | "
                f"{pct(row['final_accuracy_conservative'])} | {pct(row['wrong_to_correct_rate'])} | "
                f"{pct(row['correct_to_wrong_rate_verified'])} | {pct(row['correct_invalid_rate'])} |"
            )
    lines += ["", "The initial accuracy is 50% by construction. Invalid contracts count as not correct in final accuracy, but are reported separately rather than mislabeled as verified harms.", "",
              "## Status-minus-neutral paired bootstrap", "",
              "| Checkpoint | Δ wrong→correct (95% CI) | Δ correct→wrong (95% CI) | Assisted gate |",
              "|---|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        w = within_checkpoint[checkpoint]["wrong_to_correct_status_minus_neutral"]
        h = within_checkpoint[checkpoint]["correct_to_wrong_status_minus_neutral"]
        lines.append(
            f"| {DISPLAY[checkpoint]} | {pct(w['point'])} [{pct(w['ci95_lower'])}, {pct(w['ci95_upper'])}] | "
            f"{pct(h['point'])} [{pct(h['ci95_lower'])}, {pct(h['ci95_upper'])}] | "
            f"{'PASS' if assisted_gate[checkpoint]['all_components_pass'] else 'FAIL'} |"
        )
    lines += ["", "## Protected pre-hint probes", "",
              "| Checkpoint | Layer / C | Balanced accuracy | ROC-AUC | Correct recall | Wrong recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        row = probe["results"][checkpoint]
        lines.append(
            f"| {DISPLAY[checkpoint]} | {row['selected_layer']} / {row['selected_c']} | "
            f"{pct(row['balanced_accuracy'])} | {pct(row['roc_auc'])} | "
            f"{pct(row['correct_recall'])} | {pct(row['wrong_recall'])} |"
        )
    surface = controls["surface_control"]["selected"]["development"]
    lines += [
        "", "Surface-only development BA was " + pct(surface["balanced_accuracy"]) +
        "; 100 shuffled-label fits stayed at chance on average and none matched the selected hidden probes (empirical p=0.0099 for each checkpoint).",
        "", "## Conclusion", "",
        "The model family contains a real, generalizing correctness signal before feedback, but the generative review policy does not use it safely. V3 increases error sensitivity while catastrophically over-revising correct answers under neutral review. Status feedback enables limited assisted repair, especially for V2/V3, but does not demonstrate autonomous error detection. No checkpoint is promoted.",
        "", "The linear probe is an external diagnostic harness over frozen hidden states, not part of the standalone model's generative path. It establishes signal availability, not autonomous signal use. A future model-plus-probe repair pipeline must be reported as harness-controlled unless model-only detect-preserve-repair behavior is demonstrated on a new protected set.",
        "", "Raw prompts, raw model outputs (including all generated reasoning/text), parsed decisions, verifier outcomes, activations, probe models, and hashes are retained under `outputs/phase5_gpu_vllm/`.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt = {
        "schema_version": "phase5_protected_opening_receipt_v1",
        "status": "complete",
        "protected_open_count": 1,
        "selection_changed_after_opening": False,
        "final_report": {"path": args.output_json.relative_to(ROOT).as_posix(),
                         "sha256": sha256(args.output_json)},
        "protected_evidence_sha256": evidence,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output_json), "markdown": str(args.output_md),
                      "receipt": str(args.receipt)}, indent=2))


if __name__ == "__main__":
    main()
