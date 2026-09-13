"""Compare any candidate router with canonical Decision-Only V1 on 200 frozen rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rows", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-rows", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-floor", type=float, default=0.70)
    args = parser.parse_args()
    baseline_path = Path(args.baseline_rows).resolve()
    candidate_path = Path(args.candidate_rows).resolve()
    old_rows = {row["source_id"]: row for row in read_jsonl(baseline_path)}
    new_rows = {row["source_id"]: row for row in read_jsonl(candidate_path)}
    if set(old_rows) != set(new_rows) or len(old_rows) != 200:
        raise RuntimeError("Candidate and baseline must contain the same 200 source IDs")
    old_summary = read_json(Path(args.baseline_summary).resolve())["metrics"]
    new_summary = read_json(Path(args.candidate_summary).resolve())["metrics"]
    old_metrics = {
        "balanced_accuracy": old_summary["balanced_decision_accuracy"],
        "keep_recall": old_summary["keep_recall"],
        "revise_recall": old_summary["revise_recall"],
        "decision_accuracy": old_summary["decision_accuracy"],
        "exact_contract": old_summary["exact_decision_contract_rate"],
    }
    new_metrics = {
        "balanced_accuracy": new_summary["balanced_accuracy"],
        "keep_recall": new_summary["keep_recall"],
        "revise_recall": new_summary["revise_recall"],
        "decision_accuracy": new_summary["decision_accuracy"],
        "exact_contract": new_summary["exact_contract_accuracy"],
    }
    paired = []
    for source_id in sorted(old_rows):
        old = old_rows[source_id]
        new = new_rows[source_id]
        expected = old["expected_decision"]
        if expected != new["expected_decision"]:
            raise RuntimeError(f"Label mismatch: {source_id}")
        old_correct = old["decision"] == expected
        new_correct = new["predicted_decision"] == expected
        paired.append({
            "source_id": source_id,
            "dataset": old["dataset"],
            "domain": old["domain"],
            "expected_decision": expected,
            "baseline_decision": old["decision"],
            "candidate_decision": new["predicted_decision"],
            "baseline_correct": old_correct,
            "candidate_correct": new_correct,
        })
    old_only = sum(row["baseline_correct"] and not row["candidate_correct"] for row in paired)
    new_only = sum(row["candidate_correct"] and not row["baseline_correct"] for row in paired)
    gate = (
        new_metrics["balanced_accuracy"] > old_metrics["balanced_accuracy"]
        and new_metrics["revise_recall"] > old_metrics["revise_recall"]
        and new_metrics["keep_recall"] >= args.keep_floor
    )
    slices: dict[str, Any] = {}
    for field in ("domain", "dataset"):
        slices[field] = {}
        for value in sorted({row[field] for row in paired}):
            subset = [row for row in paired if row[field] == value]
            slices[field][value] = {
                "rows": len(subset),
                "baseline_accuracy": sum(row["baseline_correct"] for row in subset) / len(subset),
                "candidate_accuracy": sum(row["candidate_correct"] for row in subset) / len(subset),
            }
    result = {
        "schema_version": "phase3_generic_frozen_router_comparison_v1",
        "candidate": args.candidate_name,
        "rows": 200,
        "same_source_ids": True,
        "baseline": old_metrics,
        "candidate_metrics": new_metrics,
        "delta": {key: new_metrics[key] - old_metrics[key] for key in old_metrics},
        "promotion_gate": {"passed": gate, "keep_recall_floor": args.keep_floor},
        "paired": {"baseline_only_correct": old_only, "candidate_only_correct": new_only, "mcnemar_exact_p": exact_mcnemar(old_only, new_only)},
        "slices": slices,
        "transition_counts": dict(sorted(Counter(f"{old_rows[r['source_id']]['decision']}->{new_rows[r['source_id']]['predicted_decision']}" for r in paired).items())),
        "hashes": {"baseline_rows": hashlib.sha256(baseline_path.read_bytes()).hexdigest(), "candidate_rows": hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {args.candidate_name}: frozen router comparison", "",
        f"Status: **{'PASS' if gate else 'FAIL'}**", "",
        "| Metric | Decision-Only V1 | Candidate | Delta |", "|---|---:|---:|---:|",
    ]
    for label, key in (("Balanced accuracy", "balanced_accuracy"), ("KEEP recall", "keep_recall"), ("REVISE recall", "revise_recall"), ("Decision accuracy", "decision_accuracy"), ("Exact contract", "exact_contract")):
        lines.append(f"| {label} | {pct(old_metrics[key])} | {pct(new_metrics[key])} | {100*(new_metrics[key]-old_metrics[key]):+.1f} pp |")
    lines += ["", f"Paired candidate-only/baseline-only correct: {new_only}/{old_only}; exact McNemar p={exact_mcnemar(old_only, new_only):.6f}.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (output / "paired_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
