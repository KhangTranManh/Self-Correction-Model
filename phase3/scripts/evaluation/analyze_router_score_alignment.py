"""Compare frozen hidden-state scores with native Decision-Only V1 decisions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--native-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    scores = {row["source_id"]: row for row in read_jsonl(Path(args.scores))}
    native = read_jsonl(Path(args.native_results))
    if set(scores) != {row["source_id"] for row in native}:
        raise RuntimeError("Score/native source sets do not match")
    ordered = [(row, scores[row["source_id"]]) for row in native]
    y = np.asarray([row[0]["expected_decision"] == "REVISE" for row in ordered], dtype=int)
    native_pred = np.asarray([row[0]["decision"] == "REVISE" for row in ordered], dtype=int)
    score = np.asarray([float(row[1]["revise_probability"]) for row in ordered])
    median = float(np.median(score))
    rank_pred = (score >= median).astype(int)

    order = np.argsort(score)
    quartile = np.empty(len(score), dtype=int)
    quartile[order] = np.arange(len(score)) * 4 // len(score)
    quartiles = []
    for value in range(4):
        mask = quartile == value
        quartiles.append({
            "quartile": value + 1,
            "rows": int(mask.sum()),
            "score_min": float(score[mask].min()),
            "score_max": float(score[mask].max()),
            "expected_revise_rate": float(y[mask].mean()),
            "native_revise_rate": float(native_pred[mask].mean()),
        })
    disagreements = rank_pred != native_pred
    rank_better = disagreements & (rank_pred == y) & (native_pred != y)
    native_better = disagreements & (native_pred == y) & (rank_pred != y)
    report = {
        "schema_version": "phase3_router_score_alignment_v1",
        "rows": len(ordered),
        "score_auc_for_expected_revise": float(roc_auc_score(y, score)),
        "score_auc_for_native_revise": float(roc_auc_score(native_pred, score)),
        "native": {
            "balanced_accuracy": float(balanced_accuracy_score(y, native_pred)),
            "distribution": dict(Counter("REVISE" if x else "KEEP" for x in native_pred)),
        },
        "rank_balanced": {
            "threshold": median,
            "balanced_accuracy": float(balanced_accuracy_score(y, rank_pred)),
            "distribution": dict(Counter("REVISE" if x else "KEEP" for x in rank_pred)),
        },
        "native_rank_agreement": float(np.mean(native_pred == rank_pred)),
        "disagreement_rows": int(disagreements.sum()),
        "rank_correct_native_wrong": int(rank_better.sum()),
        "native_correct_rank_wrong": int(native_better.sum()),
        "net_correct_gain_from_rank": int(rank_better.sum() - native_better.sum()),
        "score_quartiles": quartiles,
        "interpretation_rule": (
            "High expected-label AUC with weaker native-decision AUC indicates that correctness is "
            "decodable but incompletely used by the native decision policy."
        ),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "alignment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Decision-Only V1 score/policy alignment", "",
        f"- Correctness score AUC: {report['score_auc_for_expected_revise']:.3f}",
        f"- Score AUC for predicting native REVISE: {report['score_auc_for_native_revise']:.3f}",
        f"- Native balanced accuracy: {report['native']['balanced_accuracy']:.3f}",
        f"- Rank-balanced diagnostic accuracy: {report['rank_balanced']['balanced_accuracy']:.3f}",
        f"- Native/rank agreement: {report['native_rank_agreement']:.3f}",
        f"- Net corrected rows from rank decision: {report['net_correct_gain_from_rank']:+d}", "",
        "The rank threshold assumes the known 50/50 frozen benchmark prevalence and is a diagnostic, not a deployment calibration.", "",
    ]
    (output / "alignment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
