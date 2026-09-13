"""Repeated source-disjoint validation of the preselected linear probe family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "lower_95_empirical": float(np.quantile(array, 0.025)),
        "upper_95_empirical": float(np.quantile(array, 0.975)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.dataset))
    archive = np.load(args.activations, allow_pickle=False)
    layers = [int(value) for value in archive["layers"]]
    if args.layer not in layers:
        raise RuntimeError(f"Layer {args.layer} not found")
    if [str(x) for x in archive["source_ids"]] != [str(row["source_id"]) for row in rows]:
        raise RuntimeError("Activation row order mismatch")
    ids = [str(row["source_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate source IDs violate source-disjoint validation")
    x = archive["activations"][:, layers.index(args.layer), :].astype(np.float32)
    y = np.asarray([int(row["class_id"]) for row in rows])
    splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)

    fold_rows = []
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=args.c, solver="liblinear", max_iter=5000, random_state=args.seed + fold),
        )
        model.fit(x[train], y[train])
        score = model.predict_proba(x[test])[:, 1]
        pred = (score >= args.threshold).astype(int)
        fold_rows.append({
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "source_overlap": len(set(ids[i] for i in train) & set(ids[i] for i in test)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "keep_recall": float(recall_score(y[test], pred, pos_label=0, zero_division=0)),
            "revise_recall": float(recall_score(y[test], pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[test], score)),
        })
    report = {
        "schema_version": "phase3_probe_stability_v1",
        "rows": len(rows),
        "unique_sources": len(set(ids)),
        "folds": args.folds,
        "repeats": args.repeats,
        "evaluations": len(fold_rows),
        "preselected": {"layer": args.layer, "c": args.c, "threshold": args.threshold},
        "all_source_overlap_zero": all(row["source_overlap"] == 0 for row in fold_rows),
        "summary": {
            metric: interval([row[metric] for row in fold_rows])
            for metric in ("balanced_accuracy", "keep_recall", "revise_recall", "roc_auc")
        },
        "fold_results": fold_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "all_source_overlap_zero": report["all_source_overlap_zero"]}, indent=2))


if __name__ == "__main__":
    main()
