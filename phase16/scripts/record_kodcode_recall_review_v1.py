"""Record the manual taxonomy review of the frozen KodCode recall sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_SAMPLE_SHA256 = "d1a97333184f50b831703d93a019e3d251965b3e4f0c4c57287eb3f24fee632c"
DIRECT = {1, 12, 25}
ADAPTER = {26}
NEAR_NEGATIVE = {27, 90}
UNRELATED = {
    0, 5, 6, 7, 10, 11, 13, 14, 15, 16, 19, 20, 22, 30, 32, 34, 35,
    36, 37, 39, 40, 41, 42, 49, 50, 51, 52, 53, 56, 57, 58, 61, 62,
    63, 65, 68, 70, 72, 74, 76, 77, 80, 81, 82, 83, 93, 94, 95, 97,
}
DEFERRED_MULTIOP = {
    2, 4, 9, 18, 21, 23, 24, 28, 29, 31, 33, 43, 44, 46, 48, 54, 55,
    59, 60, 66, 69, 71, 73, 75, 79, 84, 85, 86, 88, 89, 91, 92, 99,
}
PLAIN_REJECT = {3, 8, 17, 38, 45, 47, 64, 67, 78, 87, 96, 98}

POSITIVE_DETAILS = {
    1: ("select", "implicit_integer_range", None,
        "explicit function selects integers in the inclusive start-to-end range under a stated predicate"),
    12: ("sum", "explicit_collection", None,
         "explicit function sums BST node values whose keys lie in the inclusive L-to-R range"),
    25: ("select", "explicit_collection", None,
         "explicit function selects log messages by level and inclusive timestamp bounds"),
    26: ("select", "explicit_collection", "stdio_to_function",
         "single standard-input operation selects timestamped messages within inclusive bounds"),
}
NEAR_DETAILS = {
    27: ("count", "explicit_collection",
         "explicitly half-open timestamp interval [S,E) is a structurally matched near-negative"),
    90: ("select", "implicit_integer_range",
         "explicitly half-open integer range [start,end) is a structurally matched near-negative"),
}
PLAIN_REASONS = {
    3: "source asks for complexity feedback rather than one implementable target contract",
    8: "longest-simple-path contract conflicts with the supplied two-DFS hint on cyclic graphs",
    17: "treasure collection semantics do not define whether revisits or repeated collection are allowed",
    38: "discussion/refactoring prompt does not define one checkable implementation target",
    45: "the claimed maximum after a fixed list of reversals is semantically ambiguous",
    47: "network and filesystem side effects prevent admission to the locked callable-task pool",
    64: "largest subgrid containing the maximum number of artefacts has no defined tie objective",
    67: "grid values are not defined as passable or blocked although examples treat zero as blocked",
    78: "per-element boolean classification is outside the locked count, sum, and select outputs",
    87: "the second worked range-sum example contradicts its own Fibonacci sequence",
    96: "the third example returns values outside the stated inclusive input range",
    98: "source requests prose comparing existing approaches, not one implementation target",
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def build(sample_raw: bytes) -> dict:
    digest = hashlib.sha256(sample_raw).hexdigest()
    if digest != EXPECTED_SAMPLE_SHA256:
        raise ValueError(f"frozen sample hash mismatch: {digest}")
    sample_doc = json.loads(sample_raw)
    sample = sample_doc["sample"]
    partitions = [DIRECT, ADAPTER, NEAR_NEGATIVE, UNRELATED, DEFERRED_MULTIOP, PLAIN_REJECT]
    all_indices = [index for group in partitions for index in group]
    if len(sample) != 100 or sorted(all_indices) != list(range(100)) or len(set(all_indices)) != 100:
        raise ValueError("review partition does not cover the frozen sample exactly once")

    rows = []
    for index, family in enumerate(sample):
        control_subtype = adapter_kind = deferred_status = None
        variant = "none"
        universe_mode = "not_applicable"
        if index in DIRECT or index in ADAPTER:
            lane = "direct_candidate" if index in DIRECT else "execution_adapter_candidate"
            variant, universe_mode, adapter_kind, reason = POSITIVE_DETAILS[index]
        elif index in NEAR_NEGATIVE:
            lane = "control_candidate"
            control_subtype = "near_negative"
            variant, universe_mode, reason = NEAR_DETAILS[index]
        elif index in UNRELATED:
            lane = "control_candidate"
            control_subtype = "unrelated"
            reason = "execution-ready task exercises a skill unrelated to filtering keys within one closed L-to-R interval"
        elif index in DEFERRED_MULTIOP:
            lane = "hard_reject"
            deferred_status = "deferred_multiop"
            reason = "multi-query or mutable-state operation stream is outside the locked single-operation contract"
        else:
            lane = "hard_reject"
            reason = PLAIN_REASONS[index]
        rows.append({
            "review_index": index,
            "family_id": family["family_id"],
            "subset": family["subset"],
            "representative_id": family["representative_id"],
            "question_sha256": family["representative_question_sha256"],
            "review_lane": lane,
            "control_subtype": control_subtype,
            "adapter_kind": adapter_kind,
            "deferred_status": deferred_status,
            "variant": variant,
            "universe_mode": universe_mode,
            "reason": reason,
            "oracle_status": "pending",
            "history_status": "pending",
            "near_duplicate_status": "pending",
            "sandbox_status": "pending",
        })

    lane_names = ("direct_candidate", "execution_adapter_candidate", "control_candidate", "hard_reject")
    counts = {name: sum(row["review_lane"] == name for row in rows) for name in lane_names}
    false_negatives = counts["direct_candidate"] + counts["execution_adapter_candidate"]
    positive_precision = 26 / 100
    negative_fn_rate = false_negatives / len(rows)
    p_interval = wilson(26, 100)
    fn_interval = wilson(false_negatives, len(rows))
    n_positive = 627
    n_negative = sample_doc["predicted_negative_components"]

    def recall(p: float, fn_rate: float) -> float:
        estimated_tp = p * n_positive
        return estimated_tp / (estimated_tp + fn_rate * n_negative)

    return {
        "schema_version": "p16_kodcode_recall_review_v1",
        "taxonomy": "phase16/docs/FAMILY_REVIEW_TAXONOMY_V1.md",
        "sample_sha256": digest,
        "counts": {
            "reviewed": len(rows),
            **counts,
            "false_negatives": false_negatives,
            "true_negatives": len(rows) - false_negatives,
            "near_negative_controls": sum(row["control_subtype"] == "near_negative" for row in rows),
            "unrelated_controls": sum(row["control_subtype"] == "unrelated" for row in rows),
            "deferred_multiop": sum(row["deferred_status"] == "deferred_multiop" for row in rows),
            "fully_certified": 0,
        },
        "estimates": {
            "positive_sample_precision": positive_precision,
            "positive_sample_precision_wilson95": list(p_interval),
            "negative_sample_false_negative_rate": negative_fn_rate,
            "negative_sample_false_negative_rate_wilson95": list(fn_interval),
            "negative_sample_true_negative_rate": 1 - negative_fn_rate,
            "positive_family_population": n_positive,
            "negative_component_population": n_negative,
            "plugin_recall": recall(positive_precision, negative_fn_rate),
            "sensitivity_recall_low": recall(p_interval[0], fn_interval[1]),
            "sensitivity_recall_high": recall(p_interval[1], fn_interval[0]),
        },
        "rows": rows,
        "limitations": (
            "Manual lane audit and two-stratum estimate only; positive strata use 627 shortlist families "
            "while the broad graph merges them into 625 components. No task is certified and GPU remains closed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = build(args.sample.read_bytes())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "estimates": result["estimates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
