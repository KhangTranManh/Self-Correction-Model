"""Read-only, source-level audit of one CodeContests train shard and MATH history.

No dataset solution is executed. The first shard is not a random sample and
must not be used to estimate the full CodeContests population.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taco_inventory import normalized_text, sha256_file  # noqa: E402


def count_tests(value: dict | None) -> tuple[int, bool]:
    value = value or {}
    inputs, outputs = value.get("input") or [], value.get("output") or []
    return min(len(inputs), len(outputs)), len(inputs) == len(outputs)


def audit(workspace: Path, shard: Path, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    columns = ["description", "public_tests", "private_tests", "generated_tests",
               "source", "difficulty", "solutions", "incorrect_solutions",
               "cf_contest_id", "cf_index", "cf_tags", "input_file", "output_file"]
    rows = pq.read_table(shard, columns=columns).to_pylist()
    taco_hashes = set()
    with (workspace / "phase16/data/taco_inventory_v2/train_metadata.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            taco_hashes.add(json.loads(line)["normalized_question_sha256"])
    source_counts, difficulty_counts = Counter(), Counter()
    count = Counter()
    test_count = {kind: [] for kind in ("public", "private", "generated", "total")}
    for row in rows:
        # Keep raw numeric codes: the HF conversion need not preserve the
        # original proto enum mapping for every source.
        source_counts[str(row["source"])] += 1
        difficulty_counts[str(row["difficulty"])] += 1
        description = row.get("description") or ""
        digest = hashlib.sha256(normalized_text(description).encode()).hexdigest()
        taco_overlap = digest in taco_hashes
        if taco_overlap:
            count["exact_text_overlap_taco_train"] += 1
        if row.get("cf_contest_id"):
            count["has_cf_contest_id"] += 1
        if row.get("cf_tags"):
            count["has_cf_tags"] += 1
        if row.get("input_file") or row.get("output_file"):
            count["special_file_io"] += 1
        total = 0
        for kind in ("public", "private", "generated"):
            n, paired = count_tests(row.get(f"{kind}_tests"))
            test_count[kind].append(n)
            total += n
            if not paired:
                count[f"unpaired_{kind}_tests"] += 1
        test_count["total"].append(total)
        if total >= 5:
            count["at_least_5_raw_tests"] += 1
        python3 = {}
        for key in ("solutions", "incorrect_solutions"):
            value = row.get(key) or {}
            languages = value.get("language") or []
            solutions = value.get("solution") or []
            if solutions:
                count[f"has_{key}"] += 1
            python3[key] = 3 in languages  # ContestProblem.Solution.Language.PYTHON3
            if python3[key]:
                count[f"has_python3_{key}"] += 1
            if len(languages) != len(solutions):
                count[f"unpaired_{key}"] += 1
        if python3["solutions"] and python3["incorrect_solutions"]:
            count["has_both_python3_solution_classes"] += 1
            if total >= 5 and not taco_overlap:
                count["both_python3_5_tests_no_taco_exact_text"] += 1
    result = {
        "schema_version": "p16_codecontests_first_train_shard_audit_v2",
        "source_repo": "deepmind/code_contests",
        "revision": "802411c3010cb00d1b05bad57ca77365a3c699d6",
        "shard": shard.name,
        "shard_sha256": sha256_file(shard),
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "difficulty_code_counts": dict(sorted(difficulty_counts.items())),
        "counts": dict(sorted(count.items())),
        "test_count_distribution": {
            kind: {"min": min(values), "median": statistics.median(values),
                   "max": max(values), "zero": sum(v == 0 for v in values),
                   "at_least_5": sum(v >= 5 for v in values)}
            for kind, values in test_count.items()
        },
        "limitations": [
            "One first train shard only; not a random sample or full-population estimate.",
            "Test counts are raw counts, not independently validated cases.",
            "Correct/incorrect dataset solutions were not executed or reclassified.",
            "Only exact normalized-text overlap with TACO train was checked.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"rows": len(rows), "counts": result["counts"],
                      "test_count_distribution": result["test_count_distribution"]},
                     sort_keys=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("shard", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.workspace, args.shard, args.output)
