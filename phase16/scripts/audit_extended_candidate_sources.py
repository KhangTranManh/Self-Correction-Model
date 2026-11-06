"""CPU text/provenance preflight for additional Phase 16 code sources.

This never executes dataset tests or solutions. Text matches are candidates,
not C-upper-inclusive eligibility. Network-backed scans use projected columns.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_c_contract_pool import AGGREGATE, CLOSED, EXPLICIT_BOUNDS, SEQUENCE  # noqa: E402
from taco_inventory import normalized_text, sha256_file  # noqa: E402


ELEMENT_OPERATION = re.compile(
    r"\b(?:count|number|sum|total|select|filter|collect|return|find)\b", re.I
)
RANGE_ENTITY = re.compile(
    r"\b(?:elements?|items?|values?|numbers?|integers?|entries|points?)\b", re.I
)


def text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def flags(text: str) -> dict[str, bool]:
    result = {
        "explicit_bounds": bool(EXPLICIT_BOUNDS.search(text)),
        "closed_language": bool(CLOSED.search(text)),
        "aggregate": bool(AGGREGATE.search(text)),
        "sequence": bool(SEQUENCE.search(text)),
        "element_operation": bool(ELEMENT_OPERATION.search(text)),
        "range_entity": bool(RANGE_ENTITY.search(text)),
    }
    result["four_text_gates"] = all(result[k] for k in
                                    ("explicit_bounds", "closed_language", "aggregate", "sequence"))
    result["conservative_text_gate"] = result["four_text_gates"] and all(
        result[k] for k in ("element_operation", "range_entity")
    )
    return result


def prior_hashes(workspace: Path) -> set[str]:
    manifest = workspace / "phase16/results/four_source_candidate_audit_v2.json"
    if not manifest.exists():
        return set()
    return {row["normalized_text_sha256"] for row in
            json.loads(manifest.read_text(encoding="utf-8"))["rows"]}


def audit_leetcode(workspace: Path, previous: set[str]) -> dict:
    paths = sorted((workspace / "phase16/data/extended_source_raw/leetcodedataset").glob("*.jsonl"))
    counts = Counter()
    candidates = []
    ids, hashes = set(), defaultdict(list)
    files = []
    for path in paths:
        files.append({"path": str(path.relative_to(workspace)), "sha256": sha256_file(path)})
        split = "test" if "-test." in path.name else "train"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            source_id = str(row["task_id"])
            text = row.get("problem_description") or ""
            digest = text_hash(text)
            item_flags = flags(text)
            counts["rows"] += 1
            counts[f"split_{split}"] += 1
            counts["duplicate_id_rows"] += source_id in ids
            ids.add(source_id)
            hashes[digest].append(source_id)
            counts["exact_prior_overlap"] += digest in previous
            counts["four_text_gates"] += item_flags["four_text_gates"]
            counts["conservative_text_gate"] += item_flags["conservative_text_gate"]
            if item_flags["four_text_gates"]:
                tests = row.get("input_output") or []
                candidates.append({
                    "id": source_id, "question_id": str(row.get("question_id")),
                    "split": split, "difficulty": row.get("difficulty"),
                    "estimated_date": str(row.get("estimated_date")),
                    "tags": row.get("tags") or [], "entry_point": row.get("entry_point"),
                    "source_text_sha256": digest, "exact_prior_overlap": digest in previous,
                    "declared_io_examples": len(tests) if isinstance(tests, list) else None,
                    "assert_count_text": (row.get("test") or "").count("assert "),
                    "flags": item_flags,
                })
    counts["exact_within_source_duplicate_groups"] = sum(len(v) > 1 for v in hashes.values())
    return {"revision": "215604aeed660029df7de2fea5a4d7b6ed476a08",
            "files": files, "counts": dict(counts), "candidates": candidates}


def audit_streaming(repo: str, config: str, split: str, columns: list[str],
                    text_column: str, id_column: str, previous: set[str], limit: int | None) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(repo, config, split=split, streaming=True).select_columns(columns)
    counts = Counter()
    candidates = []
    hashes = defaultdict(int)
    started = time.time()
    for row in dataset:
        text = row.get(text_column) or ""
        source_id = str(row.get(id_column))
        digest = text_hash(text)
        item_flags = flags(text)
        counts["rows"] += 1
        counts["exact_prior_overlap"] += digest in previous
        counts["four_text_gates"] += item_flags["four_text_gates"]
        counts["conservative_text_gate"] += item_flags["conservative_text_gate"]
        hashes[digest] += 1
        if item_flags["four_text_gates"]:
            candidate = {"id": source_id, "text": text,
                         "source_text_sha256": digest,
                         "exact_prior_overlap": digest in previous, "flags": item_flags}
            for key in columns:
                if key not in {text_column, id_column}:
                    value = row.get(key)
                    if key == "metadata" and isinstance(value, dict):
                        value = {k: value.get(k) for k in ("seed_ids", "prompt_id", "row_id")}
                    candidate[key] = value
            candidates.append(candidate)
        if counts["rows"] % 25000 == 0:
            print(json.dumps({"repo": repo, "rows": counts["rows"],
                              "hits": counts["four_text_gates"],
                              "seconds": round(time.time() - started, 1)}), flush=True)
        if limit is not None and counts["rows"] >= limit:
            break
    counts["exact_within_source_duplicate_groups"] = sum(n > 1 for n in hashes.values())
    return {"repo": repo, "config": config, "split": split,
            "scan_limit": limit, "counts": dict(counts), "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", choices=("leetcode", "kodcode", "open_r1"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    workspace = args.workspace.resolve()
    previous = prior_hashes(workspace)
    if args.source == "leetcode":
        result = audit_leetcode(workspace, previous)
    elif args.source == "kodcode":
        result = audit_streaming("KodCode/KodCode-V1", "default", "train",
                                 ["question", "question_id", "subset", "metadata",
                                  "benchmark_similarity", "benchmark_task_id"],
                                 "question", "question_id", previous, args.limit)
        result["revision"] = "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5"
    else:
        result = audit_streaming("open-r1/codeforces", "verifiable", "train",
                                 ["id", "description", "contest_id", "aliases", "generated_tests",
                                  "official_tests_complete", "input_mode", "executable"],
                                 "description", "id", previous, args.limit)
        result["revision"] = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
    document = {
        "schema_version": "p16_extended_source_text_audit_v1",
        "source": args.source, "result": result,
        "interpretation": "CPU text/provenance preflight only; zero tasks are certified by this audit.",
    }
    args.output.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
