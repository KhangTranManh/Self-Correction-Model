"""Build a four-source *candidate inventory*, never a frozen study pool.

Only downloaded shards are read. No dataset or model code is executed.
Every row remains in the audit, including duplicates and historical reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_c_contract_pool import AGGREGATE, CLOSED, EXPLICIT_BOUNDS, SEQUENCE  # noqa: E402
from audit_codecontests_subset import paired_count, python3_available  # noqa: E402
from taco_inventory import normalized_text, sha256_file  # noqa: E402


REVISIONS = {
    "taco": "d593ed0a2becbbc952230bb89be09189bf1056dc",
    "codecontests": "802411c3010cb00d1b05bad57ca77365a3c699d6",
    "humanevalplus": "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
    "mbppplus": "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
    "livecodebench_v1": "28efd4d97f479657d49989f83e30b9d51481ad9e",
}


def digest(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def contract_flags(text: str) -> dict[str, bool]:
    return {
        "explicit_bounds": bool(EXPLICIT_BOUNDS.search(text)),
        "closed_language": bool(CLOSED.search(text)),
        "aggregate": bool(AGGREGATE.search(text)),
        "sequence": bool(SEQUENCE.search(text)),
    }


def iter_rows(paths: list[Path], columns: list[str]):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=256):
            yield from batch.to_pylist()


def audit(workspace: Path) -> dict:
    base = workspace / "phase16/data"
    taco_meta = [json.loads(line) for line in
                 (base / "taco_inventory_v2/train_metadata.jsonl").open(encoding="utf-8")]
    overlap = json.loads((base / "taco_inventory_v2/exact_overlap.json").read_text(encoding="utf-8"))
    history = {row["id"] for row in overlap["historical_exact_hits"]}
    duplicate_later = {item for group in overlap["within_taco_exact_text_groups"]
                       for item in group[1:]}
    result_rows = []
    files = defaultdict(list)

    def add(source: str, source_id: str, text: str, io_mode: str, family: str,
            preflight: str, historical: str) -> None:
        flags = contract_flags(text)
        result_rows.append({
            "id": f"{source}:{source_id}", "source": source,
            "revision": REVISIONS[source], "source_id": str(source_id),
            "normalized_text_sha256": digest(text), "io_mode": io_mode,
            "family_key_provisional": family, "preflight": preflight,
            "historical_status": historical, "c_text_flags": flags,
            "c_all_text_gates": all(flags.values()),
            "oracle_status": "not_validated", "role": "unassigned",
        })

    taco_paths = sorted((base / "taco_raw/ALL").glob("train-*.parquet"))
    files["taco"] = [{"path": str(p.relative_to(workspace)), "sha256": sha256_file(p)}
                     for p in taco_paths]
    taco_count = 0
    for item in iter_rows(taco_paths, ["question"]):
        meta = taco_meta[taco_count]
        taco_count += 1
        preflight = (
            "not_call_based" if meta["io_mode"] != "call_based" else
            "historical_exact" if meta["id"] in history else
            "within_source_duplicate" if meta["id"] in duplicate_later else
            "picture" if meta["picture_flag"] else "metadata_candidate"
        )
        add("taco", meta["id"], item["question"] or "", meta["io_mode"],
            meta.get("canonical_url") or meta["id"], preflight,
            "exact_hit" if meta["id"] in history else "not_resolved")
    if taco_count != len(taco_meta):
        raise ValueError("TACO raw/metadata count mismatch")

    cc_paths = sorted((base / "codecontests_candidate_raw/data").glob("train-*.parquet"))
    files["codecontests"] = [{"path": str(p.relative_to(workspace)), "sha256": sha256_file(p)}
                             for p in cc_paths]
    cc_columns = ["description", "public_tests", "private_tests", "generated_tests",
                  "solutions", "incorrect_solutions", "input_file", "output_file",
                  "cf_contest_id", "cf_index"]
    for index, item in enumerate(iter_rows(cc_paths, cc_columns)):
        test_counts = [paired_count(item.get(f"{kind}_tests"))
                       for kind in ("public", "private", "generated")]
        test_gate = all(paired for _, paired in test_counts) and sum(
            n for n, _ in test_counts) >= 5
        candidate = (test_gate and not (item.get("input_file") or item.get("output_file"))
                     and python3_available(item.get("solutions"))
                     and python3_available(item.get("incorrect_solutions")))
        contest = item.get("cf_contest_id") or "unknown"
        family = f"cf:{contest}" if str(contest) not in {"0", "", "None"} else f"cc:unknown:{index}"
        add("codecontests", str(index), item.get("description") or "", "standard_input",
            family, "metadata_candidate" if candidate else "metadata_reject",
            "not_resolved")

    for source, filename, text_field, io_mode in (
        ("humanevalplus", "test-00000-of-00001-5973903632b82d40.parquet", "prompt", "call_based"),
        ("mbppplus", "test-00000-of-00001-d5781c9c51e02795.parquet", "prompt", "call_based"),
        ("livecodebench_v1", "test-00000-of-00003.parquet", "question_content", "standard_input"),
    ):
        paths = list((base / "four_source_raw" / source).rglob(filename))
        if len(paths) != 1:
            raise ValueError(f"expected one shard for {source}: {paths}")
        path = paths[0]
        files[source] = [{"path": str(path.relative_to(workspace)), "sha256": sha256_file(path)}]
        columns = (["task_id", "prompt"] if source != "livecodebench_v1" else
                   ["question_id", "question_content", "platform", "contest_id", "metadata"])
        for index, item in enumerate(iter_rows(paths, columns)):
            sid = str(item.get("task_id") if source != "livecodebench_v1" else item.get("question_id"))
            if source == "humanevalplus":
                historical = "known_all_164_used_phase1"
                family = f"humaneval:{sid}"
            elif source == "mbppplus":
                historical = "id_level_audit_required_phase1_reuse"
                family = f"mbpp:{sid}"
            else:
                historical = "not_resolved"
                family = f"lcb:{item.get('platform')}:{item.get('contest_id')}"
            actual_io = io_mode
            if source == "livecodebench_v1":
                try:
                    detail = json.loads(item.get("metadata") or "{}")
                except (json.JSONDecodeError, TypeError):
                    detail = {}
                actual_io = "call_based" if detail.get("func_name") else "standard_input"
            add(source, sid if sid != "None" else str(index), item[text_field] or "",
                actual_io, family, "source_row_only", historical)

    by_hash = defaultdict(list)
    for row in result_rows:
        by_hash[row["normalized_text_sha256"]].append(row)
    for group in by_hash.values():
        sources = sorted({row["source"] for row in group})
        for row in group:
            row["exact_duplicate_sources"] = sources if len(group) > 1 else []
            if row["source"] == "codecontests" and "taco" in sources:
                row["preflight"] = "exact_taco_overlap"

    counts = {source: dict(Counter(
        {"rows": sum(r["source"] == source for r in result_rows),
         "c_all_text_gates": sum(r["source"] == source and r["c_all_text_gates"]
                                 for r in result_rows),
         "metadata_candidates": sum(r["source"] == source and r["preflight"] == "metadata_candidate"
                                    for r in result_rows),
         "exact_duplicate_rows": sum(r["source"] == source and bool(r["exact_duplicate_sources"])
                                     for r in result_rows),
        }
    )) for source in REVISIONS}
    return {
        "schema_version": "p16_four_source_candidate_audit_v2",
        "purpose": "CPU source inventory only; not a frozen pool or GPU release",
        "sources": files, "counts": counts, "rows": result_rows,
        "limitations": [
            "CodeContests: only 3/39 train shards; LiveCodeBench: only 1/3 release_v1 shards.",
            "Exact normalized text only; cross-source near duplicates and historical family reuse unresolved.",
            "EvalPlus extends HumanEval/MBPP; HumanEval/164 used in Phase 1 and MBPP needs ID-level history audit.",
            "No task-specific contract, independent oracle, sandboxed execution or natural A0 error verified.",
            "C text flags are high-recall-ish heuristics, neither necessary nor sufficient eligibility.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = audit(args.workspace.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
