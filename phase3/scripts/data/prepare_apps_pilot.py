"""Select and freeze a diverse 300-problem APPS pilot before model inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.apps_verifier import verify


if hasattr(sys, "set_int_max_str_digits"):
    # APPS contains legitimate stress tests with integers longer than Python
    # 3.10's later-added JSON conversion guard. Preserve those source tests.
    sys.set_int_max_str_digits(0)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "apps_pilot.yaml"


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            row["_source_index"] = line_number - 1
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _normal_problem(text: str) -> str:
    return " ".join(text.split()).casefold()


def _stable_rank(seed: int, problem_id: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def _length_bin(text: str) -> str:
    length = len(text)
    if length < 800:
        return "short_lt_800"
    if length < 1600:
        return "medium_800_1599"
    if length < 3200:
        return "long_1600_3199"
    return "very_long_ge_3200"


def _round_robin_length_bins(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    bin_order = (
        "short_lt_800",
        "medium_800_1599",
        "long_1600_3199",
        "very_long_ge_3200",
    )
    grouped: dict[str, deque[dict[str, Any]]] = {}
    for name in bin_order:
        values = [row for row in rows if row["_length_bin"] == name]
        values.sort(key=lambda row: (_stable_rank(seed, int(row["id"])), row["_source_index"]))
        grouped[name] = deque(values)
    ordered: list[dict[str, Any]] = []
    while any(grouped.values()):
        for name in bin_order:
            if grouped[name]:
                ordered.append(grouped[name].popleft())
    return ordered


def _load_existing_problem_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    hashes: set[str] = set()
    for row in _read_jsonl(path):
        problem = str(row.get("problem") or row.get("question") or "")
        if problem.strip():
            hashes.add(hashlib.sha256(_normal_problem(problem).encode()).hexdigest())
    return hashes


def _parse_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    question = row.get("question")
    if not isinstance(row.get("id"), int):
        return None, "untraceable_source_id"
    if not isinstance(question, str) or not question.strip():
        return None, "missing_question"
    try:
        tests = json.loads(row.get("input_output") or "")
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_input_output_json"
    try:
        solutions = json.loads(row.get("solutions") or "[]")
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_solutions_json"
    if not isinstance(tests, dict):
        return None, "invalid_test_structure"
    inputs, outputs = tests.get("inputs"), tests.get("outputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(outputs, list)
        or len(inputs) != len(outputs)
    ):
        return None, "missing_or_mismatched_tests"
    if not isinstance(solutions, list) or not any(
        isinstance(solution, str) and solution.strip() for solution in solutions
    ):
        return None, "missing_reference_solution"
    parsed = dict(row)
    parsed["_tests"] = tests
    parsed["_solutions"] = solutions
    parsed["_test_mode"] = "call_based" if tests.get("fn_name") else "standard_input"
    parsed["_source_host"] = urlparse(str(row.get("url") or "")).netloc or "unknown"
    parsed["_length_bin"] = _length_bin(question)
    return parsed, None


def _candidate(row: dict[str, Any], reference_solution: str, seed: int) -> dict[str, Any]:
    problem_id = int(row["id"])
    return {
        "id": f"apps_train_{problem_id}",
        "dataset": "apps",
        "split": "train",
        "domain": "code",
        "source_index": row["_source_index"],
        "problem_id": problem_id,
        "problem": row["question"],
        "starter_code": row.get("starter_code") or "",
        "reference_answer": reference_solution,
        "ground_truth": reference_solution,
        "tests": row["_tests"],
        "input_output_raw": row["input_output"],
        "difficulty": row["difficulty"],
        "url": row.get("url") or "",
        "source_host": row["_source_host"],
        "test_mode": row["_test_mode"],
        "test_count": len(row["_tests"]["inputs"]),
        "prompt_length_chars": len(row["question"]),
        "prompt_length_bin": row["_length_bin"],
        "source_file": "codeparrot/apps@21e74ddf8de1a21436da12e3e653065c5213e9d1/train.jsonl",
        "selection_reason": "deterministic_introductory_diversity_quota",
        "selection_seed": seed,
        "selected_before_model_inference": True,
        "eligible_for_expansion_pilot": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--existing-source",
        default=None,
        help="Optional existing source JSONL used only for normalized prompt-overlap exclusion.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    verifier_cfg = config["verifier"]
    seed = int(dataset_cfg["selection_seed"])
    target = int(dataset_cfg["pilot_size"])
    quotas = {key: int(value) for key, value in dataset_cfg["diversity_quotas"].items()}
    if sum(quotas.values()) != target:
        raise ValueError(f"Diversity quotas sum to {sum(quotas.values())}, expected {target}")

    raw_path = _resolve(dataset_cfg["raw_file"])
    existing_source = Path(args.existing_source) if args.existing_source else None
    existing_hashes = _load_existing_problem_hashes(existing_source)
    raw_rows = _read_jsonl(raw_path)
    exclusions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_problem_hashes: set[str] = set(existing_hashes)
    difficulty = dataset_cfg["difficulty"]

    for row in raw_rows:
        if row.get("difficulty") != difficulty:
            continue
        parsed, reason = _parse_row(row)
        problem_id = row.get("id")
        if reason is None and problem_id in seen_ids:
            reason = "duplicate_source_id"
        problem = str(row.get("question") or "")
        problem_hash = hashlib.sha256(_normal_problem(problem).encode()).hexdigest() if problem else ""
        if reason is None and problem_hash in seen_problem_hashes:
            reason = "duplicate_or_existing_problem_content"
        if reason:
            exclusions.append(
                {
                    "id": f"apps_train_{problem_id}",
                    "problem_id": problem_id,
                    "source_index": row["_source_index"],
                    "exclusion_reason": reason,
                    "stage": "preselection_validation",
                }
            )
            continue
        seen_ids.add(int(problem_id))
        seen_problem_hashes.add(problem_hash)
        eligible.append(parsed)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[f"{row['_source_host']}|{row['_test_mode']}"] .append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    selected_by_stratum: Counter[str] = Counter()
    reference_failures = 0
    print(
        f"[select] raw={len(raw_rows)} introductory={sum(r.get('difficulty') == difficulty for r in raw_rows)} "
        f"eligible_metadata={len(eligible)} target={target}",
        flush=True,
    )
    for stratum, quota in quotas.items():
        ranked = _round_robin_length_bins(grouped.get(stratum, []), seed)
        for row in ranked:
            if selected_by_stratum[stratum] >= quota:
                break
            reference_solution = None
            last_detail = "no_solution_checked"
            for solution in row["_solutions"]:
                if not isinstance(solution, str) or not solution.strip():
                    continue
                result = verify(
                    solution,
                    row["_tests"],
                    per_test_timeout_seconds=int(verifier_cfg["per_test_timeout_seconds"]),
                    memory_limit_mb=int(verifier_cfg["memory_limit_mb"]),
                    max_output_bytes=int(verifier_cfg["max_output_bytes"]),
                )
                last_detail = result["detail"]
                if result["passed"]:
                    reference_solution = solution
                    break
            if reference_solution is None:
                reference_failures += 1
                exclusions.append(
                    {
                        "id": f"apps_train_{row['id']}",
                        "problem_id": row["id"],
                        "source_index": row["_source_index"],
                        "exclusion_reason": "no_reference_solution_passed_local_verifier",
                        "verifier_detail": last_detail,
                        "stage": "reference_execution_validation",
                    }
                )
                continue
            selected.append(_candidate(row, reference_solution, seed))
            selected_ids.add(int(row["id"]))
            selected_by_stratum[stratum] += 1
            if len(selected) % 25 == 0:
                print(f"[select] validated={len(selected)}/{target}", flush=True)
        if selected_by_stratum[stratum] < quota:
            raise RuntimeError(
                f"Stratum {stratum!r} supplied {selected_by_stratum[stratum]}/{quota} "
                "reference-validated tasks; refusing to silently change the frozen diversity plan"
            )

    # Restore source order after deterministic, quota-aware selection.
    selected.sort(key=lambda row: row["source_index"])
    ids = [row["id"] for row in selected]
    content_hashes = [hashlib.sha256(_normal_problem(row["problem"]).encode()).hexdigest() for row in selected]
    if len(selected) != target or len(ids) != len(set(ids)) or len(content_hashes) != len(set(content_hashes)):
        raise RuntimeError("Final candidate uniqueness/count validation failed")

    candidate_path = _resolve(config["paths"]["candidates"])
    exclusions_path = _resolve(config["paths"]["selection_exclusions"])
    summary_path = _resolve(config["paths"]["candidate_summary"])
    _write_jsonl(candidate_path, selected)
    _write_jsonl(exclusions_path, exclusions)
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    source_counts = Counter(row["source_host"] for row in selected)
    mode_counts = Counter(row["test_mode"] for row in selected)
    length_counts = Counter(row["prompt_length_bin"] for row in selected)
    summary = {
        "dataset": dataset_cfg["name"],
        "revision": dataset_cfg["revision"],
        "split": dataset_cfg["split"],
        "difficulty": difficulty,
        "selection_seed": seed,
        "selection_used_model_correctness": False,
        "selected_before_model_inference": True,
        "raw_rows_inspected": len(raw_rows),
        "introductory_rows_inspected": sum(row.get("difficulty") == difficulty for row in raw_rows),
        "metadata_eligible_rows": len(eligible),
        "candidate_count": len(selected),
        "unique_candidate_ids": len(set(ids)),
        "existing_source_overlap": sum(
            hashlib.sha256(_normal_problem(row["problem"]).encode()).hexdigest() in existing_hashes
            for row in selected
        ),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "duplicate_content_count": len(content_hashes) - len(set(content_hashes)),
        "reference_execution_failures_before_backfill": reference_failures,
        "counts_by_source_host": dict(sorted(source_counts.items())),
        "counts_by_test_mode": dict(sorted(mode_counts.items())),
        "counts_by_prompt_length_bin": dict(sorted(length_counts.items())),
        "quota_plan": quotas,
        "quota_actual": dict(sorted(selected_by_stratum.items())),
        "test_count": {
            "total": sum(row["test_count"] for row in selected),
            "min": min(row["test_count"] for row in selected),
            "max": max(row["test_count"] for row in selected),
        },
        "candidate_ids_first_five_in_source_order": ids[:5],
        "raw_file": str(raw_path),
        "raw_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "candidate_file": str(candidate_path),
        "candidate_sha256": candidate_sha256,
        "exclusion_counts": dict(sorted(Counter(row["exclusion_reason"] for row in exclusions).items())),
        "config_file": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
