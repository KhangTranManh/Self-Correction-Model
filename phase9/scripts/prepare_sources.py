"""Freeze the Phase 9 fresh source pool (40 development + 400 protected).

Several Phase 1-4 inventory files hashed by the Phase 5 audit are missing or
changed on this workstation, so strict hash re-verification is impossible.
Instead this builder (1) reproduces the Phase 8 selection from the available
inventory and requires an exact match to the frozen Phase 8 manifest and
eligible count, then (2) over-excludes: every JSON/JSONL under the Phase 1-4
directories plus all Phase 5-8 pools. Over-exclusion can only remove
candidates, never admit a previously used source.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase4.lib.rollout import as_problem, sha256, verify  # noqa: E402
from phase5.scripts.prepare_candidates import (  # noqa: E402
    FINAL, STEP, previous_sources, question_key, records, value,
)


RAW = ROOT / "phase5/data/raw/gsm8k_train.jsonl"
P5_REPORT = ROOT / "phase5/data/candidates_v1/candidate_report.json"
P8_POOL = ROOT / "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl"
P8_REPORT = P8_POOL.with_name("candidate_report.json")
P8_EXTRA = (
    ROOT / "phase5/data/candidates_v1/candidate_problems.jsonl",
    ROOT / "phase6/data/harness_confirmation_source_pool_v1.jsonl",
    ROOT / "phase6/data/pilot_v1.jsonl",
    ROOT / "phase6/data/fuzzy_hint_pilot_v1.jsonl",
    ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl",
    ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl",
)
OVER_EXCLUDE_DIRS = ("phase1", "phase2", "phase3", "phase4", "phase5/data",
                     "phase6/data", "phase7/data")
OUT = ROOT / "phase9/data/source_pool_v1"
SEED = 20260930
DEV, PROTECTED = 40, 400


def eligible(excluded_questions: set, excluded_indices: set, prefix: str,
             rank_key) -> tuple[list[dict], Counter]:
    accepted, seen, reasons = [], set(), Counter()
    for index, (row, _) in enumerate(records(RAW)):
        question, answer = row["question"], row["answer"]
        key = question_key(question)
        if index in excluded_indices or key in excluded_questions:
            reasons["prior_source"] += 1
            continue
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        if len(question.split()) > 120:
            reasons["question_too_long"] += 1
            continue
        steps = STEP.findall(answer)
        if not 2 <= len(steps) <= 6:
            reasons["step_count"] += 1
            continue
        try:
            if any(value(expr) != value(declared) for expr, declared in steps):
                raise ValueError("Reference step mismatch")
            match = FINAL.search(answer)
            if match is None:
                raise ValueError("Missing final answer")
            reference = str(value(match.group(1)))
        except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
            reasons["unverifiable_reference"] += 1
            continue
        problem = {
            "id": f"{prefix}_{index}", "dataset_index": index, "domain": "math",
            "question": question, "question_sha256": key, "reference_answer": reference,
            "reference_step_count": len(steps), "source_rank_sha256": rank_key(index),
        }
        if not verify(as_problem(problem), reference)["passed"]:
            reasons["reference_failed_verifier"] += 1
            continue
        accepted.append(problem)
    accepted.sort(key=lambda row: (row["source_rank_sha256"], row["id"]))
    return accepted, reasons


def json_files(relative: str) -> list[Path]:
    # Skip raw dataset folders (e.g. phase5/data/raw/gsm8k_train.jsonl): they list
    # every question and are the sampling frame, not evidence of prior use.
    return sorted(p for p in (ROOT / relative).rglob("*")
                  if p.is_file() and p.suffix in (".json", ".jsonl") and "raw" not in p.parts)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite frozen pool: {OUT}")
    p8 = json.loads(P8_REPORT.read_text(encoding="utf-8"))
    if lf_sha256(RAW) != p8["raw_dataset_sha256"]:
        raise RuntimeError("Raw GSM8K train file changed")

    # (1) Reproduce the Phase 8 selection from the inventory available here.
    p5 = json.loads(P5_REPORT.read_text(encoding="utf-8"))
    listed = [Path(path.replace("\\", "/").replace("D:/AGI/", str(ROOT).replace("\\", "/") + "/"))
              for path in p5["exclusion_files_sha256"]]
    available = [path for path in listed if path.exists()]
    questions, indices, _ = previous_sources([*available, *P8_EXTRA])
    p8_rank = lambda i: hashlib.sha256(f"phase8_fresh_pool_v1|20260925|{i}".encode()).hexdigest()
    p8_accepted, _ = eligible(questions, indices, "phase8_gsm8k_train", p8_rank)
    frozen_p8 = [row["id"] for row in records_list(P8_POOL)]
    reproduced = [row["id"] for row in p8_accepted[:400]]
    reproduction = {
        "phase5_listed_inventory_files": len(listed),
        "available_inventory_files": len(available),
        "eligible_reproduced": len(p8_accepted),
        "eligible_recorded": p8["eligible_after_exclusions_and_verifier"],
        "selection_matches_frozen_phase8": reproduced == frozen_p8,
    }
    if not reproduction["selection_matches_frozen_phase8"] or \
            len(p8_accepted) != p8["eligible_after_exclusions_and_verifier"]:
        raise RuntimeError(f"Phase 8 selection not reproduced: {reproduction}")

    # (2) Over-exclude everything known, including the opened Phase 8 pool.
    extra_files = [path for rel in OVER_EXCLUDE_DIRS for path in json_files(rel)]
    roots = [*available, *P8_EXTRA, P8_POOL, *extra_files]
    questions, indices, inventory = previous_sources(roots)
    rank = lambda i: hashlib.sha256(f"phase9_source_pool_v1|{SEED}|{i}".encode()).hexdigest()
    accepted, reasons = eligible(questions, indices, "phase9_gsm8k_train", rank)
    phase8_ids = {row["dataset_index"] for row in records_list(P8_POOL)}
    if any(row["dataset_index"] in phase8_ids for row in accepted):
        raise RuntimeError("Phase 8 source leaked into Phase 9 eligibility")
    if len(accepted) < DEV + PROTECTED:
        raise RuntimeError(f"Only {len(accepted)} eligible sources")
    OUT.mkdir(parents=True)
    report = {"schema_version": "phase9_source_pool_v1",
              "status": "frozen_before_any_phase9_generation",
              "raw_dataset_sha256": lf_sha256(RAW),
              "phase8_reproduction": reproduction,
              "exclusion_files_scanned": len(inventory),
              "over_exclusion_dirs": list(OVER_EXCLUDE_DIRS),
              "eligible_after_exclusions": len(accepted),
              "rejected_by_first_reason": dict(sorted(reasons.items())),
              "seed": SEED,
              "selection_rule": "lexicographic SHA256(phase9_source_pool_v1|seed|raw_index); "
                                "first 40 development, next 400 protected",
              "protected_labels_opened": False}
    for name, rows in (("development", accepted[:DEV]),
                       ("protected", accepted[DEV:DEV + PROTECTED])):
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                                for row in rows), encoding="utf-8", newline="\n")
        report[f"{name}_rows"] = len(rows)
        report[f"{name}_sha256"] = sha256(path)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


def lf_sha256(path: Path) -> str:
    """SHA-256 of LF-normalized bytes (Git blob form; Windows checkouts use CRLF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def records_list(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main()
