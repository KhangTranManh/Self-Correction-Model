"""Freeze a fresh CPU-only source pool for a future probe-routed blind test."""

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
P7_REPORT = ROOT / "phase7/data/candidates_v1/candidate_report.json"
EXTRA = (
    ROOT / "phase5/data/candidates_v1/candidate_problems.jsonl",
    ROOT / "phase6/data/harness_confirmation_source_pool_v1.jsonl",
    ROOT / "phase6/data/pilot_v1.jsonl",
    ROOT / "phase6/data/fuzzy_hint_pilot_v1.jsonl",
    ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl",
    ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl",
)
OUT = ROOT / "phase8/data/fresh_source_pool_v1"
SEED = 20260925
COUNT = 400


def rank(index: int) -> str:
    return hashlib.sha256(f"phase8_fresh_pool_v1|{SEED}|{index}".encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite frozen pool: {OUT}")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use CPython 3.10 to match the prior verifier environment")
    p5 = json.loads(P5_REPORT.read_text(encoding="utf-8"))
    p7 = json.loads(P7_REPORT.read_text(encoding="utf-8"))
    if sha256(RAW) != p7["raw_train_sha256"]:
        raise RuntimeError("Raw dataset changed")
    prior = {Path(path): digest for path, digest in p5["exclusion_files_sha256"].items()}
    for path, digest in prior.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Prior source inventory changed: {path}")
    for path in EXTRA[:-1]:
        expected = p7["additional_exclusion_sha256"][str(path.relative_to(ROOT)).replace("/", "\\")]
        if sha256(path) != expected:
            raise RuntimeError(f"Phase 6 exclusion changed: {path}")
    if sha256(EXTRA[-1]) != p7["candidate_manifest_sha256"]:
        raise RuntimeError("Phase 7 candidate pool changed")
    excluded_questions, excluded_indices, inventory = previous_sources([*prior, *EXTRA])
    accepted, seen, reasons = [], set(), Counter()
    raw = list(records(RAW))
    for index, (row, _) in enumerate(raw):
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
            "id": f"phase8_gsm8k_train_{index}", "dataset_index": index,
            "domain": "math", "question": question,
            "question_sha256": key, "reference_answer": reference,
            "reference_step_count": len(steps), "source_rank_sha256": rank(index),
        }
        if not verify(as_problem(problem), reference)["passed"]:
            reasons["reference_failed_verifier"] += 1
            continue
        accepted.append(problem)
    accepted.sort(key=lambda row: (row["source_rank_sha256"], row["id"]))
    if len(accepted) < COUNT:
        raise RuntimeError(f"Only {len(accepted)} fresh eligible sources")
    selected = accepted[:COUNT]
    OUT.mkdir(parents=True)
    path = OUT / "candidate_problems.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8", newline="\n")
    report = {
        "schema_version": "phase8_fresh_source_pool_v1",
        "status": "protected_source_pool_frozen_before_initial_answers_or_probe_scores",
        "raw_dataset_sha256": sha256(RAW),
        "prior_phase_inventory_files_verified": len(prior),
        "additional_exclusions": {str(path.relative_to(ROOT)): sha256(path) for path in EXTRA},
        "all_exclusion_files_verified": len(inventory),
        "eligible_after_exclusions_and_verifier": len(accepted),
        "selected": len(selected), "seed": SEED,
        "selection_rule": "lexicographic SHA256(phase8_fresh_pool_v1|seed|raw_index)",
        "candidate_manifest_sha256": sha256(path),
        "rejected_by_first_reason": dict(sorted(reasons.items())),
        "python": sys.version.split()[0], "gpu_used": False,
        "initial_answers_generated": False,
        "protected_labels_opened": False,
    }
    (OUT / "candidate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
