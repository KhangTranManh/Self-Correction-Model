"""Build the Phase 7 CPU-only candidate pool with Phase 1-6 exclusions."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase4.lib.rollout import as_problem, sha256, verify
from phase5.scripts.prepare_candidates import (FINAL, STEP, previous_sources,
                                               question_key, records, value)


CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"
PRIOR_REPORT = ROOT / "phase5/data/candidates_v1/candidate_report.json"
ADDITIONAL = (
    ROOT / "phase5/data/candidates_v1/candidate_problems.jsonl",
    ROOT / "phase6/data/harness_confirmation_source_pool_v1.jsonl",
    ROOT / "phase6/data/pilot_v1.jsonl",
    ROOT / "phase6/data/fuzzy_hint_pilot_v1.jsonl",
    ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl",
)
OUTPUT_DIR = ROOT / "phase7/data/candidates_v1"


def rank(source_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|phase7_source_v1|{source_id}".encode()).hexdigest()


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite candidate directory: {OUTPUT_DIR}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_plan = config["source_plan"]
    seed = int(config["seed"])
    ceiling = int(source_plan["initial_generation_ceiling"])
    raw_path = ROOT / source_plan["source"]
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use CPython 3.10 for the frozen math verifier")
    if not raw_path.is_file() or not PRIOR_REPORT.is_file() or any(not p.is_file() for p in ADDITIONAL):
        raise FileNotFoundError("A required source or exclusion inventory is missing")
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    prior_files = {Path(name): expected for name, expected in
                   prior["exclusion_files_sha256"].items()}
    for path, expected in prior_files.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Prior-phase inventory changed: {path}")
    excluded_questions, excluded_indices, inventory = previous_sources(
        [*prior_files, *ADDITIONAL]
    )
    for path, expected in prior_files.items():
        if inventory[str(path.resolve())] != expected:
            raise RuntimeError(f"Prior-phase inventory hash changed: {path}")

    accepted: list[dict] = []
    reasons = Counter()
    seen_questions: set[str] = set()
    raw_rows = list(records(raw_path))
    for index, (row, _) in enumerate(raw_rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"Unexpected GSM8K schema at row {index}")
        question, reference_text = row["question"], row["answer"]
        key = question_key(question)
        if index in excluded_indices or key in excluded_questions:
            reasons["prior_source"] += 1
            continue
        if key in seen_questions:
            reasons["duplicate_question"] += 1
            continue
        seen_questions.add(key)
        if len(question.split()) > 120:
            reasons["question_too_long"] += 1
            continue
        steps = STEP.findall(reference_text)
        if not 2 <= len(steps) <= 6:
            reasons["step_count"] += 1
            continue
        try:
            for expression, declared in steps:
                if value(expression) != value(declared):
                    raise ValueError("Reference step mismatch")
            match = FINAL.search(reference_text)
            if match is None:
                raise ValueError("No numeric final answer")
            reference = str(value(match.group(1)))
        except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
            reasons["unverifiable_reference"] += 1
            continue
        source_id = f"phase7_gsm8k_train_{index}"
        problem = {
            "id": source_id, "dataset_index": index, "domain": "math",
            "question": question, "reference_answer": reference,
            "question_sha256": key, "reference_step_count": len(steps),
            "source_rank_sha256": rank(source_id, seed),
        }
        if not verify(as_problem(problem), reference)["passed"]:
            reasons["reference_failed_fresh_verifier"] += 1
            continue
        accepted.append(problem)

    accepted.sort(key=lambda row: (row["source_rank_sha256"], row["id"]))
    selected = accepted[:ceiling]
    if len(selected) != ceiling:
        raise RuntimeError(f"Only {len(selected)} eligible sources; requested {ceiling}")
    OUTPUT_DIR.mkdir(parents=True)
    output = OUTPUT_DIR / "candidate_problems.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in selected), encoding="utf-8", newline="\n")
    report = {
        "schema_version": "phase7_candidate_report_v1",
        "status": "cpu_candidates_frozen_no_model_answers",
        "raw_train_sha256": sha256(raw_path),
        "raw_train_rows": len(raw_rows),
        "phase5_prior_inventory_report_sha256": sha256(PRIOR_REPORT),
        "prior_inventory_files_verified": len(prior_files),
        "additional_exclusion_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in ADDITIONAL},
        "excluded_question_hashes": len(excluded_questions),
        "excluded_gsm8k_indices": len(excluded_indices),
        "eligible_after_cpu_rules_and_verifier": len(accepted),
        "selected": len(selected),
        "candidate_manifest_sha256": sha256(output),
        "selection_rule": "SHA256(seed|phase7_source_v1|phase7_gsm8k_train_<index>)",
        "seed": seed,
        "rejected_by_first_reason": dict(sorted(reasons.items())),
        "python": sys.version.split()[0],
        "gpu_used": False,
        "phase7_initial_answers_generated": False,
    }
    (OUTPUT_DIR / "candidate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: report[key] for key in (
        "eligible_after_cpu_rules_and_verifier", "selected", "candidate_manifest_sha256",
        "prior_inventory_files_verified", "gpu_used")}, indent=2))


if __name__ == "__main__":
    main()
