"""CPU audit of the Phase 5 candidate pool before any GPU use."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase4.lib.rollout import as_problem, read_jsonl, sha256, verify
from phase5.scripts.prepare_candidates import previous_sources, question_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path,
                        default=Path("phase5/data/candidates_v1/candidate_problems.jsonl"))
    parser.add_argument("--report", type=Path,
                        default=Path("phase5/data/candidates_v1/verifier_audit.json"))
    parser.add_argument("--exclude", type=Path, required=True, action="append")
    args = parser.parse_args()
    rows = read_jsonl(args.candidates)
    selection = json.loads(args.candidates.with_name("candidate_report.json").read_text(encoding="utf-8"))
    if len(rows) != selection["selected"] or sha256(args.candidates) != selection["candidate_problems_sha256"]:
        raise ValueError("Candidate count or hash mismatch")
    excluded_questions, excluded_indices, inventory = previous_sources(args.exclude)
    if inventory != selection["exclusion_files_sha256"]:
        raise ValueError("Prior-phase exclusion inventory changed")
    errors, seen = [], set()
    for row in rows:
        key = question_key(row["question"])
        if row["id"] in seen or key in excluded_questions or row["dataset_index"] in excluded_indices:
            errors.append([row["id"], "duplicate_or_prior_source"])
        seen.add(row["id"])
        result = verify(as_problem(row), row["reference_answer"])
        if not result["passed"]:
            errors.append([row["id"], "reference_failed_math_verifier"])
    report = {
        "all_passed": not errors,
        "rows": len(rows),
        "verified_references": len(rows) - sum(e[1] == "reference_failed_math_verifier" for e in errors),
        "source_overlap": sum(e[1] == "duplicate_or_prior_source" for e in errors),
        "errors": errors[:30],
        "candidate_problems_sha256": sha256(args.candidates),
        "exclusion_inputs": list(inventory),
        "runtime": sys.version.split()[0],
        "gpu_used": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "exclusion_inputs"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
