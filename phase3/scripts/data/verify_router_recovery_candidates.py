"""Fresh-verify Router Recovery V2 generations and admit strict same-origin pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine


REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."
BAD_CODE = re.compile(
    r"syntax|timeout|nameerror|typeerror|indentation|zerodivisionerror|"
    r"indexerror|keyerror|valueerror|attributeerror|recursionerror|memoryerror|"
    r"oserror|no code|empty",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def symbols(text: str) -> set[str]:
    return set(re.findall(r"(?:def|class)\s+([A-Za-z_]\w*)", text))


def stable(value: str) -> str:
    return hashlib.sha256(f"20260908|router_recovery_verify|{value}".encode()).hexdigest()


def pair_quality(correct: str, wrong: str, domain: str, wrong_detail: str) -> tuple[bool, list[str], float, float]:
    reasons = []
    ratio = min(len(correct), len(wrong)) / max(len(correct), len(wrong), 1)
    similarity = SequenceMatcher(None, correct, wrong).ratio()
    if not wrong or len(wrong) < 30:
        reasons.append("empty_or_too_short")
    if correct.strip() == wrong.strip():
        reasons.append("identical")
    if ratio < 0.50:
        reasons.append("length_ratio_below_0.5")
    # Character similarity is not a valid semantic/format gate for free-form
    # math reasoning from the same model. Length matching plus the fresh final-
    # answer verifier provide the hard constraints there. Code retains a strict
    # text/interface similarity gate.
    if domain == "code" and similarity < 0.40:
        reasons.append("similarity_too_low")
    if domain == "code":
        if BAD_CODE.search(wrong_detail):
            reasons.append("non_semantic_code_failure")
        correct_symbols = symbols(correct)
        if correct_symbols and not (correct_symbols & symbols(wrong)):
            reasons.append("interface_mismatch")
    return not reasons, reasons, ratio, similarity


def behavior_rows(pair: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, answer in (("KEEP", pair["correct_answer"]), ("REVISE", pair["wrong_answer"])):
        rows.append({
            "schema_version": "phase3_router_recovery_v2_behavior_v1",
            "construction_id": f"router_recovery_v2::{pair['source_id']}::{label}",
            "pair_id": pair["pair_id"],
            "source_id": pair["source_id"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "label": label,
            "class_id": 0 if label == "KEEP" else 1,
            "model_origin": pair["model_origin"],
            "same_model_origin": True,
            "source_pool": pair["source_pool"],
            "neutral_template_id": "canonical_review_v1",
            "messages": [pair["problem_message"], {"role": "assistant", "content": answer}, {"role": "user", "content": REVIEW}],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--existing-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--combined-output-dir", required=True)
    parser.add_argument("--minimum-pairs", type=int, default=200)
    parser.add_argument("--target-pairs", type=int, default=300)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_rows = read_jsonl(manifest_path)
    task_by_id = {row["task_id"]: row for row in manifest_rows}
    if len(task_by_id) != len(manifest_rows):
        raise RuntimeError("Duplicate task IDs in generation manifest")
    candidates = []
    for value in args.candidate:
        candidates.extend(read_jsonl(Path(value).resolve()))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate["task_id"] not in task_by_id:
            raise RuntimeError(f"Candidate task not in manifest: {candidate['task_id']}")
        grouped[candidate["task_id"]].append(candidate)

    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    admitted = []
    audit = []
    for task in manifest_rows:
        source = resolver.resolve(task["problem_ref"], task["source_id"])
        checks = []
        for candidate in grouped.get(task["task_id"], []):
            answer = str(candidate.get("raw_answer") or "").strip()
            check = verifier.verify(source, answer) if answer else {"passed": False, "detail": "empty"}
            checks.append((candidate, answer, check))

        known_correct = task.get("existing_verified_correct_answer")
        correct_options = []
        if known_correct:
            correct_answer = str(known_correct).strip()
            correct_check = verifier.verify(source, correct_answer)
            if correct_check["passed"]:
                correct_options.append((None, correct_answer, correct_check))
        else:
            correct_options.extend((candidate, answer, check) for candidate, answer, check in checks if check["passed"])

        options = []
        for correct_candidate, correct_answer, correct_check in correct_options:
            for wrong_candidate, wrong_answer, wrong_check in checks:
                if wrong_check["passed"] or (correct_candidate and wrong_candidate["candidate_id"] == correct_candidate["candidate_id"]):
                    continue
                accepted, reasons, ratio, similarity = pair_quality(
                    correct_answer, wrong_answer, task["domain"], str(wrong_check["detail"])
                )
                audit.append({
                    "task_id": task["task_id"],
                    "source_id": task["source_id"],
                    "candidate_id": wrong_candidate["candidate_id"],
                    "correct_candidate_id": correct_candidate["candidate_id"] if correct_candidate else None,
                    "dataset": task["dataset"],
                    "domain": task["domain"],
                    "model_origin": task["model_origin_to_sample"],
                    "fresh_verifier_passed": False,
                    "fresh_verifier_detail": wrong_check["detail"],
                    "length_ratio": ratio,
                    "similarity": similarity,
                    "accepted_pair_option": accepted,
                    "rejection_reasons": reasons,
                })
                if accepted:
                    options.append((similarity + ratio, correct_candidate, correct_answer, correct_check, wrong_candidate, wrong_answer, wrong_check, ratio, similarity))

        if not options:
            audit.append({
                "task_id": task["task_id"], "source_id": task["source_id"],
                "dataset": task["dataset"], "domain": task["domain"],
                "model_origin": task["model_origin_to_sample"], "accepted_pair_option": False,
                "rejection_reasons": ["no_strict_correct_wrong_pair"],
                "verified_correct_candidates": len(correct_options),
                "verified_wrong_candidates": sum(not check["passed"] for _, _, check in checks),
            })
            continue
        best = max(options, key=lambda item: (item[0], stable(item[4]["candidate_id"])))
        _, correct_candidate, correct_answer, correct_check, wrong_candidate, wrong_answer, wrong_check, ratio, similarity = best
        admitted.append({
            "schema_version": "phase3_router_recovery_v2_pair_v1",
            "pair_id": task["pair_id"],
            "source_id": task["source_id"],
            "dataset": task["dataset"],
            "domain": task["domain"],
            "problem_ref": task["problem_ref"],
            "problem_message": task["problem_message"],
            "model_origin": task["model_origin_to_sample"],
            "same_model_origin": True,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
            "correct_candidate_id": correct_candidate["candidate_id"] if correct_candidate else None,
            "wrong_candidate_id": wrong_candidate["candidate_id"],
            "correct_verifier_detail": correct_check["detail"],
            "wrong_verifier_detail": wrong_check["detail"],
            "length_ratio": ratio,
            "similarity": similarity,
            "source_pool": f"router_recovery_generation::{Path(args.manifest).stem}",
            "fresh_verified": True,
            "synthetic_wrong": False,
        })

    admitted.sort(key=lambda row: (row["domain"], row["dataset"], stable(row["pair_id"])))
    output = Path(args.output_dir).resolve()
    write_jsonl(output / "verified_pair_options.jsonl", admitted)
    write_jsonl(output / "candidate_audit.jsonl", audit)

    existing = read_jsonl(Path(args.existing_pairs).resolve())
    combined_by_source = {row["source_id"]: row for row in existing}
    duplicate_new = []
    for row in admitted:
        if row["source_id"] in combined_by_source:
            duplicate_new.append(row["source_id"])
        else:
            combined_by_source[row["source_id"]] = row
    newly_admitted = [row for row in admitted if row["source_id"] not in set(duplicate_new)]
    write_jsonl(output / "verified_pairs.jsonl", newly_admitted)
    combined = sorted(combined_by_source.values(), key=lambda row: (row["domain"], row["dataset"], stable(row["pair_id"])))
    combined_behavior = [member for pair in combined for member in behavior_rows(pair)]
    combined_output = Path(args.combined_output_dir).resolve()
    write_jsonl(combined_output / "verified_pairs.partial.jsonl", combined)
    write_jsonl(combined_output / "development_all.partial.jsonl", combined_behavior)

    summary = {
        "schema_version": "phase3_router_recovery_v2_batch_verification_v1",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "candidate_rows": len(candidates),
        "candidate_sources": len(grouped),
        "strict_pair_options": len(admitted),
        "duplicate_existing_sources": sorted(duplicate_new),
        "admitted_pairs": len(newly_admitted),
        "admitted_datasets": dict(Counter(row["dataset"] for row in newly_admitted)),
        "admitted_domains": dict(Counter(row["domain"] for row in newly_admitted)),
        "combined_pairs": len(combined),
        "combined_domains": dict(Counter(row["domain"] for row in combined)),
        "combined_datasets": dict(Counter(row["dataset"] for row in combined)),
        "minimum_pairs": args.minimum_pairs,
        "target_pairs": args.target_pairs,
        "minimum_gate_pass": len(combined) >= args.minimum_pairs,
        "fresh_correct_pass": len(newly_admitted),
        "fresh_wrong_fail": len(newly_admitted),
        "synthetic_wrong_answers": 0,
        "training_started": False,
    }
    write_json(output / "summary.json", summary)
    write_json(combined_output / "current_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
