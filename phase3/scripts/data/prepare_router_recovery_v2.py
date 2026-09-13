"""Build the CPU-available Router Recovery V2 pair pool and optional GPU manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine
from phase3.scripts.data.collect_apps_pilot import _prompt as build_apps_prompt
from phase3.scripts.data.collect_initial_attempts import _problem as build_problem_record
from phase3.scripts.data.collect_initial_attempts import build_prompt as build_phase1_prompt


REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."
SEED = 20260908


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}|router_recovery_v2|{value}".encode()).hexdigest()


def source_id(row: dict[str, Any]) -> str:
    return str(row.get("source_id", row.get("id", "")))


def protected_sources() -> tuple[set[str], dict[str, int]]:
    paths = {
        "probe_256": ROOT / "runs" / "representation_probe" / "probe_dataset.jsonl",
        "frozen_200": ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl",
        "decision_only_all": ROOT / "data" / "decision_only" / "decision_only_dataset.jsonl",
        "calibration_v1": ROOT / "data" / "router_calibration_v1" / "calibration_all.jsonl",
    }
    groups = {name: {source_id(row) for row in read_jsonl(path)} for name, path in paths.items()}
    union = set().union(*groups.values())
    return union, {name: len(values) for name, values in groups.items()}


def behavioral_rows(pair: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for label, answer in (("KEEP", pair["correct_answer"]), ("REVISE", pair["wrong_answer"])):
        result.append({
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
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "router_recovery_v2"))
    parser.add_argument("--target-pairs", type=int, default=300)
    parser.add_argument("--minimum-pairs", type=int, default=200)
    parser.add_argument("--dual-outcome-mbpp-tasks", type=int, default=160)
    parser.add_argument("--extra-math-tasks", type=int, default=60)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    protected, protected_counts = protected_sources()
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()

    v1_attempts = {row["id"]: row for row in read_jsonl(ROOT / "data" / "attempts" / "self_correction_v1" / "raw_attempts.jsonl")}
    base_attempts = {row["id"]: row for row in read_jsonl(ROOT / "data" / "attempts" / "base" / "raw_attempts.jsonl")}
    v1_apps = {row["id"]: row for row in read_jsonl(ROOT / "data" / "apps_pilot" / "attempts" / "self_correction_v1" / "raw_attempts.jsonl")}
    base_apps = {row["id"]: row for row in read_jsonl(ROOT / "data" / "apps_pilot" / "attempts" / "base" / "raw_attempts.jsonl")}
    apps_sources = {row["id"]: row for row in read_jsonl(ROOT / "data" / "apps_pilot" / "candidates" / "code_candidates.jsonl")}
    general_sources = {row["id"]: row for row in read_jsonl(ROOT / "data" / "source" / "problems.jsonl")}

    pairs: dict[str, dict[str, Any]] = {}
    audit = []

    # Existing strict same-origin pairs from the completed Stage 04 verifier.
    stage04 = read_jsonl(PROJECT_ROOT / "outputs" / "phase3_five_stage_pipeline" / "stages" / "04_same_origin" / "data" / "verified_pairs.jsonl")
    old_manifest = {row["source_id"]: row for row in read_jsonl(ROOT / "data" / "offline_next" / "future_same_origin_generation_manifest.jsonl")}
    for row in stage04:
        sid = row["source_id"]
        reasons = []
        if sid in protected:
            reasons.append("protected_source_overlap")
        task = old_manifest.get(sid)
        if not task:
            reasons.append("missing_problem_message")
        correct_check = None
        wrong_check = None
        if task and not reasons:
            source = resolver.resolve(row["problem_ref"], sid)
            correct_check = verifier.verify(source, row["correct_answer"])
            wrong_check = verifier.verify(source, row["wrong_answer"])
            if not correct_check["passed"]:
                reasons.append("correct_failed_fresh_verifier")
            if wrong_check["passed"]:
                reasons.append("wrong_passed_fresh_verifier")
        if reasons:
            audit.append({"source_id": sid, "source_pool": "stage04_verified", "accepted": False, "reasons": reasons})
            continue
        candidate = {
            "schema_version": "phase3_router_recovery_v2_pair_v1",
            "pair_id": f"router_recovery_v2::{sid}",
            "source_id": sid,
            "dataset": row["dataset"],
            "domain": row["domain"],
            "problem_ref": row["problem_ref"],
            "problem_message": task["problem_message"],
            "model_origin": row["model_origin"],
            "same_model_origin": True,
            "correct_answer": row["correct_answer"],
            "wrong_answer": row["wrong_answer"],
            "correct_verifier_detail": correct_check["detail"],
            "wrong_verifier_detail": wrong_check["detail"],
            "length_ratio": row["length_ratio"],
            "similarity": row["similarity"],
            "source_pool": "stage04_verified",
            "fresh_verified": True,
            "synthetic_wrong": False,
        }
        pairs[sid] = candidate
        audit.append({"source_id": sid, "source_pool": "stage04_verified", "accepted": True, "reasons": []})

    # Unused stochastic V1 GSM8K failures paired with the original correct V1 answer.
    generated_wrong = read_jsonl(ROOT / "data" / "router_calibration_v1" / "generation" / "verified_wrong.jsonl")
    for row in generated_wrong:
        sid = row["source_id"]
        reasons = []
        original = v1_attempts.get(sid)
        if sid in protected:
            reasons.append("protected_source_overlap")
        if sid in pairs:
            reasons.append("duplicate_existing_pair")
        if not original or not original.get("initial_correct"):
            reasons.append("missing_verified_correct_same_origin")
        if reasons:
            audit.append({"source_id": sid, "source_pool": "calibration_unused_generation", "accepted": False, "reasons": reasons})
            continue
        correct = str(original["initial_output"]).strip()
        wrong = str(row["messages"][1]["content"]).strip()
        ratio = min(len(correct), len(wrong)) / max(len(correct), len(wrong), 1)
        similarity = SequenceMatcher(None, correct, wrong).ratio()
        if ratio < 0.50:
            audit.append({"source_id": sid, "source_pool": "calibration_unused_generation", "accepted": False, "reasons": ["length_ratio_below_0.5"]})
            continue
        source = resolver.resolve(row["source_ref"], sid)
        correct_check = verifier.verify(source, correct)
        wrong_check = verifier.verify(source, wrong)
        if not correct_check["passed"] or wrong_check["passed"]:
            rejection = []
            if not correct_check["passed"]:
                rejection.append("correct_failed_fresh_verifier")
            if wrong_check["passed"]:
                rejection.append("wrong_passed_fresh_verifier")
            audit.append({"source_id": sid, "source_pool": "calibration_unused_generation", "accepted": False, "reasons": rejection})
            continue
        pairs[sid] = {
            "schema_version": "phase3_router_recovery_v2_pair_v1",
            "pair_id": f"router_recovery_v2::{sid}",
            "source_id": sid,
            "dataset": "gsm8k",
            "domain": "math",
            "problem_ref": row["source_ref"],
            "problem_message": row["messages"][0],
            "model_origin": "self_correction_v1",
            "same_model_origin": True,
            "correct_answer": correct,
            "wrong_answer": wrong,
            "correct_verifier_detail": correct_check["detail"],
            "wrong_verifier_detail": wrong_check["detail"],
            "length_ratio": ratio,
            "similarity": similarity,
            "source_pool": "calibration_unused_generation",
            "fresh_verified": True,
            "synthetic_wrong": False,
        }
        audit.append({"source_id": sid, "source_pool": "calibration_unused_generation", "accepted": True, "reasons": []})

    existing_pairs = sorted(pairs.values(), key=lambda row: (row["domain"], row["dataset"], stable(row["pair_id"])))
    existing_behavior = [member for pair in existing_pairs for member in behavioral_rows(pair)]

    # Known-correct code tasks: mine a plausible wrong output from the same origin.
    generation: dict[str, dict[str, Any]] = {}
    all_code_ids = sorted(set(v1_apps) | set(v1_attempts))
    for sid in all_code_ids:
        candidates = []
        if sid.startswith("apps_"):
            options = (("self_correction_v1", v1_apps.get(sid)), ("base", base_apps.get(sid)))
            problem_ref = f"data/apps_pilot/candidates/code_candidates.jsonl#id={sid}"
            dataset = "apps"
            prompt_text = build_apps_prompt(apps_sources[sid])
        elif sid.startswith("mbpp_"):
            options = (("self_correction_v1", v1_attempts.get(sid)), ("base", base_attempts.get(sid)))
            problem_ref = f"data/source/problems.jsonl#id={sid}"
            dataset = "mbpp"
            prompt_text = build_phase1_prompt(build_problem_record(general_sources[sid]))
        else:
            continue
        if sid in protected or sid in pairs:
            continue
        for origin, attempt in options:
            if attempt and attempt.get("initial_correct"):
                candidates.append((origin, attempt))
        if not candidates:
            continue
        origin, attempt = candidates[0]
        task_id = f"recovery_known_wrong::{origin}::{sid}"
        generation[task_id] = {
            "schema_version": "phase3_router_recovery_v2_generation_task_v1",
            "task_id": task_id,
            "pair_id": f"router_recovery_v2::{sid}",
            "source_id": sid,
            "problem_ref": problem_ref,
            "dataset": dataset,
            "domain": "code",
            "problem_message": {"role": "user", "content": prompt_text},
            "existing_verified_correct_answer": attempt["initial_output"],
            "existing_correct_answer_ref": f"{origin}_attempt::{sid}",
            "model_origin_to_sample": origin,
            "model_id_to_sample": "Kxck/Self_Correction_v1" if origin == "self_correction_v1" else "Qwen/Qwen2.5-7B-Instruct",
            "desired_new_state": "verified_wrong_but_plausible",
            "samples_requested": 12,
            "priority": "known_correct_code",
        }

    # Dual-outcome MBPP tasks may yield both a correct and plausible wrong V1 sample.
    mbpp_dual = []
    for sid, attempt in v1_attempts.items():
        if not sid.startswith("mbpp_") or sid in protected or sid in pairs:
            continue
        if any(task["source_id"] == sid for task in generation.values()):
            continue
        mbpp_dual.append((stable(sid), sid, attempt))
    for _, sid, attempt in sorted(mbpp_dual)[: args.dual_outcome_mbpp_tasks]:
        task_id = f"recovery_dual_outcome::self_correction_v1::{sid}"
        generation[task_id] = {
            "schema_version": "phase3_router_recovery_v2_generation_task_v1",
            "task_id": task_id,
            "pair_id": f"router_recovery_v2::{sid}",
            "source_id": sid,
            "problem_ref": f"data/source/problems.jsonl#id={sid}",
            "dataset": "mbpp",
            "domain": "code",
            "problem_message": {"role": "user", "content": build_phase1_prompt(build_problem_record(general_sources[sid]))},
            "existing_verified_correct_answer": None,
            "existing_correct_answer_ref": None,
            "model_origin_to_sample": "self_correction_v1",
            "model_id_to_sample": "Kxck/Self_Correction_v1",
            "desired_new_state": "one_verified_correct_and_one_plausible_wrong",
            "samples_requested": 16,
            "priority": "dual_outcome_mbpp",
        }

    # Extra known-correct math tasks fill the 40% math target if existing pairs fall short.
    existing_math = sum(pair["domain"] == "math" for pair in existing_pairs)
    math_needed = max(0, round(args.minimum_pairs * 0.40) - existing_math)
    math_candidates = []
    for sid, attempt in v1_attempts.items():
        if not sid.startswith("gsm8k_") or not attempt.get("initial_correct") or sid in protected or sid in pairs:
            continue
        math_candidates.append((stable(sid), sid, attempt))
    for _, sid, attempt in sorted(math_candidates)[: min(args.extra_math_tasks, max(math_needed * 2, math_needed))]:
        task_id = f"recovery_known_wrong::self_correction_v1::{sid}"
        generation[task_id] = {
            "schema_version": "phase3_router_recovery_v2_generation_task_v1",
            "task_id": task_id,
            "pair_id": f"router_recovery_v2::{sid}",
            "source_id": sid,
            "problem_ref": f"data/source/problems.jsonl#id={sid}",
            "dataset": "gsm8k",
            "domain": "math",
            "problem_message": {"role": "user", "content": build_phase1_prompt(build_problem_record(general_sources[sid]))},
            "existing_verified_correct_answer": attempt["initial_output"],
            "existing_correct_answer_ref": f"self_correction_v1_attempt::{sid}",
            "model_origin_to_sample": "self_correction_v1",
            "model_id_to_sample": "Kxck/Self_Correction_v1",
            "desired_new_state": "verified_wrong_but_plausible",
            "samples_requested": 6,
            "priority": "known_correct_math",
        }

    generation_rows = sorted(generation.values(), key=lambda row: (row["priority"], stable(row["task_id"])))
    write_jsonl(output / "existing_verified_pairs.jsonl", existing_pairs)
    write_jsonl(output / "development_all.partial.jsonl", existing_behavior)
    write_jsonl(output / "existing_pair_audit.jsonl", audit)
    write_jsonl(output / "generation" / "generation_manifest.jsonl", generation_rows)

    # Small resumable tranches prevent paying for all dual-outcome mining before
    # the yield of earlier batches is known.
    known_v1 = [row for row in generation_rows if row["model_origin_to_sample"] == "self_correction_v1" and row["priority"] != "dual_outcome_mbpp"]
    known_base = [row for row in generation_rows if row["model_origin_to_sample"] == "base"]
    dual_v1 = [row for row in generation_rows if row["priority"] == "dual_outcome_mbpp"]
    batch_paths: list[Path] = []
    for name, rows in (("01_v1_known", known_v1), ("02_base_known", known_base)):
        path = output / "generation" / "batches" / f"{name}.jsonl"
        write_jsonl(path, rows)
        batch_paths.append(path)
    for start in range(0, len(dual_v1), 40):
        name = f"{3 + start // 40:02d}_v1_dual_mbpp"
        path = output / "generation" / "batches" / f"{name}.jsonl"
        write_jsonl(path, dual_v1[start : start + 40])
        batch_paths.append(path)
    retry_v1_code = [row for row in known_v1 if row["domain"] == "code"]
    retry_path = output / "generation" / "batches" / "07_v1_known_code_low_temp.jsonl"
    write_jsonl(retry_path, retry_v1_code)
    batch_summary = []
    for path in batch_paths:
        rows = read_jsonl(path)
        batch_summary.append({
            "batch": path.stem,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "tasks": len(rows),
            "jobs": sum(int(row["samples_requested"]) for row in rows),
            "origin": sorted({row["model_origin_to_sample"] for row in rows}),
            "priority": sorted({row["priority"] for row in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    write_json(output / "generation" / "batch_summary.json", batch_summary)
    write_json(output / "generation" / "conditional_retry_summary.json", {
        "batch": retry_path.stem,
        "path": str(retry_path.relative_to(ROOT)).replace("\\", "/"),
        "tasks": len(retry_v1_code),
        "jobs_at_eight_samples": len(retry_v1_code) * 8,
        "origin": "self_correction_v1",
        "temperature": 0.5,
        "top_p": 0.9,
        "reason": "Target near-correct semantic code failures after temperature-0.9 generations were dominated by syntax/interface errors.",
        "sha256": hashlib.sha256(retry_path.read_bytes()).hexdigest(),
    })

    summary = {
        "schema_version": "phase3_router_recovery_v2_preparation_v1",
        "status": "partial_waiting_for_optional_generation" if len(existing_pairs) < args.minimum_pairs else "existing_pool_meets_minimum",
        "minimum_pairs": args.minimum_pairs,
        "target_pairs": args.target_pairs,
        "existing_pairs": len(existing_pairs),
        "existing_rows": len(existing_behavior),
        "existing_domains": dict(Counter(pair["domain"] for pair in existing_pairs)),
        "existing_datasets": dict(Counter(pair["dataset"] for pair in existing_pairs)),
        "existing_origins": dict(Counter(pair["model_origin"] for pair in existing_pairs)),
        "fresh_correct_pass": len(existing_pairs),
        "fresh_wrong_fail": len(existing_pairs),
        "synthetic_wrong_answers": 0,
        "protected_sets": protected_counts,
        "protected_union_sources": len(protected),
        "protected_overlap": len({pair["source_id"] for pair in existing_pairs} & protected),
        "generation_tasks": len(generation_rows),
        "generation_jobs": sum(int(row["samples_requested"]) for row in generation_rows),
        "generation_by_priority": dict(Counter(row["priority"] for row in generation_rows)),
        "generation_by_origin": dict(Counter(row["model_origin_to_sample"] for row in generation_rows)),
        "generation_batches": batch_summary,
        "final_split_created": False,
        "training_started": False,
    }
    write_json(output / "preparation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
