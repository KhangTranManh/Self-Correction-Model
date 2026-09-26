"""Evaluate complete Phase 7 paired outputs with the frozen math verifier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem
from src.data.verifiers.math import MathVerifier, _extract_answer
from phase7.scripts.collect_initials_4bit import read_jsonl, sha256, write_atomic


CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
ARMS = ("blind_resolve", "answer_visible_resolve")


def exact_p(discordant_blind_only: int, discordant_visible_only: int) -> float:
    total = discordant_blind_only + discordant_visible_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(discordant_blind_only,
                                                      discordant_visible_only) + 1))
    return min(1.0, 2.0 * tail / 2 ** total)


def bootstrap_interval(differences: list[int], seed: int, repeats: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(differences)
    if n == 0:
        return [0.0, 0.0]
    estimates = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(repeats))
    return [estimates[int(0.025 * repeats)], estimates[int(0.975 * repeats)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "protected"), required=True)
    args = parser.parse_args()
    split_dir = ROOT / "phase7/data/split_v1"
    split = json.loads((split_dir / "split_report.json").read_text(encoding="utf-8"))
    ids_path = split_dir / f"{args.split}_ids.json"
    if sha256(ids_path) != split[f"{args.split}_ids_sha256"]:
        raise ValueError("Split IDs changed")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    initials_path = ROOT / "outputs/phase7_initials_v1/initial_rollouts.jsonl"
    if sha256(initials_path) != split["initial_rollouts_sha256"]:
        raise ValueError("Initial answers changed")
    initials = {row["problem_id"]: row for row in read_jsonl(initials_path)}
    source_path = ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl"
    sources = {row["id"]: row for row in read_jsonl(source_path)}
    verifier = MathVerifier()
    results = {}
    p_values = {}
    for checkpoint in CHECKPOINTS:
        folder = ROOT / "outputs/phase7_paired_v1" / args.split / checkpoint
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        audit_path = folder / "paired_outputs.audit.jsonl"
        output_path = folder / "paired_outputs.jsonl"
        if (summary["status"] != "complete" or
                sha256(audit_path) != summary["audit_sha256"] or
                sha256(output_path) != summary["outputs_sha256"]):
            raise ValueError(f"Incomplete or changed paired outputs: {checkpoint}")
        rows = read_jsonl(output_path)
        if len(rows) != 2 * len(ids):
            raise ValueError("Paired output count mismatch")
        by_id = {}
        for index, row in enumerate(rows):
            pid, arm = ids[index // 2], ARMS[index % 2]
            if row["problem_id"] != pid or row["arm"] != arm:
                raise ValueError("Paired output order mismatch")
            source = sources[pid]
            problem = Problem(id=pid, domain="math", question=source["question"],
                              reference_answer=source["reference_answer"])
            verdict = verifier.verify(problem, row["output"])
            by_id.setdefault(pid, {})[arm] = {
                "correct": bool(verdict.passed),
                "parsed_final": _extract_answer(row["output"]),
                "hit_token_cap": bool(row["hit_token_cap"]),
            }
        wrong_ids = [pid for pid in ids if not initials[pid]["initial_correct"]]
        correct_ids = [pid for pid in ids if initials[pid]["initial_correct"]]
        if len(wrong_ids) != len(correct_ids):
            raise ValueError("Selected split is not balanced")
        wrong_pairs = [(by_id[pid][ARMS[0]]["correct"],
                        by_id[pid][ARMS[1]]["correct"]) for pid in wrong_ids]
        table = {
            "both_correct": sum(a and b for a, b in wrong_pairs),
            "blind_only": sum(a and not b for a, b in wrong_pairs),
            "visible_only": sum(not a and b for a, b in wrong_pairs),
            "both_wrong": sum(not a and not b for a, b in wrong_pairs),
        }
        p = exact_p(table["blind_only"], table["visible_only"])
        p_values[checkpoint] = p
        differences = [int(a) - int(b) for a, b in wrong_pairs]
        arm_stats = {}
        for arm in ARMS:
            repaired = sum(by_id[pid][arm]["correct"] for pid in wrong_ids)
            harmed = sum(not by_id[pid][arm]["correct"] for pid in correct_ids)
            arm_stats[arm] = {
                "wrong_to_correct": repaired, "wrong_count": len(wrong_ids),
                "wrong_to_correct_rate": repaired / len(wrong_ids),
                "correct_to_wrong": harmed, "correct_count": len(correct_ids),
                "correct_to_wrong_rate": harmed / len(correct_ids),
                "oracle_wrong_only_final_accuracy": (len(correct_ids) + repaired) / len(ids),
                "truncated": sum(by_id[pid][arm]["hit_token_cap"] for pid in ids),
                "unparsed_final": sum(by_id[pid][arm]["parsed_final"] is None for pid in ids),
                "same_parsed_final_as_old": sum(
                    by_id[pid][arm]["parsed_final"] == _extract_answer(initials[pid]["initial_output"])
                    for pid in ids),
            }
        results[checkpoint] = {
            "wrong_paired_table": table,
            "blind_minus_visible_wrong_to_correct": sum(differences) / len(differences),
            "bootstrap_95_ci": bootstrap_interval(differences, 20260924),
            "exact_p_unadjusted": p,
            "arms": arm_stats,
            "same_parsed_final_between_arms": sum(
                by_id[pid][ARMS[0]]["parsed_final"] == by_id[pid][ARMS[1]]["parsed_final"]
                for pid in ids),
        }
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted = {}
    running = 0.0
    for index, checkpoint in enumerate(ordered):
        running = max(running, min(1.0, p_values[checkpoint] * (len(ordered) - index)))
        adjusted[checkpoint] = running
    for checkpoint, value in adjusted.items():
        results[checkpoint]["holm_adjusted_p"] = value
    report = {
        "schema_version": "phase7_paired_analysis_v1", "split": args.split,
        "source_count": len(ids), "initial_wrong": len(wrong_ids),
        "initial_correct": len(correct_ids), "results": results,
        "nonoracle_probe_route": "unavailable_frozen_phase5_probe_artifact_missing",
        "note": "Oracle-known-wrong values are diagnostics; the model saw no correctness label.",
    }
    output = ROOT / "outputs/phase7_paired_v1" / args.split / "analysis.json"
    write_atomic(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
