"""Open Phase 8 protected labels once after all generations and probe scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402


SOURCE = ROOT / "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl"
FIRST_ROOT = ROOT / "outputs/phase8_first_pass_v2"
THREE_ROOT = ROOT / "outputs/phase8_three_arms_v2"
PROBE_ROOT = ROOT / "outputs/phase8_probe_scores_v2"
OUT = ROOT / "outputs/phase8_analysis_v2"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def checked_answers(folder: Path, name: str, expected: int | None = None) -> list[dict]:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    path = folder / name
    if summary["status"] != "complete" or sha256(path) != summary[f"{name.split('.')[0]}_sha256"]:
        raise RuntimeError(f"Incomplete or changed artifact: {path}")
    rows = read_jsonl(path)
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"Unexpected count: {path}")
    return rows


def ci_difference(left: np.ndarray, right: np.ndarray, seed: int) -> list[float]:
    if len(left) != len(right) or len(left) == 0:
        raise RuntimeError("Paired CI needs equal nonempty vectors")
    rng = np.random.default_rng(seed)
    delta = left.astype(float) - right.astype(float)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    values = delta[indices].mean(axis=1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_p(left: np.ndarray, right: np.ndarray) -> float:
    better = int(np.sum(left & ~right))
    worse = int(np.sum(~left & right))
    discordant = better + worse
    return float(binomtest(min(better, worse), discordant, 0.5).pvalue) if discordant else 1.0


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    result = {}
    maximum = 0.0
    for rank, key in enumerate(ordered):
        maximum = max(maximum, min(1.0, raw[key] * (len(ordered) - rank)))
        result[key] = maximum
    return result


def main() -> None:
    if OUT.exists():
        raise RuntimeError("Refusing a second protected opening or report overwrite")
    source_report = json.loads(SOURCE.with_name("candidate_report.json").read_text(encoding="utf-8"))
    if sha256(SOURCE) != source_report["candidate_manifest_sha256"]:
        raise RuntimeError("Source manifest changed")
    first = {}
    for stage in ("initial", "sample_repeat", "greedy_same_prompt"):
        first[stage] = checked_answers(FIRST_ROOT / stage, "answers.jsonl", 400)
    three, scores = {}, {}
    for checkpoint in CHECKPOINTS:
        three[checkpoint] = checked_answers(THREE_ROOT / checkpoint, "answers.jsonl")
        scores[checkpoint] = checked_answers(PROBE_ROOT / checkpoint, "scores.jsonl", 400)
    donors_path = ROOT / "phase8/data/distractors_v2/assignments.jsonl"
    donors_report = json.loads(donors_path.with_name("report.json").read_text(encoding="utf-8"))
    if sha256(donors_path) != donors_report["assignments_sha256"]:
        raise RuntimeError("Frozen distractor map changed")
    donors = {row["problem_id"]: row for row in read_jsonl(donors_path)}

    # This is the first read of protected gold answers in the Phase 8 analyzer.
    sources = read_jsonl(SOURCE)
    ids = [row["id"] for row in sources]
    if len(ids) != 400 or len(set(ids)) != 400:
        raise RuntimeError("Protected source count/uniqueness mismatch")
    verifier = MathVerifier()
    problems = {row["id"]: Problem(id=row["id"], domain="math",
                question=row["question"], reference_answer=row["reference_answer"])
                for row in sources}

    def correct(pid: str, answer: str) -> bool:
        return bool(verifier.verify(problems[pid], answer).passed)

    first_by_stage = {}
    first_correct = {}
    for stage, rows in first.items():
        by_id = {row["problem_id"]: row["output"] for row in rows}
        if set(by_id) != set(ids):
            raise RuntimeError(f"Missing first-pass IDs: {stage}")
        first_by_stage[stage] = by_id
        first_correct[stage] = np.array([correct(pid, by_id[pid]) for pid in ids], dtype=bool)
    baseline = first_correct["initial"]
    report = {
        "schema_version": "phase8_protected_analysis_v2",
        "source_manifest_sha256": sha256(SOURCE), "protected_rows": 400,
        "natural_initial_correct": int(baseline.sum()),
        "natural_initial_wrong": int((~baseline).sum()),
        "first_pass_controls": {stage: {"correct": int(values.sum()),
                                         "accuracy": float(values.mean()),
                                         "paired_delta_vs_initial": float(values.mean() - baseline.mean()),
                                         "paired_delta_ci": ci_difference(values, baseline, 20260925 + j)}
                                for j, (stage, values) in enumerate(first_correct.items())},
        "checkpoints": {},
    }
    verdict_rows = []
    pipeline_raw_p = {}
    mechanism_raw_p = {name: {} for name in ("own_vs_blind", "distractor_vs_blind", "own_vs_distractor")}
    for model_index, checkpoint in enumerate(CHECKPOINTS):
        arm_output = {(row["problem_id"], row["arm"]): row["output"] for row in three[checkpoint]}
        expected = {(pid, arm) for pid in ids for arm in ("blind", "own_visible")}
        expected |= {(pid, "distractor") for pid in ids if donors[pid]["donor_problem_id"]}
        if len(arm_output) != len(three[checkpoint]) or set(arm_output) != expected:
            raise RuntimeError(f"Missing or duplicate three-arm task: {checkpoint}")
        score_by_id = {row["problem_id"]: row for row in scores[checkpoint]}
        if set(score_by_id) != set(ids):
            raise RuntimeError(f"Missing probe scores: {checkpoint}")
        flags = np.array([bool(score_by_id[pid]["flag"]) for pid in ids], dtype=bool)
        if any(bool(score_by_id[pid]["flag"]) !=
               (float(score_by_id[pid]["probability_wrong"]) >= 0.5) for pid in ids):
            raise RuntimeError("Probe flag disagrees with locked threshold")
        blind = np.array([correct(pid, arm_output[(pid, "blind")]) for pid in ids], dtype=bool)
        own = np.array([correct(pid, arm_output[(pid, "own_visible")]) for pid in ids], dtype=bool)
        routed = np.where(flags, blind, baseline)
        random_ids = set(sorted(ids, key=lambda pid: hashlib.sha256(
            f"phase8_random_route_v1|{checkpoint}|{pid}".encode()).hexdigest())[:int(flags.sum())])
        random_route = np.array([pid in random_ids for pid in ids], dtype=bool)
        random_final = np.where(random_route, blind, baseline)
        matched = np.array([donors[pid]["donor_problem_id"] is not None for pid in ids], dtype=bool)
        distractor = np.array([correct(pid, arm_output[(pid, "distractor")])
                               for pid in ids if donors[pid]["donor_problem_id"]], dtype=bool)
        wrong = ~baseline
        wrong_matched = wrong & matched
        own_wrong = own[wrong_matched]
        blind_wrong = blind[wrong_matched]
        dist_wrong = distractor[wrong[matched]]
        if len(dist_wrong) != len(own_wrong):
            raise RuntimeError("Distractor matching order mismatch")
        contrasts = {}
        for name, left, right in (
                ("own_vs_blind", own[wrong], blind[wrong]),
                ("distractor_vs_blind", dist_wrong, blind_wrong),
                ("own_vs_distractor", own_wrong, dist_wrong)):
            contrasts[name] = {"n": len(left), "difference": float(left.mean() - right.mean()),
                               "ci": ci_difference(left, right, 20260930 + model_index),
                               "p_raw": paired_p(left, right)}
            mechanism_raw_p[name][checkpoint] = contrasts[name]["p_raw"]
        pipeline_raw_p[checkpoint] = paired_p(routed, baseline)
        report["checkpoints"][checkpoint] = {
            "probe": {"flagged": int(flags.sum()),
                      "wrong_recall": float(flags[wrong].mean()),
                      "precision": float(wrong[flags].mean()) if flags.any() else None,
                      "correct_preservation": float((~flags[baseline]).mean()),
                      "wrong_flagged": int((flags & wrong).sum()),
                      "correct_flagged": int((flags & baseline).sum())},
            "arms": {"blind_wrong_to_correct": int((blind & wrong).sum()),
                     "blind_correct_to_wrong": int((~blind & baseline).sum()),
                     "own_wrong_to_correct": int((own & wrong).sum()),
                     "own_correct_to_wrong": int((~own & baseline).sum()),
                     "distractor_matched": int(matched.sum()),
                     "distractor_wrong_matched": int(wrong_matched.sum()),
                     "wrong_case_contrasts": contrasts},
            "pipeline": {"initial_accuracy": float(baseline.mean()),
                         "routed_accuracy": float(routed.mean()),
                         "blind_all_accuracy": float(blind.mean()),
                         "random_equal_route_accuracy": float(random_final.mean()),
                         "routed_minus_keep": float(routed.mean() - baseline.mean()),
                         "routed_minus_keep_ci": ci_difference(routed, baseline, 20261001 + model_index),
                         "routed_wrong_to_correct": int((routed & wrong).sum()),
                         "routed_correct_to_wrong": int((~routed & baseline).sum()),
                         "p_raw": pipeline_raw_p[checkpoint]},
        }
        for index, pid in enumerate(ids):
            verdict_rows.append({
                "checkpoint": checkpoint, "problem_id": pid,
                "initial_correct": bool(baseline[index]),
                "sample_repeat_correct": bool(first_correct["sample_repeat"][index]),
                "greedy_same_prompt_correct": bool(first_correct["greedy_same_prompt"][index]),
                "probe_probability_wrong": float(score_by_id[pid]["probability_wrong"]),
                "probe_flag": bool(flags[index]), "blind_correct": bool(blind[index]),
                "own_visible_correct": bool(own[index]),
                "distractor_correct": (correct(pid, arm_output[(pid, "distractor")])
                                       if matched[index] else None),
                "routed_correct": bool(routed[index]),
                "random_route": bool(random_route[index]),
            })
    pipeline_adjusted = holm(pipeline_raw_p)
    mechanism_adjusted = {name: holm(values) for name, values in mechanism_raw_p.items()}
    for checkpoint in CHECKPOINTS:
        report["checkpoints"][checkpoint]["pipeline"]["p_holm"] = pipeline_adjusted[checkpoint]
        for name, adjusted in mechanism_adjusted.items():
            report["checkpoints"][checkpoint]["arms"]["wrong_case_contrasts"][name]["p_holm"] = adjusted[checkpoint]
    OUT.mkdir(parents=True)
    verdict_path = OUT / "verdicts.jsonl"
    verdict_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in verdict_rows),
                            encoding="utf-8", newline="\n")
    report["verdict_rows_sha256"] = sha256(verdict_path)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
