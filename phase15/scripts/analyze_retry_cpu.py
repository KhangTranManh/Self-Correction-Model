"""Exploratory retry-policy replay; no model calls or training.

Uses already opened Phase 8/15 development data. Decisions use output signals
only; references are used exclusively to score those decisions offline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase14.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.answers import vote  # noqa: E402
from src.data.verifiers.math import _to_sympy  # noqa: E402


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(scores, labels):
    positive = [s for s, y in zip(scores, labels) if y]
    negative = [s for s, y in zip(scores, labels) if not y]
    if not positive or not negative:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def rate(rows, predicate):
    n = sum(bool(predicate(row)) for row in rows)
    return {"count": n, "n": len(rows), "rate": n / len(rows) if rows else None}


def pair_analysis(rows):
    both = [not any(row["correct"][:2]) for row in rows]
    signals = {
        "disagreement": [int(not row["agree"]) for row in rows],
        "either_capped": [int(any(row["caps"][:2])) for row in rows],
        "either_non_numeric": [int(any(not bool(getattr(v, "is_number", False)) for v in row["values"][:2])) for row in rows],
        "max_pair_tokens": [max(row["tokens"][:2]) for row in rows],
    }
    groups = {}
    for name, selected in (("agreement", lambda r: r["agree"]), ("disagreement", lambda r: not r["agree"])):
        group = [row for row in rows if selected(row)]
        groups[name] = {
            "n": len(group),
            "both_wrong": rate(group, lambda r: not any(r["correct"][:2])),
            "first_correct": rate(group, lambda r: r["correct"][0]),
        }
        if group and len(group[0]["correct"]) == 3:
            groups[name]["third_correct_when_both_wrong"] = rate(
                [r for r in group if not any(r["correct"][:2])], lambda r: r["correct"][2]
            )
    return {
        "n": len(rows), "both_wrong": sum(both), "groups": groups,
        "both_wrong_signal_auc": {name: auc(scores, both) for name, scores in signals.items()},
        "both_wrong_signal_auc_within_disagreement": {
            name: auc([score for row, score in zip(rows, scores) if not row["agree"]],
                      [label for row, label in zip(rows, both) if not row["agree"]])
            for name, scores in signals.items() if name != "disagreement"
        },
    }


def replay(rows):
    outcomes = {name: [] for name in ("first", "second_sample", "greedy", "three_mixed_vote", "agreement_gated_three_mixed_vote", "greedy_on_disagreement")}
    costs = {name: {"calls": [], "tokens": []} for name in outcomes}
    for row in rows:
        voted = vote(row["texts"], [0, 1, 2])
        if row["agree"] and voted not in (0, 1):
            raise AssertionError("Two equal first answers must win plurality")
        chosen = {
            "first": 0, "second_sample": 1, "greedy": 2,
            "three_mixed_vote": voted,
            "agreement_gated_three_mixed_vote": 0 if row["agree"] else voted,
            "greedy_on_disagreement": 0 if row["agree"] else 2,
        }
        for name, index in chosen.items():
            outcomes[name].append(int(row["correct"][index]))
            used = [index] if name in ("first", "second_sample", "greedy") else ([0, 1] if row["agree"] and name != "three_mixed_vote" else [0, 1, 2])
            costs[name]["calls"].append(len(used))
            costs[name]["tokens"].append(sum(row["tokens"][i] for i in used))
    rng = random.Random(20261102)
    boot_indices = [[rng.randrange(len(rows)) for _ in rows] for _ in range(2000)]
    policies = {}
    for name, values in outcomes.items():
        differences = [v - b for v, b in zip(values, outcomes["first"])]
        boot = sorted(sum(differences[i] for i in indices) / len(rows) for indices in boot_indices)
        policies[name] = {
            "correct": sum(values), "n": len(rows), "accuracy": sum(values) / len(rows),
            "fixes_vs_first": sum(v and not b for v, b in zip(values, outcomes["first"])),
            "harms_vs_first": sum(b and not v for v, b in zip(values, outcomes["first"])),
            "difference_vs_first": sum(differences) / len(rows),
            "paired_bootstrap_95_ci": [boot[49], boot[1949]],
            "mean_calls": sum(costs[name]["calls"]) / len(rows),
            "mean_generated_tokens": sum(costs[name]["tokens"]) / len(rows),
        }
    if outcomes["three_mixed_vote"] != outcomes["agreement_gated_three_mixed_vote"]:
        raise AssertionError("Agreement gate must preserve three-answer plurality decisions")
    return policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = []
    def load(path):
        inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
        return read(path)

    p8sources = load(ROOT / "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl")
    stages = []
    validations = {}
    base = ROOT / "outputs/phase8_remote_3090/outputs/phase8_first_pass_v1"
    for stage in ("initial", "sample_repeat", "greedy_same_prompt"):
        path = base / stage / "answers.jsonl"
        summary_path = path.with_name("summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inputs.append({"path": summary_path.relative_to(ROOT).as_posix(), "sha256": sha(summary_path)})
        normalized = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if normalized != summary["answers_sha256"]:
            raise AssertionError(f"Hash mismatch: {path}")
        records = load(path)
        mapped = {r["problem_id"]: r for r in records}
        if len(mapped) != len(records) or set(mapped) != {r["id"] for r in p8sources}:
            raise AssertionError("Phase 8 stages must have unique identical source coverage")
        stages.append(mapped)
        validations[stage] = {"hash_matches": True, "rows": len(records), "temperature": summary["settings"]["temperature"]}

    def make_row(source, records):
        values = [parse(r["output"]) for r in records]
        gold = _to_sympy(str(source["reference_answer"]))
        if gold is None:
            raise AssertionError(f"Unparseable reference: {source['id']}")
        return {"id": source["id"], "values": values, "texts": [r["output"] for r in records],
                "correct": [same(v, gold) for v in values],
                "tokens": [r["generated_tokens"] for r in records],
                "caps": [r.get("hit_cap", r.get("hit_token_cap", False)) for r in records],
                "agree": same(values[0], values[1])}

    p8 = [make_row(source, [stage[source["id"]] for stage in stages]) for source in p8sources]
    expected = (283, 309, 307)
    actual = tuple(sum(r["correct"][i] for r in p8) for i in range(3))
    historical_comparison = {"local_v1_correct": actual, "published_v2_correct": expected,
                             "matches": actual == expected}

    sources15 = load(ROOT / "phase15/data/pilot_v1/screen.jsonl")
    audit_path = ROOT / "outputs/phase15_kaggle_pilot_v1/v2/phase15_pilot_v1/audit.jsonl"
    audit = load(audit_path)
    summary15 = json.loads(audit_path.with_name("summary.json").read_text(encoding="utf-8"))
    if sha(audit_path) != summary15["audit_sha256"]:
        raise AssertionError("Phase 15 audit hash mismatch")
    solutions = {(r["problem_id"], r["sample"]): r for r in audit if r["type"] == "screen_solution"}
    p15 = [make_row(s, [solutions[(s["id"], i)] for i in (0, 1)]) for s in sources15]
    pair_lookup = {r["id"]: r for r in p15}
    judgments = [r for r in audit if r["type"] == "screen_judgment"]
    judge_targets = Counter("both_wrong" if not any(pair_lookup[r["problem_id"]]["correct"]) else "one_right" if sum(pair_lookup[r["problem_id"]]["correct"]) == 1 else "both_right" for r in judgments)
    output = {
        "schema": "phase15_retry_cpu_v1", "exploratory": True,
        "inputs": inputs, "phase8_integrity": validations,
        "phase8": {"historical_comparison": historical_comparison,
                   "pairs": pair_analysis(p8), "policies": replay(p8)},
        "phase15": {"pairs": pair_analysis(p15), "judge_label_coverage": dict(judge_targets),
                    "judge_rows": len(judgments), "both_wrong_detection_identifiable": len(judge_targets) > 1 and judge_targets["both_wrong"] > 0},
        "limitations": [
            "Phase 12-14 per-problem raw outputs and hidden states are absent locally.",
            "Phase 8 uses two temperature-0.7 answers and one greedy answer, not standard vote@3.",
            "Only archived Phase 8 v1 outputs are local; the published final report used v2. Replay describes these hashed v1 artifacts only.",
            "Phase 8 and Phase 15 are different models, precision and source pools; do not pool them.",
            "Phase 15 judge inputs were selected using one-right gold labels; both-wrong detection cannot be evaluated on them.",
            "Generated-token costs exclude input/prefill tokens, judge cost and GPU latency.",
            "Data were previously opened; all new replay results are exploratory and cannot validate a fresh learned controller.",
            "Agreement-gated plurality is algebraically identical to full three-answer plurality; savings are structural, not learned.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
