"""Build the Phase 9 "wrong -> correction -> proof" case report (after analysis only).

For every protected source where the first answer (A0) and the first blind
attempt (B1) disagree, the self-check judge wrote a step-by-step check and a
final solution. This report measures, against the deterministic verifier,
whether the judge blamed the right solution and whether its correction is
right, and prints deterministic examples of each outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier, _extract_answer  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase8.scripts.batched import read_jsonl  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402

CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
OUTCOMES = {
    "real_fix": "First answer wrong -> model's correction is right",
    "false_alarm": "First answer right -> model's final answer is wrong",
    "both_wrong_fixed": "Both candidates wrong -> model still reached the right answer",
    "failed_fix": "First answer wrong -> model's final answer still wrong",
    "kept_right": "First answer right -> model's final answer also right",
}


def classify(a0_ok: bool, b1_ok: bool, j_ok: bool) -> str:
    if a0_ok:
        return "kept_right" if j_ok else "false_alarm"
    if j_ok:
        return "real_fix" if b1_ok else "both_wrong_fixed"
    return "failed_fix"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", type=Path, default=ROOT / "outputs/phase9_remote_v100")
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "phase9/docs/CASE_REPORT.md")
    args = parser.parse_args()
    out = args.mirror / "outputs"
    if not (out / "phase9_analysis_v1/report.json").exists():
        raise RuntimeError("Run only after the single protected analysis")
    rows = [json.loads(line) for line in
            (ROOT / "phase9/data/source_pool_v1/protected.jsonl").read_text(encoding="utf-8").splitlines() if line]
    first = {r["problem_id"]: r["output"] for r in read_jsonl(out / "phase9_first_v1/answers.jsonl")}
    verifier = MathVerifier()
    lines = ["# Phase 9 case report — wrong, correction, and proof", "",
             "Protected sources where the first answer (A0) and an independent blind attempt (B1)",
             "disagree. The model saw both solutions, checked them, and wrote a final solution.",
             "**Proof** of right and wrong is the deterministic verifier against the reference",
             "answer; the model's written check is a claim that this report scores.", ""]
    summary, examples = [], {}
    for checkpoint in CHECKPOINTS:
        blind = {(r["problem_id"], r["attempt"]): r["output"]
                 for r in read_jsonl(out / f"phase9_checkpoints_v1/{checkpoint}/blind_answers.jsonl")}
        judge = {r["problem_id"]: r["output"]
                 for r in read_jsonl(out / f"phase9_checkpoints_v1/{checkpoint}/judge_answers.jsonl")}
        counts = {key: 0 for key in OUTCOMES}
        blamed = {"right": 0, "wrong": 0, "neither": 0}
        cases = {key: [] for key in OUTCOMES}
        for row in rows:
            pid = row["id"]
            a0, b1, j = first[pid], blind[(pid, 1)], judge[pid]
            if same(parse(a0), parse(b1)):
                continue
            problem = Problem(id=pid, domain="math", question=row["question"],
                              reference_answer=row["reference_answer"])
            ok = lambda text: bool(verifier.verify(problem, text).passed)
            a0_ok, b1_ok, j_ok = ok(a0), ok(b1), ok(j)
            outcome = classify(a0_ok, b1_ok, j_ok)
            counts[outcome] += 1
            if a0_ok != b1_ok:  # exactly one candidate is right: did the judge side with it?
                right = a0 if a0_ok else b1
                blamed["right" if same(parse(j), parse(right)) else
                       "wrong" if same(parse(j), parse(b1 if a0_ok else a0)) else "neither"] += 1
            cases[outcome].append({"pid": pid, "question": row["question"],
                                   "reference": row["reference_answer"], "a0": a0, "b1": b1,
                                   "judge": j, "a0_ok": a0_ok, "b1_ok": b1_ok, "j_ok": j_ok})
        total = sum(counts.values())
        one_right = sum(blamed.values())
        summary.append((checkpoint, total, counts, blamed, one_right))
        examples[checkpoint] = cases

    lines += ["## Summary per model", "",
              "| Model | Disagreements | Real fixes | False alarms | Failed fixes | Kept right | Both wrong → fixed | Sided with the right solution* |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for checkpoint, total, counts, blamed, one_right in summary:
        share = f"{blamed['right']}/{one_right} ({blamed['right'] / one_right:.0%})" if one_right else "—"
        lines.append(f"| {checkpoint} | {total} | {counts['real_fix']} | {counts['false_alarm']} | "
                     f"{counts['failed_fix']} | {counts['kept_right']} | {counts['both_wrong_fixed']} | {share} |")
    lines += ["", "\\* Among disagreements where exactly one of A0/B1 is right: how often the model's",
              "final answer matches the right one (the rest match the wrong one or neither).", ""]

    best = max(summary, key=lambda item: item[2]["real_fix"] - item[2]["false_alarm"])[0]
    lines += [f"## Examples ({best}, the model with the best fixes-minus-false-alarms)", "",
              "Examples are chosen deterministically by hash order, not by hand.", ""]
    verdict = lambda good: "✅ correct" if good else "❌ wrong"
    for outcome, title in OUTCOMES.items():
        chosen = sorted(examples[best][outcome],
                        key=lambda c: hashlib.sha256(f"phase9_case|{c['pid']}".encode()).hexdigest())
        if not chosen:
            continue
        lines += [f"### {title}", ""]
        for case in chosen[:args.examples]:
            lines += [f"**Problem `{case['pid']}`:** {case['question']}", "",
                      f"**Reference answer (verifier proof):** {case['reference']}", "",
                      f"- Solution A (first answer) → `{_extract_answer(case['a0'])}` — {verdict(case['a0_ok'])}",
                      f"- Solution B (blind attempt) → `{_extract_answer(case['b1'])}` — {verdict(case['b1_ok'])}",
                      f"- Model's final answer after checking → `{_extract_answer(case['judge'])}` — {verdict(case['j_ok'])}",
                      "", "<details><summary>Model's check and corrected solution</summary>", "",
                      "```text", case["judge"].strip(), "```", "", "</details>", "",
                      "<details><summary>Solution A and Solution B</summary>", "",
                      "```text", "--- Solution A ---", case["a0"].strip(), "",
                      "--- Solution B ---", case["b1"].strip(), "```", "", "</details>", ""]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
