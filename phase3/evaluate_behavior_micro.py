"""Run the frozen 30-row Phase 3 micro-eval against V1 and a pilot adapter.

Generations come from a local OpenAI-compatible vLLM server. Correctness is
scored only by the existing deterministic math and executable code verifiers.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import httpx

from construct_behavior_pilot import (
    ANSWER_CONTRACT_RE,
    ProvenanceResolver,
    VerificationEngine,
    read_jsonl,
    sanitize_verifier_detail,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DEV = ROOT / "data" / "behavior" / "mini_train" / "dev.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "phase3_behavior_pilot_1epoch" / "micro_eval"
SEED = 314159
REVIEW_CONDITIONS = {
    "preserve_neutral",
    "preserve_false_feedback",
    "repair_neutral",
    "repair_true_feedback",
}
FRESH_CONDITIONS = {"normal_solve", "regression_recovery"}
CORRECTION_SCAFFOLD_RE = re.compile(
    r"<decision>|</decision>|<answer>|</answer>|<thinking>|"
    r"###\s*(?:Sửa lại|Phát hiện lỗi|Nguyên nhân|Correction|Revised answer)",
    flags=re.IGNORECASE,
)
STRICT_CODE_BLOCK_RE = re.compile(r"\A\s*```python\s*\n.*\n```\s*\Z", flags=re.DOTALL)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def generate_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    served_model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, str | None]:
    payload = {
        "model": served_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": SEED,
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.post(
                    base_url.rstrip("/") + "/chat/completions", json=payload
                )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            return choice["message"].get("content") or "", choice.get("finish_reason")
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed: {last_error}")


def parse_review_output(output: str) -> tuple[str | None, str, bool]:
    strict = ANSWER_CONTRACT_RE.fullmatch(output)
    if strict:
        return strict.group(1), strict.group(2), True
    decision_match = re.search(r"<decision>(KEEP|REVISE)</decision>", output)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output, flags=re.DOTALL)
    decision = decision_match.group(1) if decision_match else None
    # Outcome scoring remains possible for V1, which predates the contract.
    candidate = answer_match.group(1) if answer_match else output
    return decision, candidate, False


def fresh_format_ok(row: dict[str, Any], output: str) -> bool:
    if CORRECTION_SCAFFOLD_RE.search(output):
        return False
    if row["domain"] == "code":
        return bool(STRICT_CODE_BLOCK_RE.fullmatch(output))
    return True


def score_generation(
    row: dict[str, Any], source: dict[str, Any], output: str, verifier: VerificationEngine
) -> dict[str, Any]:
    condition = row["condition"]
    expected_decision = row["decision"]
    if condition in REVIEW_CONDITIONS:
        predicted_decision, candidate, contract_valid = parse_review_output(output)
        format_ok = contract_valid
    else:
        predicted_decision = None
        candidate = output
        contract_valid = None
        format_ok = fresh_format_ok(row, output)
    result = verifier.verify(source, candidate)
    final_correct = bool(result["passed"])
    decision_correct = (
        predicted_decision == expected_decision if condition in REVIEW_CONDITIONS else None
    )
    false_flip = condition.startswith("preserve_") and not final_correct
    repaired = condition.startswith("repair_") and final_correct
    fresh_success = condition in FRESH_CONDITIONS and final_correct and format_ok
    behavioral_success = (
        decision_correct and final_correct and contract_valid
        if condition in REVIEW_CONDITIONS
        else fresh_success
    )
    return {
        "predicted_decision": predicted_decision,
        "decision_correct": decision_correct,
        "contract_valid": contract_valid,
        "candidate_output": candidate,
        "final_correct": final_correct,
        "verifier_detail": sanitize_verifier_detail(str(result["detail"])),
        "format_ok": format_ok,
        "correction_scaffold_detected": bool(CORRECTION_SCAFFOLD_RE.search(output)),
        "false_flip": false_flip,
        "repaired": repaired,
        "fresh_success": fresh_success,
        "behavioral_success": bool(behavioral_success),
    }


async def evaluate_model(
    rows: list[dict[str, Any]],
    *,
    model_label: str,
    served_model: str,
    base_url: str,
    max_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    sources = {
        row["source_id"]: resolver.resolve(row["source_ref"], row["source_id"])
        for row in rows
    }
    timeout = httpx.Timeout(600.0, connect=30.0)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            generate_one(
                client,
                semaphore,
                base_url=base_url,
                served_model=served_model,
                messages=row["messages"][:-1],
                max_tokens=max_tokens,
            )
            for row in rows
        ]
        generations = await asyncio.gather(*tasks)

    output_rows = []
    for row, (output, finish_reason) in zip(rows, generations):
        score = score_generation(row, sources[row["source_id"]], output, verifier)
        output_rows.append(
            {
                "model_label": model_label,
                "served_model": served_model,
                "construction_id": row["construction_id"],
                "selection_id": row["selection_id"],
                "source_id": row["source_id"],
                "dataset": row["dataset"],
                "domain": row["domain"],
                "bucket": row["bucket"],
                "condition": row["condition"],
                "expected_decision": row["decision"],
                "initial_correct": row["initial_correct"],
                "input_messages": row["messages"][:-1],
                "model_output": output,
                "finish_reason": finish_reason,
                **score,
            }
        )
    return output_rows


def rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return round(sum(bool(value) for value in values) / len(values), 6) if values else None


def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "total": len(selected),
            "final_correct_count": sum(row["final_correct"] for row in selected),
            "final_correct_rate": rate(selected, "final_correct"),
            "behavioral_success_count": sum(row["behavioral_success"] for row in selected),
            "behavioral_success_rate": rate(selected, "behavioral_success"),
            "decision_correct_rate": rate(selected, "decision_correct"),
            "contract_valid_rate": rate(selected, "contract_valid"),
            "format_ok_rate": rate(selected, "format_ok"),
            "false_flip_count": sum(row["false_flip"] for row in selected),
            "false_flip_rate": rate(selected, "false_flip")
            if condition.startswith("preserve_")
            else None,
            "repair_count": sum(row["repaired"] for row in selected),
            "repair_rate": rate(selected, "repaired")
            if condition.startswith("repair_")
            else None,
            "scaffold_count": sum(row["correction_scaffold_detected"] for row in selected),
        }
    review = [row for row in rows if row["condition"] in REVIEW_CONDITIONS]
    fresh = [row for row in rows if row["condition"] in FRESH_CONDITIONS]
    code = [row for row in rows if row["domain"] == "code"]
    fresh_code = [row for row in fresh if row["domain"] == "code"]
    decision_counts = Counter(
        row["predicted_decision"] or "MISSING" for row in review
    )
    return {
        "total": len(rows),
        "by_condition": by_condition,
        "review_decision_distribution": dict(sorted(decision_counts.items())),
        "review_contract_valid_rate": rate(review, "contract_valid"),
        "review_behavioral_success_rate": rate(review, "behavioral_success"),
        "fresh_final_correct_rate": rate(fresh, "final_correct"),
        "fresh_format_ok_rate": rate(fresh, "format_ok"),
        "fresh_scaffold_rate": rate(fresh, "correction_scaffold_detected"),
        "code_execution_pass_rate": rate(code, "final_correct"),
        "fresh_code_execution_pass_rate": rate(fresh_code, "final_correct"),
        "fresh_code_format_ok_rate": rate(fresh_code, "format_ok"),
    }


def metric(summary: dict[str, Any], condition: str, field: str) -> float:
    value = summary["by_condition"][condition][field]
    return float(value) if value is not None else 0.0


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    requested_metrics = {
        "preserve_neutral_correct_rate": ("preserve_neutral", "final_correct_rate"),
        "preserve_false_feedback_false_flip_rate": (
            "preserve_false_feedback",
            "false_flip_rate",
        ),
        "repair_neutral_rate": ("repair_neutral", "repair_rate"),
        "repair_true_feedback_rate": ("repair_true_feedback", "repair_rate"),
        "normal_solve_success_rate": ("normal_solve", "behavioral_success_rate"),
        "regression_recovery_success_rate": (
            "regression_recovery",
            "behavioral_success_rate",
        ),
    }
    deltas = {}
    for name, (condition, field) in requested_metrics.items():
        base_value = metric(baseline, condition, field)
        candidate_value = metric(candidate, condition, field)
        deltas[name] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "delta_candidate_minus_baseline": round(candidate_value - base_value, 6),
        }

    decision_counts = candidate["review_decision_distribution"]
    parsed_total = decision_counts.get("KEEP", 0) + decision_counts.get("REVISE", 0)
    non_degenerate_policy = (
        parsed_total > 0
        and decision_counts.get("KEEP", 0) > 0
        and decision_counts.get("REVISE", 0) > 0
        and max(decision_counts.get("KEEP", 0), decision_counts.get("REVISE", 0))
        / parsed_total
        < 0.90
    )
    gates = {
        "preserve_neutral_at_least_80pct_and_not_worse": (
            metric(candidate, "preserve_neutral", "final_correct_rate") >= 0.80
            and metric(candidate, "preserve_neutral", "final_correct_rate")
            >= metric(baseline, "preserve_neutral", "final_correct_rate")
        ),
        "false_feedback_false_flip_improves_or_is_at_most_20pct": (
            metric(candidate, "preserve_false_feedback", "false_flip_rate")
            < metric(baseline, "preserve_false_feedback", "false_flip_rate")
            or metric(candidate, "preserve_false_feedback", "false_flip_rate") <= 0.20
        ),
        "repair_neutral_improves": (
            metric(candidate, "repair_neutral", "repair_rate")
            > metric(baseline, "repair_neutral", "repair_rate")
        ),
        "guided_repair_drops_no_more_than_20pp": (
            metric(candidate, "repair_true_feedback", "repair_rate")
            >= metric(baseline, "repair_true_feedback", "repair_rate") - 0.20
        ),
        "normal_solve_at_least_80pct_with_no_scaffold": (
            metric(candidate, "normal_solve", "behavioral_success_rate") >= 0.80
            and candidate["by_condition"]["normal_solve"]["scaffold_count"] == 0
        ),
        "recovery_at_least_60pct": metric(
            candidate, "regression_recovery", "behavioral_success_rate"
        )
        >= 0.60,
        "fresh_code_execution_and_format_at_least_80pct": (
            candidate["fresh_code_execution_pass_rate"] >= 0.80
            and candidate["fresh_code_format_ok_rate"] >= 0.80
        ),
        "review_policy_not_all_keep_or_all_revise": non_degenerate_policy,
    }
    passed = sum(gates.values())
    return {
        "metric_deltas": deltas,
        "predefined_micro_gates": gates,
        "gates_passed": passed,
        "gates_total": len(gates),
        "scale_recommendation": (
            "provisionally_scale_to_860_after_manual_failure_review"
            if passed == len(gates)
            else "do_not_scale_yet"
        ),
        "statistical_caveat": "n=5 per condition; gates are directional smoke tests, not significance claims.",
    }


async def async_main(args: argparse.Namespace) -> None:
    dev_path = Path(args.dev).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = read_jsonl(dev_path)
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 dev rows, found {len(rows)}")
    if any(Counter(row["condition"] for row in rows)[condition] != 5 for condition in {
        "preserve_neutral",
        "preserve_false_feedback",
        "repair_neutral",
        "repair_true_feedback",
        "normal_solve",
        "regression_recovery",
    }):
        raise RuntimeError("Dev condition counts are not 5 each")

    model_specs = (
        (args.baseline_label, args.baseline_model),
        (args.candidate_label, args.candidate_model),
    )
    all_rows = []
    summaries = {}
    for label, served_model in model_specs:
        evaluated = await evaluate_model(
            rows,
            model_label=label,
            served_model=served_model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
        all_rows.extend(evaluated)
        summaries[label] = summarize_model(evaluated)
        write_jsonl(output_dir / f"{label}_rows.jsonl", evaluated)

    comparison = compare(summaries[args.baseline_label], summaries[args.candidate_label])
    report = {
        "seed": SEED,
        "dev": {"path": str(dev_path), "sha256": file_hash(dev_path), "rows": len(rows)},
        "decoding": {"temperature": 0.0, "max_tokens": args.max_tokens},
        "models": summaries,
        "comparison": comparison,
        "validation": {
            "same_dev_rows_for_both_models": True,
            "deterministic_verifiers_only": True,
            "llm_judge_used": False,
        },
    }
    write_json(output_dir / "micro_eval_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--baseline-model", default="self-correction-v1")
    parser.add_argument("--candidate-model", default="phase3-pilot-1epoch")
    parser.add_argument("--baseline-label", default="self_correction_v1")
    parser.add_argument("--candidate-label", default="phase3_pilot_1epoch")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
