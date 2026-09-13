"""Evaluate an inference-only decision-router plus conditional-repair system.

The initial answers are frozen Self_Correction_v1 generations.  The unchanged
Decision-Only LoRA emits the scored KEEP/REVISE tag, and the unchanged V1 model
generates one neutral repair for every row as the always-repair control.  The
selective system reuses that paired deterministic repair only for REVISE rows.
No verifier outcome or reference answer is included in any model prompt.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

import httpx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
PHASE1 = ROOT.parent / "phase1"
if str(PHASE1) not in sys.path:
    sys.path.insert(0, str(PHASE1))

from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.code import CodeVerifier  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
from apps_verifier import verify as verify_apps  # noqa: E402


SEED = 314159
DECISION_RE = re.compile(r"<decision>(KEEP|REVISE)</decision>")
CORRECTION_RE = re.compile(
    r"###\s*(?:S\u1eeda\s+l\u1ea1i|Corrected\s+answer|Correction)"
    r"\s*:?[ \t]*(?:\r?\n)?(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)
NEUTRAL_REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."
UNSEEN_REVIEW = (
    "Assess the preceding response without assuming it is right or wrong. "
    "Return whether it should be kept or revised."
)
NEUTRAL_REPAIR = (
    "Reconsider your previous answer carefully. If it contains an error, "
    "correct it and provide the best final answer."
)
RATIONALE_PROMPTS = (
    "Audit-only follow-up. Explain the concrete evidence in the problem and previous "
    "answer that supports the decision already given. This is not scored and must not "
    "change the decision.",
    "The scored classification is finished. Do not output a decision tag. Write at "
    "least two complete sentences beginning with 'Evidence:' and cite a calculation, "
    "test case, or logical step that justifies the recorded decision.",
    "Provide explanatory audit prose, not KEEP, REVISE, or XML. Identify one specific "
    "piece of evidence in the previous answer and explain in at least two sentences why "
    "it supports the already-recorded classification.",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value for paired binary outcomes."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 6)


def substantive(text: str) -> bool:
    cleaned = text.strip()
    return len(cleaned) >= 40 and len(cleaned.split()) >= 8 and not DECISION_RE.fullmatch(cleaned)


def repair_candidate(text: str) -> tuple[str, bool]:
    match = CORRECTION_RE.search(text)
    return (match.group(1).strip(), True) if match else (text.strip(), False)


def problem_from_row(row: dict[str, Any]) -> Problem | None:
    verifier = row["verifier"]
    if verifier["type"] == "apps":
        return None
    return Problem(
        id=row["source_id"],
        domain=row["domain"],
        question=row["question"],
        reference_answer=verifier.get("reference_answer"),
        entry_point=verifier.get("entry_point"),
        tests=verifier.get("tests", []),
    )


class ObjectiveVerifier:
    def __init__(self) -> None:
        self.math = MathVerifier()
        self.code = CodeVerifier(timeout_seconds=8, memory_limit_mb=512)

    def verify(self, row: dict[str, Any], answer: str) -> dict[str, Any]:
        if row["verifier"]["type"] == "apps":
            return verify_apps(
                answer,
                row["verifier"]["input_output"],
                per_test_timeout_seconds=5,
                memory_limit_mb=512,
            )
        problem = problem_from_row(row)
        result = (self.math if row["domain"] == "math" else self.code).verify(problem, answer)
        return {"passed": result.passed, "detail": result.detail}


async def chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post("chat/completions", json=payload)
                latency = time.perf_counter() - started
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            return {
                "raw": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "latency_seconds": latency,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed: {last_error}")


async def generate_cached(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    rows: list[dict[str, Any]],
    cache_path: Path,
    model: str,
    max_tokens: int,
    messages_for: Callable[[dict[str, Any]], list[dict[str, str]]],
    batch_size: int,
    seed_offset: int,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for record in read_jsonl(cache_path):
            cache[record["source_id"]] = record
    missing: list[tuple[int, dict[str, Any], list[dict[str, str]], str]] = []
    for index, row in enumerate(rows):
        messages = messages_for(row)
        prompt_hash = digest(messages)
        existing = cache.get(row["source_id"])
        if existing:
            if existing["model"] != model or existing["prompt_sha256"] != prompt_hash:
                raise RuntimeError(f"Stale cache for {row['source_id']} at {cache_path}")
        else:
            missing.append((index, row, messages, prompt_hash))
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        outputs = await asyncio.gather(*[
            chat(
                client,
                semaphore,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                seed=SEED + seed_offset + index,
            )
            for index, _, messages, _ in chunk
        ])
        for (_, row, _, prompt_hash), output in zip(chunk, outputs):
            cache[row["source_id"]] = {
                "source_id": row["source_id"],
                "model": model,
                "prompt_sha256": prompt_hash,
                **output,
            }
        write_jsonl(cache_path, [cache[row["source_id"]] for row in rows if row["source_id"] in cache])
        print(f"[{cache_path.stem}] {min(start + len(chunk), len(missing))}/{len(missing)} new", flush=True)
    return cache


async def capture_rationales(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    cache_path: Path,
    model: str,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    cached = {row["source_id"]: row for row in read_jsonl(cache_path)} if cache_path.exists() else {}
    missing = [row for row in rows if row["source_id"] not in cached]

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        prefix = [
            {"role": "user", "content": row["task_prompt"]},
            {"role": "assistant", "content": row["initial_answer"]},
            {"role": "user", "content": NEUTRAL_REVIEW},
            {"role": "assistant", "content": decisions[row["source_id"]]["raw"]},
        ]
        attempts = []
        messages = prefix
        for prompt_index, prompt in enumerate(RATIONALE_PROMPTS):
            request_messages = messages + [{"role": "user", "content": prompt}]
            result = await chat(
                client,
                semaphore,
                model=model,
                messages=request_messages,
                max_tokens=256,
                seed=SEED + 300000 + prompt_index,
            )
            attempts.append({"prompt": prompt, "substantive": substantive(result["raw"]), **result})
            if substantive(result["raw"]):
                return {
                    "source_id": row["source_id"],
                    "model": model,
                    "raw": result["raw"],
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "substantive": True,
                }
            messages = request_messages + [{"role": "assistant", "content": result["raw"]}]
        raise RuntimeError(f"No substantive rationale for {row['source_id']}")

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        results = await asyncio.gather(*[one(row) for row in chunk])
        for result in results:
            cached[result["source_id"]] = result
        write_jsonl(cache_path, [cached[row["source_id"]] for row in rows if row["source_id"] in cached])
        print(f"[decision_rationales] {min(start + len(chunk), len(missing))}/{len(missing)} new", flush=True)
    return cached


def transition(initial: bool, decision: str, repair_correct: bool | None) -> str:
    if initial and decision == "KEEP":
        return "TRUE_KEEP"
    if initial and decision == "REVISE":
        return "UNNECESSARY_BUT_SAFE_REVISION" if repair_correct else "HARMFUL_FALSE_REVISION"
    if not initial and decision == "KEEP":
        return "MISSED_ERROR"
    if not initial and decision == "REVISE":
        return "SUCCESSFUL_RECOVERY" if repair_correct else "FAILED_REPAIR"
    return "INVALID_DECISION"


def summarize_selective(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = [row for row in rows if row["initial_correct"]]
    wrong = [row for row in rows if not row["initial_correct"]]
    revise = [row for row in rows if row["decision"] == "REVISE"]
    keep = [row for row in rows if row["decision"] == "KEEP"]
    valid = [row for row in rows if row["decision"] != "INVALID"]
    true_keep = sum(row["initial_correct"] and row["decision"] == "KEEP" for row in rows)
    true_revise = sum(not row["initial_correct"] and row["decision"] == "REVISE" for row in rows)
    false_revise = sum(row["initial_correct"] and row["decision"] == "REVISE" for row in rows)
    missed = sum(not row["initial_correct"] and row["decision"] == "KEEP" for row in rows)
    recovered = sum(not row["initial_correct"] and row["final_correct"] for row in rows)
    degraded = sum(row["initial_correct"] and not row["final_correct"] for row in rows)
    return {
        "total": total,
        "initial_accuracy": safe_rate(sum(row["initial_correct"] for row in rows), total),
        "final_accuracy": safe_rate(sum(row["final_correct"] for row in rows), total),
        "absolute_accuracy_delta": safe_rate(
            sum(row["final_correct"] for row in rows) - sum(row["initial_correct"] for row in rows), total
        ),
        "decision_accuracy": safe_rate(sum(row["decision_correct"] for row in rows), total),
        "exact_decision_contract_rate": safe_rate(len(valid), total),
        "keep_recall": safe_rate(true_keep, len(correct)),
        "revise_recall": safe_rate(true_revise, len(wrong)),
        "keep_precision": safe_rate(true_keep, len(keep)),
        "revise_precision": safe_rate(true_revise, len(revise)),
        "balanced_decision_accuracy": safe_rate(
            safe_rate(true_keep, len(correct)) + safe_rate(true_revise, len(wrong)), 2
        ),
        "false_revision_count": false_revise,
        "false_revision_rate": safe_rate(false_revise, len(correct)),
        "missed_error_count": missed,
        "missed_error_rate": safe_rate(missed, len(wrong)),
        "repair_call_count": len(revise),
        "repair_call_rate": safe_rate(len(revise), total),
        "repair_success_conditioned_on_revise": safe_rate(sum(row["final_correct"] for row in revise), len(revise)),
        "wrong_revise_repair_success_rate": safe_rate(
            sum(not row["initial_correct"] and row["decision"] == "REVISE" and row["final_correct"] for row in rows),
            true_revise,
        ),
        "wrong_to_correct_count": recovered,
        "wrong_to_correct_recovery_rate": safe_rate(recovered, len(wrong)),
        "correct_to_wrong_count": degraded,
        "correct_to_wrong_degradation_rate": safe_rate(degraded, len(correct)),
        "correct_preservation_rate": safe_rate(len(correct) - degraded, len(correct)),
        "decision_distribution": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "transition_counts": dict(sorted(Counter(row["transition"] for row in rows).items())),
    }


def summarize_always(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in rows if row["initial_correct"]]
    wrong = [row for row in rows if not row["initial_correct"]]
    return {
        "total": len(rows),
        "initial_accuracy": safe_rate(sum(row["initial_correct"] for row in rows), len(rows)),
        "final_accuracy": safe_rate(sum(row["final_correct"] for row in rows), len(rows)),
        "absolute_accuracy_delta": safe_rate(
            sum(row["final_correct"] for row in rows) - sum(row["initial_correct"] for row in rows), len(rows)
        ),
        "repair_call_count": len(rows),
        "repair_call_rate": 1.0,
        "wrong_to_correct_count": sum(not row["initial_correct"] and row["final_correct"] for row in rows),
        "wrong_to_correct_recovery_rate": safe_rate(
            sum(not row["initial_correct"] and row["final_correct"] for row in rows), len(wrong)
        ),
        "correct_to_wrong_count": sum(row["initial_correct"] and not row["final_correct"] for row in rows),
        "correct_to_wrong_degradation_rate": safe_rate(
            sum(row["initial_correct"] and not row["final_correct"] for row in rows), len(correct)
        ),
        "correct_preservation_rate": safe_rate(
            sum(row["initial_correct"] and row["final_correct"] for row in rows), len(correct)
        ),
    }


def slice_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = {}
    for label in sorted({row[key] for row in rows}):
        subset = [row for row in rows if row[key] == label]
        result[label] = {
            "total": len(subset),
            "initial_accuracy": safe_rate(sum(row["initial_correct"] for row in subset), len(subset)),
            "final_accuracy": safe_rate(sum(row["final_correct"] for row in subset), len(subset)),
            "decision_accuracy": safe_rate(sum(row["decision_correct"] for row in subset), len(subset)),
            "repair_call_rate": safe_rate(sum(row["repair_called"] for row in subset), len(subset)),
        }
    return result


def slice_always(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = {}
    for label in sorted({row[key] for row in rows}):
        subset = [row for row in rows if row[key] == label]
        result[label] = {
            "total": len(subset),
            "initial_accuracy": safe_rate(sum(row["initial_correct"] for row in subset), len(subset)),
            "final_accuracy": safe_rate(sum(row["final_correct"] for row in subset), len(subset)),
            "repair_call_rate": 1.0,
        }
    return result


def token_summary(
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    repairs: dict[str, dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    initial_prompt = []
    initial_completion = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["task_prompt"]}], tokenize=False, add_generation_prompt=True
        )
        initial_prompt.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))
        initial_completion.append(len(tokenizer(row["initial_answer"], add_special_tokens=False)["input_ids"]))
    v1_only = [a + b for a, b in zip(initial_prompt, initial_completion)]
    always = [
        v1_only[i] + repairs[row["source_id"]]["total_tokens"] for i, row in enumerate(rows)
    ]
    selective = [
        v1_only[i]
        + decisions[row["source_id"]]["total_tokens"]
        + (repairs[row["source_id"]]["total_tokens"] if row["decision"] == "REVISE" else 0)
        for i, row in enumerate(rows)
    ]
    completion_only = {
        "v1_only": safe_rate(sum(initial_completion), len(rows)),
        "always_repair": safe_rate(
            sum(initial_completion[i] + repairs[row["source_id"]]["completion_tokens"] for i, row in enumerate(rows)),
            len(rows),
        ),
        "selective_repair": safe_rate(
            sum(
                initial_completion[i]
                + decisions[row["source_id"]]["completion_tokens"]
                + (repairs[row["source_id"]]["completion_tokens"] if row["decision"] == "REVISE" else 0)
                for i, row in enumerate(rows)
            ),
            len(rows),
        ),
    }
    always_generated = completion_only["always_repair"]
    selective_generated = completion_only["selective_repair"]
    decision_latencies = [decisions[row["source_id"]]["latency_seconds"] for row in rows]
    repair_latencies = [repairs[row["source_id"]]["latency_seconds"] for row in rows]
    selective_repair_latencies = [
        repairs[row["source_id"]]["latency_seconds"]
        for row in rows
        if row["decision"] == "REVISE"
    ]
    return {
        "accounting_scope": "operational prompts plus generated tokens; audit rationale and unseen-template control excluded",
        "initial_solve_is_cached": True,
        "average_initial_prompt_tokens": safe_rate(sum(initial_prompt), len(rows)),
        "average_initial_solve_completion_tokens": safe_rate(sum(initial_completion), len(rows)),
        "average_decision_stage_completion_tokens": safe_rate(
            sum(decisions[row["source_id"]]["completion_tokens"] for row in rows), len(rows)
        ),
        "average_repair_completion_tokens_when_called": safe_rate(
            sum(repairs[row["source_id"]]["completion_tokens"] for row in rows if row["decision"] == "REVISE"),
            sum(row["decision"] == "REVISE" for row in rows),
        ),
        "average_total_tokens_per_request": {
            "v1_only": safe_rate(sum(v1_only), len(rows)),
            "always_repair": safe_rate(sum(always), len(rows)),
            "selective_repair": safe_rate(sum(selective), len(rows)),
        },
        "average_generated_tokens_per_request": completion_only,
        "selective_generated_token_reduction_vs_always": safe_rate(
            always_generated - selective_generated, always_generated
        ),
        "selective_tokens_saved_vs_always_per_request": safe_rate(sum(always) - sum(selective), len(rows)),
        "selective_token_reduction_vs_always": safe_rate(sum(always) - sum(selective), sum(always)),
        "latency_seconds": {
            "initial_solve": None,
            "initial_solve_note": "unavailable because frozen V1 generations were reused",
            "average_decision_request": safe_rate(sum(decision_latencies), len(decision_latencies)),
            "average_repair_request_all": safe_rate(sum(repair_latencies), len(repair_latencies)),
            "average_repair_request_selective_calls": safe_rate(
                sum(selective_repair_latencies), len(selective_repair_latencies)
            ),
            "modeled_average_additional_always": safe_rate(sum(repair_latencies), len(rows)),
            "modeled_average_additional_selective": safe_rate(
                sum(decision_latencies) + sum(selective_repair_latencies), len(rows)
            ),
            "note": "Per-request service latency; modeled values exclude cached initial solve and batch wall time.",
        },
    }


def render_report(
    summary: dict[str, Any], always_summary: dict[str, Any], comparison: dict[str, Any], transition_matrix: dict[str, Any]
) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    metrics = summary["metrics"]
    always = always_summary["metrics"]
    tokens = comparison["token_accounting"]
    lines = [
        "# Phase 3 Two-Stage Selective Repair",
        "",
        "## Conclusion",
        "",
        f"`{comparison['conclusion']}`",
        "",
        "No weights were merged or trained. Frozen Self_Correction_v1 initial answers",
        "are routed by the unchanged Decision-Only adapter; the unchanged V1 checkpoint",
        "performs neutral repair only for REVISE decisions.",
        "",
        "## Evaluation set",
        "",
        f"- {summary['evaluation_rows']} rows: 100 initially correct and 100 initially wrong by construction.",
        "- 100 math and 100 code; GSM8K 60, MBPP 40, APPS 40, SVAMP 40, HumanEval 20.",
        "- No verifier result, expected decision, or ground truth was included in inference prompts.",
        "- Because the set is outcome-balanced, 50% initial accuracy is fixed by design and is not",
        "  an estimate of natural deployment prevalence.",
        "",
        "## Main metrics",
        "",
        "| Metric | Selective repair | Always repair |",
        "|---|---:|---:|",
        f"| Initial V1 accuracy | {pct(metrics['initial_accuracy'])} | {pct(always['initial_accuracy'])} |",
        f"| Final accuracy | {pct(metrics['final_accuracy'])} | {pct(always['final_accuracy'])} |",
        f"| Accuracy delta | {pct(metrics['absolute_accuracy_delta'])} | {pct(always['absolute_accuracy_delta'])} |",
        f"| Correct preservation | {pct(metrics['correct_preservation_rate'])} | {pct(always['correct_preservation_rate'])} |",
        f"| Wrong-to-correct recovery | {pct(metrics['wrong_to_correct_recovery_rate'])} | {pct(always['wrong_to_correct_recovery_rate'])} |",
        f"| Correct-to-wrong degradation | {pct(metrics['correct_to_wrong_degradation_rate'])} | {pct(always['correct_to_wrong_degradation_rate'])} |",
        f"| Repair-call rate | {pct(metrics['repair_call_rate'])} | 100.0% |",
        "",
        "## Router",
        "",
        f"- Exact contract: {pct(metrics['exact_decision_contract_rate'])}.",
        f"- Decision accuracy / balanced accuracy: {pct(metrics['decision_accuracy'])} / {pct(metrics['balanced_decision_accuracy'])}.",
        f"- KEEP recall: {pct(metrics['keep_recall'])}; REVISE recall: {pct(metrics['revise_recall'])}.",
        f"- False revisions: {metrics['false_revision_count']}; missed errors: {metrics['missed_error_count']}.",
        f"- Repair success over all REVISE calls: {pct(metrics['repair_success_conditioned_on_revise'])}; "
        f"on initially-wrong REVISE calls: {pct(metrics['wrong_revise_repair_success_rate'])}.",
        "",
        "## Transition table",
        "",
        "| Initial state | Decision | Repair result | Count |",
        "|---|---|---|---:|",
    ]
    for row in transition_matrix["rows"]:
        lines.append(f"| {row['initial_state']} | {row['decision']} | {row['repair_result']} | {row['count']} |")
    lines += [
        "",
        "## Domain results",
        "",
        "| Domain | Initial | Selective final | Always final | Decision | Repair call |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["by_domain"].items():
        lines.append(
            f"| {name} | {pct(values['initial_accuracy'])} | {pct(values['final_accuracy'])} | "
            f"{pct(always_summary['by_domain'][name]['final_accuracy'])} | "
            f"{pct(values['decision_accuracy'])} | {pct(values['repair_call_rate'])} |"
        )
    lines += [
        "",
        "## Dataset results",
        "",
        "| Dataset | Initial | Selective final | Always final | Decision | Repair call |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["by_dataset"].items():
        lines.append(
            f"| {name.upper()} | {pct(values['initial_accuracy'])} | {pct(values['final_accuracy'])} | "
            f"{pct(always_summary['by_dataset'][name]['final_accuracy'])} | "
            f"{pct(values['decision_accuracy'])} | {pct(values['repair_call_rate'])} |"
        )
    lines += [
        "",
        "## Token cost",
        "",
        "Operational token counts include prompt plus completion tokens and exclude optional audit calls.",
        "",
        "| System | Average tokens/request |",
        "|---|---:|",
        f"| V1 only | {tokens['average_total_tokens_per_request']['v1_only']:.1f} |",
        f"| Always repair | {tokens['average_total_tokens_per_request']['always_repair']:.1f} |",
        f"| Selective repair | {tokens['average_total_tokens_per_request']['selective_repair']:.1f} |",
        f"| Selective saving vs always | {tokens['selective_tokens_saved_vs_always_per_request']:.1f} ({pct(tokens['selective_token_reduction_vs_always'])}) |",
        "",
        f"Generated-token reduction vs always repair: {pct(tokens['selective_generated_token_reduction_vs_always'])}. "
        "The much smaller total-token reduction occurs because the full 7B router must reread the",
        "problem and initial answer for every request.",
        "",
        "### Measured service latency",
        "",
        f"- Average decision request: {tokens['latency_seconds']['average_decision_request']:.3f}s.",
        f"- Average repair request: {tokens['latency_seconds']['average_repair_request_all']:.3f}s.",
        f"- Modeled additional latency, always/selective: "
        f"{tokens['latency_seconds']['modeled_average_additional_always']:.3f}s / "
        f"{tokens['latency_seconds']['modeled_average_additional_selective']:.3f}s.",
        "",
        "Initial generations were reused from the frozen project logs, so current-run initial-solve",
        "latency is unavailable. Decision and repair request latencies are retained in raw caches.",
        "",
        "## Controls and audit",
        "",
        f"- Unseen neutral-template decision accuracy: {pct(summary['unseen_template']['decision_accuracy'])}.",
        f"- Substantive separate decision rationales: {summary['rationale_audit']['substantive_count']}/{summary['evaluation_rows']}.",
        "- Rationale generations are not used for routing, scoring, or token-cost comparison.",
        "- Always and selective systems use the same deterministic repair generation per source;",
        "  selective accounting includes it only when the router emitted REVISE.",
        "",
        "## Interpretation",
        "",
        "Selective repair produced three net additional correct answers over V1-only and four",
        "more than always-repair while making 66% fewer repair calls. It strongly reduced damage",
        "to initially-correct answers, but missed 51/100 errors, lost accuracy on code overall,",
        "and generalized weakly to unseen review wording. The accuracy/compute tradeoff improved,",
        "but the margin and total-token saving are too small for a strong claim.",
        f"Paired exact McNemar tests are not significant: V1-only vs selective "
        f"p={comparison['paired_tests']['initial_vs_selective']['exact_mcnemar_pvalue']:.3f}; "
        f"selective vs always p={comparison['paired_tests']['selective_vs_always']['exact_mcnemar_pvalue']:.3f}.",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest_path)
    if len(rows) != 200 or len({row["source_id"] for row in rows}) != 200:
        raise RuntimeError("Expected exactly 200 unique frozen evaluation rows")

    verifier = ObjectiveVerifier()
    reverification_rows = []
    for index, row in enumerate(rows, start=1):
        fresh = verifier.verify(row, row["initial_answer"])
        row["initial_verification_fresh"] = fresh
        reverification_rows.append(
            {
                "source_id": row["source_id"],
                "dataset": row["dataset"],
                "cached_correct": row["initial_correct"],
                "fresh_correct": bool(fresh["passed"]),
                "fresh_detail": fresh["detail"],
                "matches": bool(fresh["passed"]) == bool(row["initial_correct"]),
            }
        )
        if index % 25 == 0:
            print(f"[initial_reverification] {index}/{len(rows)}", flush=True)
    mismatches = [row for row in reverification_rows if not row["matches"]]
    write_json(
        output_dir / "initial_reverification.json",
        {
            "total": len(rows),
            "matching": len(rows) - len(mismatches),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        },
    )
    if mismatches:
        raise RuntimeError(f"Cached/fresh initial verifier mismatches: {len(mismatches)}")
    if args.verify_only:
        print("Initial reverification passed; stopping before model inference.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout_seconds, connect=30.0)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/") + "/", timeout=timeout) as client:
        decisions = await generate_cached(
            client,
            semaphore,
            rows=rows,
            cache_path=output_dir / "decision_generations.jsonl",
            model=args.decision_model,
            max_tokens=16,
            messages_for=lambda row: [
                {"role": "user", "content": row["task_prompt"]},
                {"role": "assistant", "content": row["initial_answer"]},
                {"role": "user", "content": NEUTRAL_REVIEW},
            ],
            batch_size=args.batch_size,
            seed_offset=0,
        )
        for row in rows:
            match = DECISION_RE.fullmatch(decisions[row["source_id"]]["raw"].strip())
            row["decision"] = match.group(1) if match else "INVALID"
            row["decision_correct"] = row["decision"] == row["expected_decision"]

        rationales = await capture_rationales(
            client,
            semaphore,
            rows,
            decisions,
            output_dir / "decision_rationales.jsonl",
            args.decision_model,
            args.batch_size,
        )
        unseen = await generate_cached(
            client,
            semaphore,
            rows=rows,
            cache_path=output_dir / "unseen_template_decisions.jsonl",
            model=args.decision_model,
            max_tokens=16,
            messages_for=lambda row: [
                {"role": "user", "content": row["task_prompt"]},
                {"role": "assistant", "content": row["initial_answer"]},
                {"role": "user", "content": UNSEEN_REVIEW},
            ],
            batch_size=args.batch_size,
            seed_offset=100000,
        )
        repairs = await generate_cached(
            client,
            semaphore,
            rows=rows,
            cache_path=output_dir / "repair_generations.jsonl",
            model=args.solver_model,
            max_tokens=args.repair_max_tokens,
            messages_for=lambda row: [
                {"role": "user", "content": row["task_prompt"]},
                {"role": "assistant", "content": row["initial_answer"]},
                {"role": "user", "content": NEUTRAL_REPAIR},
            ],
            batch_size=args.batch_size,
            seed_offset=200000,
        )

    selective_rows = []
    always_rows = []
    for row in rows:
        source_id = row["source_id"]
        raw_repair = repairs[source_id]["raw"]
        candidate, extracted = repair_candidate(raw_repair)
        repair_result = verifier.verify(row, candidate)
        decision = row["decision"]
        repair_called = decision == "REVISE"
        final_answer = candidate if repair_called else row["initial_answer"]
        final_result = repair_result if repair_called else row["initial_verification_fresh"]
        decision_record = decisions[source_id]
        rationale_record = rationales[source_id]
        selective_rows.append(
            {
                "source_id": source_id,
                "dataset": row["dataset"],
                "domain": row["domain"],
                "source_ref": row["source_ref"],
                "question": row["question"],
                "task_prompt": row["task_prompt"],
                "initial_answer": row["initial_answer"],
                "initial_correct": row["initial_correct"],
                "initial_verifier_detail": row["initial_verification_fresh"]["detail"],
                "expected_decision": row["expected_decision"],
                "decision_raw": decision_record["raw"],
                "decision": decision,
                "decision_correct": row["decision_correct"],
                "decision_finish_reason": decision_record["finish_reason"],
                "decision_usage": {key: decision_record[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")},
                "decision_rationale_raw": rationale_record["raw"],
                "decision_rationale_attempts": rationale_record["attempts"],
                "decision_rationale_substantive": rationale_record["substantive"],
                "decision_rationale_scored": False,
                "repair_called": repair_called,
                "repair_answer": candidate if repair_called else None,
                "repair_raw": raw_repair if repair_called else None,
                "repair_candidate_extracted": extracted if repair_called else None,
                "repair_correct": repair_result["passed"] if repair_called else None,
                "repair_verifier_detail": repair_result["detail"] if repair_called else None,
                "repair_usage": ({key: repairs[source_id][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")} if repair_called else None),
                "final_answer": final_answer,
                "final_correct": bool(final_result["passed"]),
                "final_verifier_detail": final_result["detail"],
                "transition": transition(row["initial_correct"], decision, repair_result["passed"] if repair_called else None),
            }
        )
        always_rows.append(
            {
                "source_id": source_id,
                "dataset": row["dataset"],
                "domain": row["domain"],
                "source_ref": row["source_ref"],
                "question": row["question"],
                "initial_answer": row["initial_answer"],
                "initial_correct": row["initial_correct"],
                "repair_prompt": NEUTRAL_REPAIR,
                "repair_raw": raw_repair,
                "repair_answer": candidate,
                "repair_candidate_extracted": extracted,
                "repair_correct": bool(repair_result["passed"]),
                "repair_verifier_detail": repair_result["detail"],
                "repair_usage": {key: repairs[source_id][key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")},
                "final_answer": candidate,
                "final_correct": bool(repair_result["passed"]),
            }
        )

    metrics = summarize_selective(selective_rows)
    always_metrics = summarize_always(always_rows)
    unseen_predictions = []
    for row in rows:
        match = DECISION_RE.fullmatch(unseen[row["source_id"]]["raw"].strip())
        prediction = match.group(1) if match else "INVALID"
        unseen_predictions.append((prediction, row["expected_decision"]))
    unseen_metrics = {
        "template": UNSEEN_REVIEW,
        "exact_contract_rate": safe_rate(sum(pred != "INVALID" for pred, _ in unseen_predictions), len(rows)),
        "decision_accuracy": safe_rate(sum(pred == gold for pred, gold in unseen_predictions), len(rows)),
    }
    token_metrics = token_summary(selective_rows, decisions, repairs, AutoTokenizer.from_pretrained(args.solver_checkpoint))
    summary = {
        "schema_version": "phase3_two_stage_selective_repair_v1",
        "seed": SEED,
        "evaluation_rows": len(rows),
        "manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
        "models": {
            "initial_solver": args.solver_checkpoint,
            "decision_router": args.decision_model,
            "repair_model": args.solver_checkpoint,
            "weights_merged": False,
            "training_performed": False,
        },
        "prompts": {"decision": NEUTRAL_REVIEW, "repair": NEUTRAL_REPAIR},
        "metrics": metrics,
        "by_domain": slice_accuracy(selective_rows, "domain"),
        "by_dataset": slice_accuracy(selective_rows, "dataset"),
        "unseen_template": unseen_metrics,
        "rationale_audit": {
            "substantive_count": sum(rationales[row["source_id"]]["substantive"] for row in rows),
            "total": len(rows),
            "separate_non_scored_pass": True,
            "hidden_reasoning_available": False,
        },
    }
    always_summary = {
        "schema_version": "phase3_always_repair_control_v1",
        "metrics": always_metrics,
        "by_domain": slice_always(always_rows, "domain"),
        "by_dataset": slice_always(always_rows, "dataset"),
    }
    transitions = {
        "rows": [
            {"initial_state": "Correct", "decision": "KEEP", "repair_result": "n/a", "count": metrics["transition_counts"].get("TRUE_KEEP", 0)},
            {"initial_state": "Correct", "decision": "REVISE", "repair_result": "still correct", "count": metrics["transition_counts"].get("UNNECESSARY_BUT_SAFE_REVISION", 0)},
            {"initial_state": "Correct", "decision": "REVISE", "repair_result": "became wrong", "count": metrics["transition_counts"].get("HARMFUL_FALSE_REVISION", 0)},
            {"initial_state": "Wrong", "decision": "KEEP", "repair_result": "n/a", "count": metrics["transition_counts"].get("MISSED_ERROR", 0)},
            {"initial_state": "Wrong", "decision": "REVISE", "repair_result": "repaired correct", "count": metrics["transition_counts"].get("SUCCESSFUL_RECOVERY", 0)},
            {"initial_state": "Wrong", "decision": "REVISE", "repair_result": "still wrong", "count": metrics["transition_counts"].get("FAILED_REPAIR", 0)},
            {"initial_state": "Any", "decision": "INVALID", "repair_result": "not called", "count": metrics["transition_counts"].get("INVALID_DECISION", 0)},
        ]
    }
    selective_final = metrics["final_accuracy"]
    initial = metrics["initial_accuracy"]
    always_final = always_metrics["final_accuracy"]
    token_saving = token_metrics["selective_token_reduction_vs_always"]
    if (
        selective_final > initial
        and metrics["false_revision_rate"] <= 0.20
        and metrics["wrong_to_correct_recovery_rate"] >= 0.10
        and metrics["correct_to_wrong_degradation_rate"] <= 0.10
        and metrics["repair_call_rate"] <= 0.50
        and metrics["revise_recall"] >= 0.60
        and metrics["balanced_decision_accuracy"] >= 0.70
        and selective_final >= always_final
        and token_saving >= 0.15
        and selective_final - initial >= 0.03
    ):
        conclusion = "selective_repair_strong"
    elif selective_final > initial and metrics["repair_call_rate"] < 0.80 and token_saving > 0:
        conclusion = "selective_repair_promising"
    elif metrics["balanced_decision_accuracy"] >= 0.70 and metrics["wrong_to_correct_recovery_rate"] < 0.10:
        conclusion = "decision_good_repair_weak"
    elif metrics["balanced_decision_accuracy"] < 0.60:
        conclusion = "router_weak"
    else:
        conclusion = "no_selective_advantage"
    comparison = {
        "schema_version": "phase3_selective_vs_always_v1",
        "conclusion": conclusion,
        "initial_accuracy": initial,
        "selective_final_accuracy": selective_final,
        "always_repair_final_accuracy": always_final,
        "selective_minus_initial": round(selective_final - initial, 6),
        "always_minus_initial": round(always_final - initial, 6),
        "selective_minus_always": round(selective_final - always_final, 6),
        "selective_repair_call_rate": metrics["repair_call_rate"],
        "repair_calls_saved_vs_always": len(rows) - metrics["repair_call_count"],
        "repair_call_reduction_vs_always": round(1.0 - metrics["repair_call_rate"], 6),
        "always_repair_call_rate": 1.0,
        "accuracy_gain_per_selective_repair_call": safe_rate(
            sum(row["final_correct"] for row in selective_rows) - sum(row["initial_correct"] for row in selective_rows),
            metrics["repair_call_count"],
        ),
        "token_accounting": token_metrics,
        "paired_repair_outputs": True,
        "paired_tests": {
            "initial_vs_selective": {
                "wrong_to_correct": sum(
                    not row["initial_correct"] and row["final_correct"] for row in selective_rows
                ),
                "correct_to_wrong": sum(
                    row["initial_correct"] and not row["final_correct"] for row in selective_rows
                ),
            },
            "selective_vs_always": {
                "selective_only_correct": sum(
                    selective["final_correct"] and not always["final_correct"]
                    for selective, always in zip(selective_rows, always_rows)
                ),
                "always_only_correct": sum(
                    always["final_correct"] and not selective["final_correct"]
                    for selective, always in zip(selective_rows, always_rows)
                ),
            },
        },
    }
    comparison["paired_tests"]["initial_vs_selective"]["exact_mcnemar_pvalue"] = exact_mcnemar_pvalue(
        comparison["paired_tests"]["initial_vs_selective"]["wrong_to_correct"],
        comparison["paired_tests"]["initial_vs_selective"]["correct_to_wrong"],
    )
    comparison["paired_tests"]["selective_vs_always"]["exact_mcnemar_pvalue"] = exact_mcnemar_pvalue(
        comparison["paired_tests"]["selective_vs_always"]["selective_only_correct"],
        comparison["paired_tests"]["selective_vs_always"]["always_only_correct"],
    )

    write_jsonl(output_dir / "two_stage_results.jsonl", selective_rows)
    write_json(output_dir / "two_stage_summary.json", summary)
    write_json(output_dir / "transition_matrix.json", transitions)
    write_jsonl(output_dir / "always_repair_control.jsonl", always_rows)
    write_json(output_dir / "always_repair_summary.json", always_summary)
    write_json(output_dir / "selective_vs_always_comparison.json", comparison)
    (output_dir / "two_stage_report.md").write_text(
        render_report(summary, always_summary, comparison, transitions), encoding="utf-8", newline="\n"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "two_stage_selective_repair"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--solver-model", default="self-correction-v1")
    parser.add_argument("--decision-model", default="phase3-decision-only-v1")
    parser.add_argument("--solver-checkpoint", default="Kxck/Self_Correction_v1")
    parser.add_argument("--repair-max-tokens", type=int, default=1536)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
