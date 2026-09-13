"""Evaluate one policy on a held-out Phase 3 decision-only split.

The scored pass permits only an exact KEEP/REVISE tag.  A second, non-scored
request captures a concise visible rationale for every sample.  This preserves
the classification contract and never claims access to hidden model reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
SEED = 314159
DECISION_RE = re.compile(r"<decision>(KEEP|REVISE)</decision>")
RATIONALE_PROMPTS = (
    "Audit-only follow-up. Briefly explain the evidence in the problem and your "
    "previous answer that supports the decision you just gave. This explanation "
    "is recorded for analysis and is not part of the scored decision. Do not "
    "change the decision.",
    "The scored classification is already finished. For the audit record, explain "
    "why that decision fits the actual correctness of the previous answer. Do not "
    "output a <decision> tag and do not answer with only KEEP or REVISE. Write at "
    "least two complete sentences beginning with 'Evidence:' and cite a concrete "
    "calculation, test case, or logical step from the problem and previous answer.",
    "Provide explanatory prose for the audit, not another classification label. "
    "Analyze the original problem and the previous answer, identify one specific "
    "piece of supporting or contradicting evidence, and explain in at least two "
    "sentences how it justifies the already-recorded decision. Never emit XML tags.",
)


def substantive_rationale(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 40:
        return False
    if DECISION_RE.fullmatch(cleaned):
        return False
    return len(cleaned.split()) >= 8


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


async def request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, str | None]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": SEED,
    }
    error: Exception | None = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.post(base_url.rstrip("/") + "/chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            return choice["message"].get("content") or "", choice.get("finish_reason")
        except Exception as exc:
            error = exc
            if attempt < 2:
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed after retries: {error}")


async def request_rationale(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    model: str,
    prefix_messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, str | None, str, list[dict[str, Any]]]:
    messages = list(prefix_messages)
    attempts: list[dict[str, Any]] = []
    for prompt in RATIONALE_PROMPTS:
        attempt_messages = messages + [{"role": "user", "content": prompt}]
        raw, finish_reason = await request(
            client,
            semaphore,
            base_url=base_url,
            model=model,
            messages=attempt_messages,
            max_tokens=max_tokens,
        )
        attempts.append(
            {
                "prompt": prompt,
                "raw": raw,
                "finish_reason": finish_reason,
                "substantive": substantive_rationale(raw),
            }
        )
        if substantive_rationale(raw):
            return raw, finish_reason, prompt, attempts
        messages = attempt_messages + [{"role": "assistant", "content": raw}]
    raise RuntimeError(
        "Model did not provide a substantive visible rationale after "
        f"{len(RATIONALE_PROMPTS)} audit prompts"
    )


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = sum(row["contract_valid"] for row in rows)
    correct = sum(row["decision_correct"] for row in rows)
    return {
        "total": len(rows),
        "exact_contract_count": exact,
        "exact_contract_accuracy": safe_rate(exact, len(rows)),
        "decision_correct_count": correct,
        "decision_accuracy": safe_rate(correct, len(rows)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {
        actual: {predicted: sum(
            row["expected_decision"] == actual and row["predicted_decision"] == predicted
            for row in rows
        ) for predicted in ("KEEP", "REVISE", "INVALID")}
        for actual in ("KEEP", "REVISE")
    }
    actual_keep = sum(row["expected_decision"] == "KEEP" for row in rows)
    actual_revise = sum(row["expected_decision"] == "REVISE" for row in rows)
    pred_keep = sum(row["predicted_decision"] == "KEEP" for row in rows)
    pred_revise = sum(row["predicted_decision"] == "REVISE" for row in rows)
    keep_recall = safe_rate(matrix["KEEP"]["KEEP"], actual_keep)
    revise_recall = safe_rate(matrix["REVISE"]["REVISE"], actual_revise)
    return {
        **subset_metrics(rows),
        "keep_recall": keep_recall,
        "revise_recall": revise_recall,
        "keep_precision": safe_rate(matrix["KEEP"]["KEEP"], pred_keep),
        "revise_precision": safe_rate(matrix["REVISE"]["REVISE"], pred_revise),
        "balanced_accuracy": round((keep_recall + revise_recall) / 2, 6)
        if keep_recall is not None and revise_recall is not None else None,
        "confusion_matrix": matrix,
        "conditionals": {
            "p_pred_keep_given_actually_correct": keep_recall,
            "p_pred_revise_given_actually_wrong": revise_recall,
            "p_pred_keep_given_actually_wrong": safe_rate(matrix["REVISE"]["KEEP"], actual_revise),
            "p_pred_revise_given_actually_correct": safe_rate(matrix["KEEP"]["REVISE"], actual_keep),
        },
        "prediction_distribution": dict(sorted(Counter(row["predicted_decision"] for row in rows).items())),
        "by_domain": {
            value: subset_metrics([row for row in rows if row["domain"] == value])
            for value in sorted({row["domain"] for row in rows})
        },
        "by_dataset": {
            value: subset_metrics([row for row in rows if row["dataset"] == value])
            for value in sorted({row["dataset"] for row in rows})
        },
        "rationale_capture": {
            "rows_requested": len(rows),
            "rows_with_visible_reasoning": sum(bool(row["visible_reasoning"].strip()) for row in rows),
            "rows_with_substantive_visible_reasoning": sum(
                row["rationale_substantive"] for row in rows
            ),
            "total_rationale_attempts": sum(row["rationale_attempt_count"] for row in rows),
            "max_rationale_attempts": max(row["rationale_attempt_count"] for row in rows),
            "capture_type": "separate_non_scored_visible_rationale_pass",
            "hidden_reasoning_available": False,
        },
    }


async def run(args: argparse.Namespace) -> None:
    split_path = Path(args.split).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_rows = read_jsonl(split_path)
    if not source_rows:
        raise RuntimeError("Evaluation split is empty")
    timeout = httpx.Timeout(600.0, connect=30.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:
        decisions = await asyncio.gather(*[
            request(
                client,
                semaphore,
                base_url=args.base_url,
                model=args.model,
                messages=row["messages"][:-1],
                max_tokens=args.decision_max_tokens,
            )
            for row in source_rows
        ])
        rationale_tasks = []
        for row, (decision_raw, _) in zip(source_rows, decisions):
            rationale_prefix = list(row["messages"][:-1]) + [
                {"role": "assistant", "content": decision_raw},
            ]
            rationale_tasks.append(
                request_rationale(
                    client,
                    semaphore,
                    base_url=args.base_url,
                    model=args.model,
                    prefix_messages=rationale_prefix,
                    max_tokens=args.rationale_max_tokens,
                )
            )
        rationales = await asyncio.gather(*rationale_tasks)

    output_rows = []
    for source, (decision_raw, decision_finish), rationale_result in zip(
        source_rows, decisions, rationales
    ):
        rationale_raw, rationale_finish, rationale_prompt, rationale_attempts = rationale_result
        match = DECISION_RE.fullmatch(decision_raw.strip())
        predicted = match.group(1) if match else "INVALID"
        output_rows.append(
            {
                "model_label": args.label,
                "served_model": args.model,
                "construction_id": source["construction_id"],
                "source_id": source["source_id"],
                "split": source["split"],
                "dataset": source["dataset"],
                "domain": source["domain"],
                "bucket": source["bucket"],
                "neutral_template_id": source["neutral_template_id"],
                "expected_decision": source["decision"],
                "input_messages": source["messages"][:-1],
                "decision_raw": decision_raw,
                "decision_finish_reason": decision_finish,
                "predicted_decision": predicted,
                "contract_valid": match is not None,
                "decision_correct": predicted == source["decision"],
                "rationale_prompt": rationale_prompt,
                "rationale_raw": rationale_raw,
                "visible_reasoning": rationale_raw,
                "rationale_finish_reason": rationale_finish,
                "rationale_attempt_count": len(rationale_attempts),
                "rationale_attempts": rationale_attempts,
                "rationale_substantive": substantive_rationale(rationale_raw),
                "reasoning_capture_type": "separate_non_scored_visible_rationale_pass",
                "reasoning_used_for_decision_score": False,
                "hidden_reasoning_available": False,
            }
        )
    summary = {
        "schema_version": "phase3_decision_only_eval_v1",
        "model_label": args.label,
        "served_model": args.model,
        "seed": SEED,
        "decoding": {
            "temperature": 0.0,
            "decision_max_tokens": args.decision_max_tokens,
            "rationale_max_tokens": args.rationale_max_tokens,
        },
        "split": {
            "path": str(split_path),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "rows": len(source_rows),
        },
        "metrics": summarize(output_rows),
    }
    write_jsonl(output_dir / f"{args.label}_rows.jsonl", output_rows)
    write_json(output_dir / f"{args.label}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--decision-max-tokens", type=int, default=16)
    parser.add_argument("--rationale-max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
