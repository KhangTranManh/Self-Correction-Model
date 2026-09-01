"""Evaluate Phase-1 self-correction through a running vLLM OpenAI server.

This is the HTTP-serving counterpart of ``evaluate_self_correction.py``.  It
uses held-out GSM8K/MBPP problems, objective verifiers, and the exact reflection
prompt/role used by the training data.  Initial and correction generations are
batched concurrently so vLLM can use continuous batching.

Example:
    python -m src.evaluate_self_correction_vllm \
        --model phase1-self-correction --math-limit 10 --code-limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
from datasets import load_dataset

from src.config import ROOT_DIR, load_config
from src.data.schema import Problem, VerifierResult
from src.prepare_public_datasets import _extract_entry_point, _extract_gsm8k_answer
from src.verifier.code_verifier import CodeVerifier
from src.verifier.math_verifier import MathVerifier

# Keep these byte-for-byte identical to generate_attempts.py/build_dataset.py.
_PROMPT_TEMPLATES = {
    "math": (
        "Giáº£i bÃ i toÃ¡n sau tá»«ng bÆ°á»›c, sau Ä‘Ã³ ghi rÃµ Ä‘Ã¡p sá»‘ cuá»‘i cÃ¹ng theo Ä‘á»‹nh dáº¡ng "
        "'ÄÃ¡p sá»‘: <giÃ¡ trá»‹>'.\n\nBÃ i toÃ¡n: {question}"
    ),
    "code": (
        "{question}\n\nChá»‰ tráº£ vá» code Python hoÃ n chá»‰nh trong 1 code block "
        "(```python ... ```), khÃ´ng giáº£i thÃ­ch thÃªm."
    ),
}
_REFLECT_PROMPT_TEMPLATE = (
    "Káº¿t quáº£ kiá»ƒm tra: SAI.\n"
    "Chi tiáº¿t lá»—i tá»« há»‡ thá»‘ng kiá»ƒm tra: {verifier_detail}\n"
    "HÃ£y tá»± rÃ  soÃ¡t láº¡i lá»i giáº£i trÃªn vÃ  sá»­a láº¡i cho Ä‘Ãºng."
)
# Match the real Unicode heading without putting locale-sensitive bytes in this
# source line. Some valid generations use the equivalent English heading, and
# some put the corrected content on the same line as the heading.
_CORRECTION_RE = re.compile(
    r"###\s*(?:S\u1eeda\s+l\u1ea1i|Corrected\s+answer|Correction)"
    r"\s*:?[ \t]*(?:\r?\n)?(.*)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _build_prompt(problem: Problem) -> str:
    return _PROMPT_TEMPLATES[problem.domain].format(question=problem.question)


def _load_held_out_problems(
    math_offset: int, math_limit: int, code_offset: int, code_limit: int
) -> list[Problem]:
    problems: list[Problem] = []
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    for index in range(math_offset, min(math_offset + math_limit, len(gsm8k))):
        row = gsm8k[index]
        problems.append(
            Problem(
                id=f"eval_gsm8k_{index:04d}",
                domain="math",
                question=row["question"],
                reference_answer=_extract_gsm8k_answer(row["answer"]),
            )
        )

    mbpp = load_dataset(
        "parquet",
        data_files=(
            "hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/full/test/0000.parquet"
        ),
        split="train",
    )
    for index in range(code_offset, min(code_offset + code_limit, len(mbpp))):
        row = mbpp[index]
        entry_point = _extract_entry_point(row["code"])
        if entry_point is None:
            continue
        tests = list(row["test_list"])
        if row.get("test_setup_code"):
            tests = [row["test_setup_code"]] + tests
        problems.append(
            Problem(
                id=f"eval_mbpp_{row['task_id']}",
                domain="code",
                question=row["text"],
                entry_point=entry_point,
                tests=tests,
            )
        )
    return problems


async def _chat(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, str | None]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
    async with semaphore:
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason")


def _new_stats() -> dict[str, int]:
    return {
        "total": 0,
        "initial_correct": 0,
        "initial_wrong": 0,
        "self_corrected": 0,
        "still_wrong": 0,
        "format_incomplete": 0,
    }


def _finish_stats(stats: dict[str, int]) -> dict:
    result = dict(stats)
    result["initial_accuracy"] = (
        stats["initial_correct"] / stats["total"] if stats["total"] else None
    )
    result["self_correction_rate"] = (
        stats["self_corrected"] / stats["initial_wrong"]
        if stats["initial_wrong"]
        else None
    )
    return result


async def _run(args: argparse.Namespace) -> dict:
    cfg = load_config(require_deepseek=False)
    problems = _load_held_out_problems(
        args.math_offset, args.math_limit, args.code_offset, args.code_limit
    )
    verifiers = {
        "math": MathVerifier(),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }
    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {api_key}"}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout_seconds)
    gen_cfg = cfg.generation

    initial_messages = [
        [{"role": "user", "content": _build_prompt(problem)}] for problem in problems
    ]
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/") + "/", headers=headers, timeout=timeout
    ) as client:
        initial_outputs = await asyncio.gather(
            *[
                _chat(
                    client,
                    semaphore,
                    model=args.model,
                    messages=messages,
                    max_tokens=gen_cfg["max_new_tokens"],
                    temperature=args.temperature,
                    top_p=gen_cfg["top_p"],
                    seed=args.seed + index,
                )
                for index, messages in enumerate(initial_messages)
            ]
        )

        records: list[dict] = []
        correction_jobs: list[tuple[int, list[dict]]] = []
        for index, (problem, (attempt, finish_reason)) in enumerate(
            zip(problems, initial_outputs)
        ):
            first = verifiers[problem.domain].verify(problem, attempt)
            record = {
                "problem_id": problem.id,
                "domain": problem.domain,
                "attempt_text": attempt,
                "initial_finish_reason": finish_reason,
                "initial_passed": first.passed,
                "verifier_detail": first.detail,
            }
            records.append(record)
            if not first.passed:
                correction_jobs.append(
                    (
                        index,
                        initial_messages[index]
                        + [
                            {"role": "assistant", "content": attempt},
                            {
                                "role": args.reflect_role,
                                "content": _REFLECT_PROMPT_TEMPLATE.format(
                                    verifier_detail=first.detail
                                ),
                            },
                        ],
                    )
                )

        correction_outputs = await asyncio.gather(
            *[
                _chat(
                    client,
                    semaphore,
                    model=args.model,
                    messages=messages,
                    max_tokens=gen_cfg.get(
                        "max_new_tokens_correction", gen_cfg["max_new_tokens"]
                    ),
                    temperature=args.temperature,
                    top_p=gen_cfg["top_p"],
                    seed=args.seed + 10_000 + index,
                )
                for index, messages in correction_jobs
            ]
        )

    for (record_index, _), (correction, finish_reason) in zip(
        correction_jobs, correction_outputs
    ):
        problem = problems[record_index]
        match = _CORRECTION_RE.search(correction)
        if match:
            corrected_solution = match.group(1).strip()
            second = verifiers[problem.domain].verify(problem, corrected_solution)
        else:
            corrected_solution = None
            second = VerifierResult(
                passed=False,
                detail="Missing completed '### Sua lai' correction section",
            )
        records[record_index].update(
            correction_text=correction,
            correction_finish_reason=finish_reason,
            corrected_solution=corrected_solution,
            format_incomplete=match is None,
            second_passed=second.passed,
            second_detail=second.detail,
            reflect_role=args.reflect_role,
        )

    overall = _new_stats()
    domains = {"math": _new_stats(), "code": _new_stats()}
    for record in records:
        for stats in (overall, domains[record["domain"]]):
            stats["total"] += 1
            if record["initial_passed"]:
                stats["initial_correct"] += 1
            else:
                stats["initial_wrong"] += 1
                if record.get("second_passed"):
                    stats["self_corrected"] += 1
                else:
                    stats["still_wrong"] += 1
                if record.get("format_incomplete"):
                    stats["format_incomplete"] += 1

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "held_out": True,
        "reflect_role": args.reflect_role,
        "seed": args.seed,
        "overall": _finish_stats(overall),
        "domains": {name: _finish_stats(stats) for name, stats in domains.items()},
        "log_file": str(log_path),
    }
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--api-key-file", default=str(ROOT_DIR / "outputs" / "vllm_api_key"))
    parser.add_argument("--model", default="phase1-self-correction")
    parser.add_argument("--math-offset", type=int, default=150)
    parser.add_argument("--math-limit", type=int, default=10)
    parser.add_argument("--code-offset", type=int, default=150)
    parser.add_argument("--code-limit", type=int, default=10)
    parser.add_argument("--reflect-role", choices=["tool", "user"], default="tool")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--log-file", default="outputs/eval_self_correction_vllm_log.jsonl"
    )
    parser.add_argument(
        "--summary-file", default="outputs/eval_self_correction_vllm_summary.json"
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), indent=2))


if __name__ == "__main__":
    main()
