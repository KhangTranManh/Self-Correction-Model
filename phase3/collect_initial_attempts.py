"""Collect, objectively verify, and retain every raw Phase 3 initial attempt.

The script talks to an OpenAI-compatible vLLM server. It is resumable by source
ID, writes records after each batch, and derives correct/wrong pools from the raw
artifact instead of filtering generations before they are stored.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml


ROOT = Path(__file__).resolve().parent
PHASE1_ROOT = ROOT.parent / "phase1"
sys.path.insert(0, str(PHASE1_ROOT))

from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.code import CodeVerifier, _extract_code  # noqa: E402
from src.data.verifiers.math import MathVerifier, _extract_answer  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "phase3.yaml"


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _problem(row: dict) -> Problem:
    return Problem(
        id=row["id"],
        domain=row["domain"],
        question=row["problem"],
        reference_answer=row.get("reference_answer"),
        entry_point=row.get("entry_point"),
        tests=row.get("tests", []),
        calc_steps=row.get("calc_steps") or None,
    )


async def _request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    served_model: str,
    messages: list[dict],
    generation: dict,
    seed: int,
) -> tuple[str, str | None]:
    payload = {
        "model": served_model,
        "messages": messages,
        "max_tokens": int(generation["max_tokens"]),
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "seed": seed,
    }
    last_error: Exception | None = None
    for attempt in range(int(generation["max_retries"])):
        try:
            async with semaphore:
                response = await client.post(
                    base_url.rstrip("/") + "/chat/completions", json=payload
                )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            return choice["message"].get("content") or "", choice.get("finish_reason")
        except Exception as exc:  # retry transport and transient server errors
            last_error = exc
            if attempt + 1 < int(generation["max_retries"]):
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed after retries: {last_error}")


def _record(
    row: dict,
    *,
    model_label: str,
    model_output: str,
    verifiers: dict[str, Any],
) -> dict:
    problem = _problem(row)
    result = verifiers[problem.domain].verify(problem, model_output)
    extracted = (
        _extract_answer(model_output)
        if problem.domain == "math"
        else _extract_code(model_output)
    )
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "problem": row["problem"],
        "model": model_label,
        "initial_output": model_output,
        "extracted_answer": extracted,
        "ground_truth": row["ground_truth"],
        "initial_correct": result.passed,
        "verifier_detail": result.detail,
    }


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _derive_artifacts(
    raw_path: Path,
    source_rows: list[dict],
    *,
    model_label: str,
    served_model: str,
    generation: dict,
) -> dict:
    rows = _read_jsonl(raw_path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise RuntimeError(f"Duplicate attempt IDs: {duplicates[:10]}")
    source_ids = {row["id"] for row in source_rows}
    if set(ids) - source_ids:
        raise RuntimeError("Attempt artifact contains IDs outside the source pool")

    correct = [row for row in rows if row["initial_correct"]]
    wrong = [row for row in rows if not row["initial_correct"]]
    _atomic_jsonl(raw_path.parent / "correct.jsonl", correct)
    _atomic_jsonl(raw_path.parent / "wrong.jsonl", wrong)

    groups: dict[str, dict] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        group_correct = sum(row["initial_correct"] for row in group)
        groups[dataset] = {
            "total": len(group),
            "correct": group_correct,
            "wrong": len(group) - group_correct,
            "accuracy": group_correct / len(group) if group else None,
        }
    raw_bytes = raw_path.read_bytes()
    summary = {
        "model_label": model_label,
        "served_model": served_model,
        "complete": len(rows) == len(source_rows),
        "expected": len(source_rows),
        "total": len(rows),
        "correct": len(correct),
        "wrong": len(wrong),
        "accuracy": len(correct) / len(rows) if rows else None,
        "datasets": groups,
        "generation": {
            "max_tokens": int(generation["max_tokens"]),
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "seed": int(generation["seed"]),
        },
        "raw_attempts_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_attempts": str(raw_path),
        "correct_pool": str(raw_path.parent / "correct.jsonl"),
        "wrong_pool": str(raw_path.parent / "wrong.jsonl"),
    }
    (raw_path.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


async def _run(args: argparse.Namespace) -> dict:
    config = _load_config()
    generation = config["generation"]
    source_path = _resolve(config["paths"]["source_problems"])
    source_rows = _read_jsonl(source_path)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    output_dir = _resolve(config["paths"]["attempts_dir"]) / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_attempts.jsonl"

    done_ids = {row["id"] for row in _read_jsonl(raw_path)} if raw_path.exists() else set()
    pending = [row for row in source_rows if row["id"] not in done_ids]
    print(
        f"[{args.model_label}] source={len(source_rows)} done={len(done_ids)} "
        f"pending={len(pending)}",
        flush=True,
    )

    verifier_config = config["verifier"]
    verifiers: dict[str, Any] = {
        "math": MathVerifier(),
        "code": CodeVerifier(
            timeout_seconds=int(verifier_config["code_timeout_seconds"]),
            memory_limit_mb=int(verifier_config["code_memory_limit_mb"]),
        ),
    }
    timeout = httpx.Timeout(float(generation["request_timeout_seconds"]))
    semaphore = asyncio.Semaphore(int(generation["concurrency"]))
    base_seed = int(generation["seed"])

    async with httpx.AsyncClient(timeout=timeout) as client:
        batch_size = int(generation["batch_size"])
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            requests = []
            source_positions = {
                row["id"]: position for position, row in enumerate(source_rows)
            }
            for row in batch:
                source_position = source_positions[row["id"]]
                messages = [{"role": "user", "content": build_prompt(_problem(row))}]
                requests.append(
                    _request_one(
                        client,
                        semaphore,
                        base_url=args.base_url,
                        served_model=args.served_model,
                        messages=messages,
                        generation=generation,
                        seed=base_seed + source_position,
                    )
                )
            outputs = await asyncio.gather(*requests)
            with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
                for row, (text, _finish_reason) in zip(batch, outputs):
                    record = _record(
                        row,
                        model_label=args.model_label,
                        model_output=text,
                        verifiers=verifiers,
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            completed = len(done_ids) + min(start + len(batch), len(pending))
            print(
                f"[{args.model_label}] saved {completed}/{len(source_rows)}",
                flush=True,
            )

    return _derive_artifacts(
        raw_path,
        source_rows,
        model_label=args.model_label,
        served_model=args.served_model,
        generation=generation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N rows for a smoke test"
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
