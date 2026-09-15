"""Shared JSONL, problem, verifier, and completion helpers for Phase 4 rollouts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "phase1"))

from src.core.schema import Problem
from src.data.verifiers.code import CodeVerifier
from src.data.verifiers.math import MathVerifier


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_problem(row: dict[str, Any]) -> Problem:
    domain = str(row["domain"])
    if domain not in {"math", "code"}:
        raise ValueError(f"Unsupported domain for {row.get('id')}: {domain}")
    problem = Problem(
        id=str(row["id"]),
        domain=domain,
        question=str(row["question"]),
        reference_answer=(
            str(row["reference_answer"]) if row.get("reference_answer") is not None else None
        ),
        entry_point=(str(row["entry_point"]) if row.get("entry_point") else None),
        tests=[str(value) for value in row.get("tests", [])],
        calc_steps=row.get("calc_steps"),
    )
    if domain == "math" and problem.reference_answer is None:
        raise ValueError(f"Math row {problem.id} lacks reference_answer")
    if domain == "code" and (not problem.entry_point or not problem.tests):
        raise ValueError(f"Code row {problem.id} lacks entry_point/tests")
    return problem


def verify(problem: Problem, answer: str) -> dict[str, Any]:
    verifier = MathVerifier() if problem.domain == "math" else CodeVerifier()
    result = verifier.verify(problem, answer)
    return {"passed": bool(result.passed), "detail": str(result.detail)}


async def completion(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    async with semaphore:
        response = await client.post("chat/completions", json=payload)
        response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])
