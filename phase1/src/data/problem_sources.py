"""Doc bo bai toan (math/code) tu file JSONL thanh danh sach Problem."""
from __future__ import annotations

import json
from pathlib import Path

from src.core.schema import Problem


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_math_problems(path: str | Path) -> list[Problem]:
    """Format 1 dong: {"id": "...", "question": "...", "reference_answer": "..."}"""
    problems = []
    for row in _read_jsonl(Path(path)):
        problems.append(
            Problem(
                id=row["id"],
                domain="math",
                question=row["question"],
                reference_answer=str(row["reference_answer"]),
                calc_steps=row.get("calc_steps") or None,
            )
        )
    return problems


def load_code_problems(path: str | Path) -> list[Problem]:
    """Format 1 dong: {"id": "...", "question": "...", "entry_point": "func_name", "tests": ["assert ..."]}"""
    problems = []
    for row in _read_jsonl(Path(path)):
        problems.append(
            Problem(
                id=row["id"],
                domain="code",
                question=row["question"],
                entry_point=row["entry_point"],
                tests=row["tests"],
            )
        )
    return problems
