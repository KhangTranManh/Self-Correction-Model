"""Generate and objectively verify APPS pilot attempts through a vLLM server."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.apps_verifier import extract_code, verify


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "apps_pilot.yaml"


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _prompt(row: dict[str, Any]) -> str:
    protocol = (
        "Write a complete Python 3 program that reads from standard input and writes to standard output."
        if row["test_mode"] == "standard_input"
        else f"Implement the requested callable `{row['tests']['fn_name']}` exactly as specified."
    )
    starter = row.get("starter_code") or ""
    starter_section = f"\n\nStarter code (preserve its required interface):\n{starter}" if starter.strip() else ""
    return (
        "Solve the following programming problem. "
        f"{protocol} Return only complete Python code in one ```python code block, with no explanation.\n\n"
        f"Problem:\n{row['problem']}{starter_section}"
    )


async def _request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    base_url: str,
    served_model: str,
    prompt: str,
    generation: dict[str, Any],
    seed: int,
) -> tuple[str, str | None]:
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": prompt}],
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
        except Exception as exc:
            last_error = exc
            if attempt + 1 < int(generation["max_retries"]):
                await asyncio.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed after retries: {last_error}")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _summary(
    raw_path: Path,
    candidates: list[dict[str, Any]],
    *,
    model_label: str,
    model_id: str,
    served_model: str,
    generation: dict[str, Any],
    candidate_sha256: str,
) -> dict[str, Any]:
    rows = _read_jsonl(raw_path)
    candidate_ids = [row["id"] for row in candidates]
    attempt_ids = [row["id"] for row in rows]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise RuntimeError("Duplicate attempt IDs")
    if set(attempt_ids) - set(candidate_ids):
        raise RuntimeError("Attempt IDs outside the frozen candidate manifest")
    correct = [row for row in rows if row["initial_correct"]]
    result = {
        "model_label": model_label,
        "model_id": model_id,
        "served_model": served_model,
        "complete": attempt_ids == candidate_ids,
        "expected": len(candidates),
        "total": len(rows),
        "correct": len(correct),
        "wrong": len(rows) - len(correct),
        "accuracy": len(correct) / len(rows) if rows else None,
        "candidate_sha256": candidate_sha256,
        "generation": {
            "max_tokens": int(generation["max_tokens"]),
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "seed": int(generation["seed"]),
        },
        "correct_by_test_mode": dict(
            sorted(Counter(row["test_mode"] for row in correct).items())
        ),
        "total_by_test_mode": dict(
            sorted(Counter(row["test_mode"] for row in rows).items())
        ),
        "correct_by_source_host": dict(
            sorted(Counter(row["source_host"] for row in correct).items())
        ),
        "raw_attempts": str(raw_path),
        "raw_attempts_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    summary_path = raw_path.parent / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    generation = config["generation"]
    verifier_cfg = config["verifier"]
    candidate_path = _resolve(config["paths"]["candidates"])
    candidate_summary_path = _resolve(config["paths"]["candidate_summary"])
    candidates = _read_jsonl(candidate_path)
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    if candidate_sha256 != candidate_summary["candidate_sha256"]:
        raise RuntimeError("Frozen candidate manifest hash differs from selection summary")
    if len(candidates) != int(config["dataset"]["pilot_size"]):
        raise RuntimeError("Candidate count differs from the configured pilot size")

    model_cfg = config["models"][args.model_label]
    if args.served_model != model_cfg["served_model_name"]:
        raise ValueError("served-model does not match the frozen model config")
    output_dir = _resolve(config["paths"]["attempts_dir"]) / args.model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_attempts.jsonl"
    existing = _read_jsonl(raw_path) if raw_path.exists() else []
    done = {row["id"]: row for row in existing}
    if any(row.get("model") != args.model_label for row in existing):
        raise RuntimeError("Existing resume artifact contains another model label")
    pending = [row for row in candidates if row["id"] not in done]
    positions = {row["id"]: index for index, row in enumerate(candidates)}
    print(
        f"[{args.model_label}] frozen_candidates={len(candidates)} done={len(done)} pending={len(pending)}",
        flush=True,
    )

    headers: dict[str, str] = {}
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(float(generation["request_timeout_seconds"]))
    semaphore = asyncio.Semaphore(int(generation["concurrency"]))
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        batch_size = int(generation["batch_size"])
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            outputs = await asyncio.gather(
                *[
                    _request_one(
                        client,
                        semaphore,
                        base_url=args.base_url,
                        served_model=args.served_model,
                        prompt=_prompt(row),
                        generation=generation,
                        seed=int(generation["seed"]) + positions[row["id"]],
                    )
                    for row in batch
                ]
            )
            new_records = []
            for row, (output, finish_reason) in zip(batch, outputs):
                verification = verify(
                    output,
                    row["tests"],
                    per_test_timeout_seconds=int(verifier_cfg["per_test_timeout_seconds"]),
                    memory_limit_mb=int(verifier_cfg["memory_limit_mb"]),
                    max_output_bytes=int(verifier_cfg["max_output_bytes"]),
                )
                new_records.append(
                    {
                        "id": row["id"],
                        "dataset": "apps",
                        "split": "train",
                        "difficulty": row["difficulty"],
                        "domain": "code",
                        "source_index": row["source_index"],
                        "problem_id": row["problem_id"],
                        "problem": row["problem"],
                        "source_host": row["source_host"],
                        "test_mode": row["test_mode"],
                        "test_count": row["test_count"],
                        "model": args.model_label,
                        "model_id": model_cfg["model_id"],
                        "initial_output": output,
                        "extracted_answer": extract_code(output),
                        "ground_truth": row["ground_truth"],
                        "initial_correct": verification["passed"],
                        "verifier_detail": verification["detail"],
                        "passed_tests": verification["passed_tests"],
                        "total_tests": verification["total_tests"],
                        "finish_reason": finish_reason,
                        "candidate_sha256": candidate_sha256,
                    }
                )
            with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
                for record in new_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(
                f"[{args.model_label}] saved={len(done) + min(start + len(batch), len(pending))}/{len(candidates)}",
                flush=True,
            )

    return _summary(
        raw_path,
        candidates,
        model_label=args.model_label,
        model_id=model_cfg["model_id"],
        served_model=args.served_model,
        generation=generation,
        candidate_sha256=candidate_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-label", choices=("base", "self_correction_v1"), required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--api-key-file", default="")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
