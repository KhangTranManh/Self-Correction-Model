"""Collect stochastic same-model generations from a local vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import httpx


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    tasks = [row for row in read_jsonl(manifest_path) if row["model_origin_to_sample"] == args.origin]
    existing = {row["candidate_id"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    jobs = []
    for task_index, task in enumerate(tasks):
        count = min(int(task["samples_requested"]), args.max_samples_per_task or 10**9)
        for sample_index in range(count):
            candidate_id = f"{task['task_id']}::{sample_index:02d}"
            if candidate_id not in existing:
                jobs.append((task_index, sample_index, candidate_id, task))
    timeout = httpx.Timeout(600, connect=30)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(job: tuple[int, int, str, dict[str, Any]], client: httpx.AsyncClient) -> dict[str, Any]:
        task_index, sample_index, candidate_id, task = job
        payload = {
            "model": args.model,
            "messages": [task["problem_message"]],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.code_max_tokens if task["domain"] == "code" else args.math_max_tokens,
            "seed": args.seed + task_index * 100 + sample_index,
        }
        last_error = None
        for attempt in range(3):
            try:
                async with semaphore:
                    started = time.perf_counter()
                    response = await client.post(args.base_url.rstrip("/") + "/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                return {
                    "candidate_id": candidate_id,
                    "task_id": task["task_id"],
                    "pair_id": task["pair_id"],
                    "source_id": task["source_id"],
                    "problem_ref": task["problem_ref"],
                    "dataset": task["dataset"],
                    "domain": task["domain"],
                    "model_origin": args.origin,
                    "served_model": args.model,
                    "sample_index": sample_index,
                    "seed": payload["seed"],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": payload["max_tokens"],
                    "raw_answer": choice["message"].get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage"),
                    "latency_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"{candidate_id}: {last_error}")

    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(jobs), args.batch_size):
            chunk = jobs[start : start + args.batch_size]
            results = await asyncio.gather(*(one(job, client) for job in chunk))
            existing.update((row["candidate_id"], row) for row in results)
            write_jsonl(output_path, sorted(existing.values(), key=lambda row: row["candidate_id"]))
            print(f"{args.origin}: {min(start + len(chunk), len(jobs))}/{len(jobs)} new; total={len(existing)}", flush=True)
    summary = {
        "schema_version": "phase3_same_origin_candidates_v1",
        "origin": args.origin,
        "served_model": args.model,
        "tasks": len(tasks),
        "new_candidates": len(jobs),
        "total_candidates": len(existing),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--origin", choices=("base", "self_correction_v1"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--math-max-tokens", type=int, default=1024)
    parser.add_argument("--code-max-tokens", type=int, default=1536)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples-per-task", type=int)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
