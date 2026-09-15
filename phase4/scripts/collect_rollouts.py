"""Collect and objectively score two-turn Phase 4 rollouts from a vLLM server."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phase1"))

from phase4.lib.transition import (  # noqa: E402
    parse_review,
    transition_name,
    transition_reward,
)
from phase4.lib.rollout import (  # noqa: E402
    as_problem,
    completion,
    read_jsonl,
    sha256,
    verify,
    write_jsonl_atomic,
)
from src.core.prompts import build_prompt  # noqa: E402


async def collect_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    config: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    problem = as_problem(row)
    rollout = config["rollout"]
    model = str(config["serving_model"])
    task_prompt = build_prompt(problem)
    reused = config["initial_by_id"].get(problem.id)
    if reused is None:
        initial_output = await completion(
            client,
            semaphore,
            model=model,
            messages=[{"role": "user", "content": task_prompt}],
            temperature=float(rollout["initial_temperature"]),
            max_tokens=int(rollout["initial_max_tokens"]),
            seed=int(config["seed"]) + index,
        )
    else:
        initial_output = str(reused["initial_output"])
    initial_verification = await asyncio.to_thread(verify, problem, initial_output)
    review_output = await completion(
        client,
        semaphore,
        model=model,
        messages=[
            {"role": "system", "content": str(rollout["review_system_prompt"])},
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": initial_output},
            {"role": "user", "content": str(rollout["neutral_prompt"])},
        ],
        temperature=float(rollout["review_temperature"]),
        max_tokens=int(rollout["review_max_tokens"]),
        seed=int(config["seed"]) + 1_000_000 + index,
    )
    action = parse_review(review_output)
    final_answer = initial_output if action.decision == "KEEP" else (action.revised_answer or "")
    final_verification = await asyncio.to_thread(verify, problem, final_answer)
    reward, reward_reason = transition_reward(
        initial_correct=initial_verification["passed"],
        final_correct=final_verification["passed"],
        action=action,
        rewards=config["reward"],
    )
    return {
        "schema_version": "phase4_transition_rollout_v1",
        "rollout_id": f"{config['cycle_id']}::{problem.id}",
        "cycle_id": config["cycle_id"],
        "policy_checkpoint": config["policy_checkpoint"],
        "initial_output_reused": reused is not None,
        "problem_id": problem.id,
        "domain": problem.domain,
        # Ground truth is retained only as verifier metadata and is never placed
        # in either model-visible prompt.
        "verifier_spec": {
            "reference_answer": problem.reference_answer,
            "entry_point": problem.entry_point,
            "tests": problem.tests,
            "calc_steps": problem.calc_steps,
        },
        "source_manifest_sha256": config["source_manifest_sha256"],
        "task_prompt": task_prompt,
        "initial_output": initial_output,
        "initial_correct": initial_verification["passed"],
        "initial_verifier_detail": initial_verification["detail"],
        "review_output": review_output,
        "decision": action.decision,
        "contract_valid": action.valid,
        "final_answer": final_answer,
        "final_correct": final_verification["passed"],
        "final_verifier_detail": final_verification["detail"],
        "transition": transition_name(
            initial_verification["passed"], final_verification["passed"]
        ),
        "reward": reward,
        "reward_reason": reward_reason,
    }


async def run(args: argparse.Namespace) -> None:
    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.problems)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Problem manifest contains duplicate ids")
    config = {
        **base_config,
        "cycle_id": args.cycle_id,
        "policy_checkpoint": args.policy_checkpoint,
        "serving_model": args.serving_model,
        "seed": args.seed if args.seed is not None else int(base_config["experiment"]["seed"]),
        "source_manifest_sha256": sha256(args.problems),
        "initial_by_id": {},
    }
    if args.initial_rollouts:
        reused_rows = read_jsonl(args.initial_rollouts)
        config["initial_by_id"] = {
            str(row["problem_id"]): row for row in reused_rows
        }
        missing_reused = [problem_id for problem_id in ids if problem_id not in config["initial_by_id"]]
        if missing_reused:
            raise ValueError(f"Initial rollout file lacks problem ids: {missing_reused[:10]}")
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/") + "/", timeout=timeout) as client:
        tasks = [collect_one(client, semaphore, row, config, i) for i, row in enumerate(rows)]
        results = await asyncio.gather(*tasks)
    results.sort(key=lambda row: row["problem_id"])
    write_jsonl_atomic(args.output, results)
    counts = Counter(row["reward_reason"] for row in results)
    print(json.dumps({
        "output": str(args.output),
        "rows": len(results),
        "source_manifest_sha256": config["source_manifest_sha256"],
        "reward_reasons": dict(sorted(counts.items())),
        "mean_reward": sum(float(row["reward"]) for row in results) / len(results) if results else None,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--serving-model", default="phase4-actor")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-rollouts", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "phase4/configs/transition_rl_v1.yaml",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
