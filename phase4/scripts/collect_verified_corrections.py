"""Build balanced SFT data from multiple verified corrections per natural failure."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phase1"))

from phase4.lib.transition import parse_review  # noqa: E402
from phase4.scripts.collect_warmstart import read_jsonl, verify, write_jsonl  # noqa: E402


def stable_rank(label: str, problem_id: str) -> str:
    return hashlib.sha256(f"20260914:expansion:{label}:{problem_id}".encode()).hexdigest()


async def generate_candidate(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    system_prompt: str,
    wrong_index: int,
    candidate_index: int,
    model: str,
) -> dict[str, Any]:
    feedback = (
        "The deterministic verifier rejected the previous answer and returned this evidence:\n"
        f"{row['initial_verifier_detail']}\n"
        "Return only <decision>REVISE</decision><answer>COMPLETE CORRECTED ANSWER</answer>."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["task_prompt"]},
            {"role": "assistant", "content": row["initial_output"]},
            {"role": "user", "content": feedback},
        ],
        "temperature": 0.9,
        "max_tokens": 1200,
        "seed": 5_000_000 + wrong_index * 100 + candidate_index,
    }
    async with semaphore:
        response = await client.post("chat/completions", json=payload)
        response.raise_for_status()
    output = str(response.json()["choices"][0]["message"]["content"])
    action = parse_review(output)
    passed = bool(
        action.valid
        and action.decision == "REVISE"
        and action.revised_answer
        and await asyncio.to_thread(verify, row, action.revised_answer)
    )
    return {
        "problem_id": row["problem_id"],
        "candidate_index": candidate_index,
        "guided_prompt": feedback,
        "guided_output": output,
        "contract_valid": action.valid,
        "correction_passed": passed,
    }


async def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rollout = config["rollout"]
    rows = read_jsonl(args.initials)
    wrong = [row for row in rows if not row["initial_correct"]]
    correct = [row for row in rows if row["initial_correct"]]
    by_id = {str(row["problem_id"]): row for row in rows}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/") + "/", timeout=timeout) as client:
        attempts = await asyncio.gather(*[
            generate_candidate(
                client,
                semaphore,
                row,
                str(rollout["review_system_prompt"]),
                wrong_index,
                candidate_index,
                args.serving_model,
            )
            for wrong_index, row in enumerate(wrong)
            for candidate_index in range(args.candidates_per_wrong)
        ])
    attempts.sort(key=lambda row: (row["problem_id"], row["candidate_index"]))
    passing_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        if attempt["correction_passed"]:
            passing_by_id[str(attempt["problem_id"])].append(attempt)
    verified = [values[0] for _, values in sorted(passing_by_id.items())]
    verified.sort(key=lambda row: stable_rank("REVISE", str(row["problem_id"])))
    correct.sort(key=lambda row: stable_rank("KEEP", str(row["problem_id"])))
    pair_count = min(len(verified), len(correct), args.max_per_label)
    if pair_count < args.minimum_per_label:
        raise RuntimeError(f"Only {pair_count} verified examples per label; need {args.minimum_per_label}")

    selected = []
    for decision, values in (("REVISE", verified[:pair_count]), ("KEEP", correct[:pair_count])):
        for item in values:
            source = by_id[str(item["problem_id"])] if decision == "REVISE" else item
            target = item["guided_output"] if decision == "REVISE" else "<decision>KEEP</decision>"
            selected.append({
                "construction_id": f"phase4_expansion::{source['problem_id']}::{decision}",
                "source_id": str(source["problem_id"]),
                "decision": decision,
                "messages": [
                    {"role": "system", "content": str(rollout["review_system_prompt"])},
                    {"role": "user", "content": str(source["task_prompt"])},
                    {"role": "assistant", "content": str(source["initial_output"])},
                    {"role": "user", "content": str(rollout["neutral_prompt"])},
                    {"role": "assistant", "content": target},
                ],
                "target_verified": True,
                "teacher_used": False,
                "candidate_index": item.get("candidate_index") if decision == "REVISE" else None,
            })
    dev_count = max(1, pair_count // 10)
    dev_ids = {
        row["source_id"]
        for decision in ("KEEP", "REVISE")
        for row in [value for value in selected if value["decision"] == decision][:dev_count]
    }
    selected.sort(key=lambda row: stable_rank("split", row["source_id"]))
    train = [row for row in selected if row["source_id"] not in dev_ids]
    dev = [row for row in selected if row["source_id"] in dev_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "attempts": write_jsonl(args.output_dir / "guided_attempts.jsonl", attempts),
        "train": write_jsonl(args.output_dir / "warmstart_train.jsonl", train),
        "dev": write_jsonl(args.output_dir / "warmstart_dev.jsonl", dev),
    }
    summary = {
        "initial_rows": len(rows),
        "wrong_initial_answers": len(wrong),
        "candidates_per_wrong": args.candidates_per_wrong,
        "guided_attempts": len(attempts),
        "passing_attempts": sum(row["correction_passed"] for row in attempts),
        "wrong_sources_with_verified_correction": len(verified),
        "rows_per_label": pair_count,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "sha256": hashes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initials", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates-per-wrong", type=int, default=4)
    parser.add_argument("--max-per-label", type=int, default=600)
    parser.add_argument("--minimum-per-label", type=int, default=200)
    parser.add_argument("--serving-model", default="phase4-actor")
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--concurrency", type=int, default=12)
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
