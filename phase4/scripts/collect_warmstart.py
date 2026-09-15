"""Build a verified exploration warm-start from Cycle 0 natural failures."""

from __future__ import annotations

import argparse
import asyncio
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
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
from src.data.verifiers.code import CodeVerifier  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def stable_rank(label: str, problem_id: str) -> str:
    return hashlib.sha256(f"20260914:{label}:{problem_id}".encode()).hexdigest()


def verify(row: dict[str, Any], answer: str) -> bool:
    spec = row["verifier_spec"]
    problem = Problem(
        id=str(row["problem_id"]),
        domain=str(row["domain"]),
        question=str(row["task_prompt"]),
        reference_answer=spec.get("reference_answer"),
        entry_point=spec.get("entry_point"),
        tests=spec.get("tests", []),
        calc_steps=spec.get("calc_steps"),
    )
    verifier = MathVerifier() if problem.domain == "math" else CodeVerifier()
    return bool(verifier.verify(problem, answer).passed)


async def correct_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    system_prompt: str,
    index: int,
) -> dict[str, Any]:
    feedback = (
        "The deterministic verifier rejected the previous answer and returned this evidence:\n"
        f"{row['initial_verifier_detail']}\n"
        "Return only <decision>REVISE</decision><answer>COMPLETE CORRECTED ANSWER</answer>."
    )
    payload = {
        "model": "phase4-actor",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["task_prompt"]},
            {"role": "assistant", "content": row["initial_output"]},
            {"role": "user", "content": feedback},
        ],
        "temperature": 0.7,
        "max_tokens": 1200,
        "seed": 3_000_000 + index,
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
        "guided_prompt": feedback,
        "guided_output": output,
        "contract_valid": action.valid,
        "correction_passed": passed,
    }


async def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rollout = config["rollout"]
    rows = read_jsonl(args.rollouts)
    wrong = [row for row in rows if not row["initial_correct"]]
    by_id = {str(row["problem_id"]): row for row in rows}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/") + "/", timeout=timeout) as client:
        guided = await asyncio.gather(*[
            correct_one(client, semaphore, row, str(rollout["review_system_prompt"]), index)
            for index, row in enumerate(wrong)
        ])
    guided.sort(key=lambda row: row["problem_id"])
    passing = [row for row in guided if row["correction_passed"]]
    correct = [row for row in rows if row["initial_correct"]]
    pair_count = min(len(passing), len(correct), args.max_per_label)
    if pair_count < args.minimum_per_label:
        raise RuntimeError(f"Only {pair_count} verified examples per label; need {args.minimum_per_label}")
    passing.sort(key=lambda row: stable_rank("REVISE", str(row["problem_id"])))
    correct.sort(key=lambda row: stable_rank("KEEP", str(row["problem_id"])))
    selected: list[dict[str, Any]] = []
    for decision, values in (("REVISE", passing[:pair_count]), ("KEEP", correct[:pair_count])):
        for item in values:
            source = by_id[str(item["problem_id"])] if decision == "REVISE" else item
            target = item["guided_output"] if decision == "REVISE" else "<decision>KEEP</decision>"
            selected.append({
                "construction_id": f"phase4_warmstart::{source['problem_id']}::{decision}",
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
                "guided_feedback_used_for_target_generation": decision == "REVISE",
            })
    selected.sort(key=lambda row: stable_rank("split", row["source_id"]))
    dev_ids = {
        row["source_id"]
        for decision in ("KEEP", "REVISE")
        for row in [value for value in selected if value["decision"] == decision][: max(1, pair_count // 5)]
    }
    train = [row for row in selected if row["source_id"] not in dev_ids]
    dev = [row for row in selected if row["source_id"] in dev_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "guided": write_jsonl(args.output_dir / "guided_attempts.jsonl", guided),
        "train": write_jsonl(args.output_dir / "warmstart_train.jsonl", train),
        "dev": write_jsonl(args.output_dir / "warmstart_dev.jsonl", dev),
    }
    summary = {
        "wrong_initial_answers": len(wrong),
        "verified_guided_corrections": len(passing),
        "rows_per_label": pair_count,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "sha256": hashes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-label", type=int, default=80)
    parser.add_argument("--minimum-per-label", type=int, default=40)
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "phase4/configs/transition_rl_v1.yaml",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
