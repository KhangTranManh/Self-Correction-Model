"""Collect hint-free reviews and build source-disjoint, real-output preferences."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))
from phase4.lib.transition import parse_review
from phase4.scripts.collect_warmstart import read_jsonl, verify, write_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_complete_jsonl_bytes(payload: bytes, path: Path):
    """Parse complete JSONL lines without treating a final partial line as data."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid UTF-8 before the final audit tail in {path}") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed complete audit line at {path}:{line_number}") from error
    return rows


def read_audit_with_final_tail_recovery(audit_path: Path):
    """Recover only a crash-truncated final append, retaining its raw evidence."""
    try:
        return read_jsonl(audit_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as original_error:
        payload = audit_path.read_bytes()
        # A malformed newline-terminated record, or any malformed preceding line,
        # is evidence of corruption rather than an interrupted append.
        if payload.endswith((b"\n", b"\r")):
            raise ValueError(f"Malformed newline-terminated audit record in {audit_path}") from original_error
        final_newline = payload.rfind(b"\n")
        complete_payload = payload[: final_newline + 1] if final_newline >= 0 else b""
        tail = payload[final_newline + 1 :]
        if not tail.strip():
            raise ValueError(f"Unreadable audit content in {audit_path}") from original_error
        # This raises for interior corruption, so only the final unterminated
        # fragment can be removed.
        rows = _read_complete_jsonl_bytes(complete_payload, audit_path)
        tail_path = audit_path.with_suffix(audit_path.suffix + ".incomplete-tail")
        if tail_path.exists() and tail_path.read_bytes() != tail:
            raise ValueError(
                f"Refusing to replace existing forensic tail {tail_path}; preserve it first"
            )
        if not tail_path.exists():
            with tail_path.open("xb") as handle:
                handle.write(tail)
                handle.flush()
                os.fsync(handle.fileno())
        temporary = audit_path.with_suffix(audit_path.suffix + ".recovery.tmp")
        with temporary.open("wb") as handle:
            handle.write(complete_payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(audit_path)
        print(f"RECOVERED final incomplete audit tail to {tail_path}", flush=True)
        return rows


def rank(problem_id: str) -> str:
    return hashlib.sha256(f"20260915:blind:{problem_id}".encode()).hexdigest()


def build_pairs(sources, attempts, output_dir: Path, minimum_train: int, minimum_dev: int):
    by_id = defaultdict(list)
    for row in attempts:
        by_id[row["problem_id"]].append(row)
    selected = {"train": [], "dev": []}
    availability = {}
    for split in selected:
        labels = {"KEEP": [], "REVISE": []}
        for source in sources:
            if source["split"] != split:
                continue
            values = sorted(by_id[source["problem_id"]], key=lambda x: x["candidate_index"])
            valid = [x for x in values if x["contract_valid"]]
            if source["initial_correct"]:
                chosen = [x for x in valid if x["decision"] == "KEEP"]
                rejected = [x for x in valid if x["decision"] == "REVISE" and not x["final_correct"]]
                decision, reason = "KEEP", "correct_keep_over_real_harmful_revision"
            else:
                chosen = [x for x in valid if x["decision"] == "REVISE" and x["final_correct"]]
                # Prefer a failed revision: this supervises correction quality rather than just action choice.
                rejected = [x for x in valid if x["decision"] == "REVISE" and not x["final_correct"]]
                reason = "verified_fix_over_real_failed_revision"
                if not rejected:
                    rejected = [x for x in valid if x["decision"] == "KEEP"]
                    reason = "verified_fix_over_real_wrong_keep"
                decision = "REVISE"
            if not chosen or not rejected:
                continue
            good, bad = chosen[0], rejected[0]
            labels[decision].append({
                "pair_id": f"blind::{source['problem_id']}",
                "source_id": source["problem_id"],
                "decision": decision,
                "preference_reason": reason,
                "prompt": source["messages"],
                "chosen": [{"role": "assistant", "content": good["output"]}],
                "rejected": [{"role": "assistant", "content": bad["output"]}],
                "chosen_candidate_index": good["candidate_index"],
                "rejected_candidate_index": bad["candidate_index"],
                "model_visible_verifier_hints": False,
            })
        availability[split] = {label: len(values) for label, values in labels.items()}
        count = min(map(len, labels.values()))
        for values in labels.values():
            values.sort(key=lambda x: rank(x["source_id"]))
            selected[split].extend(values[:count])
        selected[split].sort(key=lambda x: rank(x["source_id"]))
    assert not ({x["source_id"] for x in selected["train"]} & {x["source_id"] for x in selected["dev"]})
    hashes = {split: write_jsonl(output_dir / f"{split}.jsonl", rows) for split, rows in selected.items()}
    ready = len(selected["train"]) >= 2 * minimum_train and len(selected["dev"]) >= 2 * minimum_dev
    summary = {
        "schema_version": "phase4_blind_preferences_v1",
        "sources": len(sources), "attempts": len(attempts),
        "decision_counts": dict(Counter(x["decision"] for x in attempts)),
        "verified_fixes": sum(not x["initial_correct"] and x["contract_valid"] and x["final_correct"] for x in attempts),
        "harmful_revisions": sum(x["initial_correct"] and x["contract_valid"] and not x["final_correct"] for x in attempts),
        "available_pairs": availability,
        "train_rows": len(selected["train"]), "dev_rows": len(selected["dev"]),
        "minimum_train_per_action": minimum_train, "minimum_dev_per_action": minimum_dev,
        "training_gate_passed": ready, "sha256": hashes,
        "model_visible_verifier_hints": False, "synthetic_responses": False,
        "confirmation_candidates_read": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


async def run(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rollout = config["rollout"]
    initials = {x["problem_id"]: x for x in read_jsonl(args.initials)}
    sources, seen = [], set()
    for split, path in (("train", args.train_sources), ("dev", args.dev_sources)):
        for row in read_jsonl(path):
            problem_id = row["source_id"]
            if problem_id in seen:
                raise ValueError(f"Duplicate or overlapping source: {problem_id}")
            seen.add(problem_id)
            source = initials[problem_id]
            fresh_correct = await asyncio.to_thread(verify, source, source["initial_output"])
            if fresh_correct != bool(source["initial_correct"]):
                raise ValueError(f"Cached initial label changed: {problem_id}")
            sources.append({**source, "split": split, "messages": [
                {"role": "system", "content": str(rollout["review_system_prompt"])},
                {"role": "user", "content": source["task_prompt"]},
                {"role": "assistant", "content": source["initial_output"]},
                {"role": "user", "content": str(rollout["neutral_prompt"])},
            ]})
    sources.sort(key=lambda x: x["problem_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.serving_model, "candidates": args.candidates,
                "temperature": args.temperature, "max_tokens": args.max_tokens, "seed": args.seed,
                "initials_sha256": sha256(args.initials), "train_sources_sha256": sha256(args.train_sources),
                "dev_sources_sha256": sha256(args.dev_sources), "config_sha256": sha256(args.config)}
    manifest_path = args.output_dir / "collection_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Refusing to resume a collection with different settings or sources")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_path = args.output_dir / "blind_attempts.jsonl"
    # Keep manifest validation above recovery: an invocation with changed
    # settings must never rewrite the audit it is incompatible with.
    attempts = read_audit_with_final_tail_recovery(audit_path) if audit_path.exists() else []
    done = {(x["problem_id"], x["candidate_index"]) for x in attempts}
    if len(done) != len(attempts):
        raise ValueError("Audit has duplicate candidates")
    pending = [(index, source, candidate) for index, source in enumerate(sources)
               for candidate in range(args.candidates) if (source["problem_id"], candidate) not in done]
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/") + "/", timeout=httpx.Timeout(600, connect=30)) as client:
        async def generate(index, source, candidate):
            payload = {"model": args.serving_model, "messages": source["messages"],
                       "temperature": args.temperature, "max_tokens": args.max_tokens,
                       "seed": args.seed + index * 100 + candidate}
            async with semaphore:
                for retry in range(3):
                    try:
                        response = await client.post("chat/completions", json=payload)
                        response.raise_for_status()
                        break
                    except (httpx.HTTPError, TimeoutError):
                        if retry == 2:
                            raise
                        await asyncio.sleep(2 * (retry + 1))
            choice = response.json()["choices"][0]
            output = str(choice["message"]["content"])
            action = parse_review(output)
            final_correct = False
            if action.valid:
                final_correct = source["initial_correct"] if action.decision == "KEEP" else await asyncio.to_thread(verify, source, action.revised_answer or "")
            return {"problem_id": source["problem_id"], "split": source["split"],
                    "candidate_index": candidate, "seed": payload["seed"], "output": output,
                    "decision": action.decision, "contract_valid": action.valid,
                    "initial_correct": source["initial_correct"], "final_correct": bool(final_correct),
                    "finish_reason": choice.get("finish_reason"), "model_visible_verifier_hints": False}

        work_queue = asyncio.Queue()
        event_queue = asyncio.Queue()
        for item in pending:
            work_queue.put_nowait(item)
        for _ in range(args.concurrency):
            work_queue.put_nowait(None)

        async def worker():
            while True:
                item = await work_queue.get()
                try:
                    if item is None:
                        return
                    index, source, candidate = item
                    try:
                        value = await generate(index, source, candidate)
                    except Exception as error:
                        await event_queue.put(("error", index, candidate, error))
                    else:
                        await event_queue.put(("result", value))
                finally:
                    work_queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        failures = []
        persisted_this_run = 0
        with audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            for _ in pending:
                event = await event_queue.get()
                if event[0] == "error":
                    failures.append(event)
                    continue
                row = event[1]
                # Persist each completion before accepting the next event, so an
                # interrupted continuous run can resume without regeneration.
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                attempts.append(row)
                persisted_this_run += 1
                if persisted_this_run % 64 == 0:
                    print(f"SAVED {len(attempts)}/{len(sources) * args.candidates} blind candidates", flush=True)
        await asyncio.gather(*workers)
        print(f"SAVED {len(attempts)}/{len(sources) * args.candidates} blind candidates", flush=True)
        if failures:
            _, index, candidate, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} blind review generation failures; completed records were saved "
                f"(first at source index {index}, candidate {candidate})"
            ) from error
    build_pairs(sources, attempts, args.output_dir, args.minimum_train_per_action, args.minimum_dev_per_action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initials", type=Path, required=True)
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--dev-sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "phase4/configs/transition_rl_v1.yaml")
    parser.add_argument("--serving-model", default="phase4-actor")
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=6000000)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--minimum-train-per-action", type=int, default=20)
    parser.add_argument("--minimum-dev-per-action", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.candidates < 100 or args.concurrency < 1:
        parser.error("Candidate count must be 1..99 and concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
