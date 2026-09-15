"""Prepare source-disjoint GSM8K problems for the Phase 4 pipeline pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260914
ANSWER_RE = re.compile(r"####\s*([\-0-9,.]+)")
CALC_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def stable_rank(index: int) -> str:
    return hashlib.sha256(f"{SEED}:gsm8k:{index}".encode()).hexdigest()


def existing_problem_hashes() -> set[str]:
    hashes: set[str] = set()
    for phase in (PROJECT_ROOT / "phase1", PROJECT_ROOT / "phase3"):
        for path in phase.rglob("*.jsonl"):
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    for key in ("problem", "question"):
                        value = row.get(key)
                        if isinstance(value, str) and value.strip():
                            hashes.add(text_hash(value))
    return hashes


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "phase4/data/pilot_v1")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--confirmation-candidates", type=int, default=400)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty {args.output_dir}")

    blocked = existing_problem_hashes()
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    eligible: list[dict[str, Any]] = []
    for index, source in enumerate(dataset):
        question = str(source["question"])
        if text_hash(question) in blocked:
            continue
        raw_answer = str(source["answer"])
        match = ANSWER_RE.search(raw_answer)
        if not match:
            continue
        eligible.append({
            "id": f"phase4_gsm8k_train_{index:04d}",
            "dataset": "gsm8k",
            "source_split": "train",
            "source_index": index,
            "domain": "math",
            "question": question,
            "reference_answer": match.group(1).replace(",", ""),
            "calc_steps": [
                [expression.strip(), value.strip()]
                for expression, value in CALC_RE.findall(raw_answer)
            ],
        })
    eligible.sort(key=lambda row: (stable_rank(int(row["source_index"])), row["id"]))
    requested = args.train + args.dev + args.confirmation_candidates
    if len(eligible) < requested:
        raise RuntimeError(f"Only {len(eligible)} source-disjoint rows; need {requested}")

    splits = {
        "train_problems.jsonl": eligible[: args.train],
        "dev_problems.jsonl": eligible[args.train : args.train + args.dev],
        "confirmation_candidates.jsonl": eligible[
            args.train + args.dev : requested
        ],
    }
    hashes = {
        name: write_jsonl(args.output_dir / name, rows) for name, rows in splits.items()
    }
    selected_hashes = [text_hash(row["question"]) for rows in splits.values() for row in rows]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise RuntimeError("Phase 4 pilot split overlap detected")
    report = {
        "schema_version": "phase4_math_pilot_sources_v1",
        "seed": SEED,
        "upstream": "openai/gsm8k/main/train",
        "existing_phase1_phase3_problem_hashes": len(blocked),
        "eligible_source_disjoint_rows": len(eligible),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "sha256": hashes,
        "limitations": [
            "pipeline pilot is math-only",
            "confirmation candidates are not the final sealed balanced benchmark",
        ],
        "validation": {
            "source_disjoint_from_committed_phase1_phase3_jsonl": True,
            "split_disjoint": True,
            "natural_model_outputs_only": True,
        },
    }
    (args.output_dir / "source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
