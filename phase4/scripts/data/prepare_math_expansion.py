"""Prepare an additional source-disjoint GSM8K training-only manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260914
ANSWER_RE = re.compile(r"####\s*([\-0-9,.]+)")
CALC_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def stable_rank(index: int) -> str:
    return hashlib.sha256(f"{SEED}:gsm8k:{index}".encode()).hexdigest()


def existing_problem_hashes() -> set[str]:
    hashes: set[str] = set()
    for phase in (PROJECT_ROOT / "phase1", PROJECT_ROOT / "phase3"):
        for path in phase.rglob("*.jsonl"):
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument(
        "--skip",
        type=int,
        default=800,
        help="Skip the deterministic rows allocated to the original Phase 4 pilot.",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty {args.output_dir}")

    blocked = existing_problem_hashes()
    eligible = []
    for index, source in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
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
    selected = eligible[args.skip : args.skip + args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"Requested {args.count} rows after offset {args.skip}, got {len(selected)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "train_problems.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    manifest.write_text(payload, encoding="utf-8", newline="\n")
    report = {
        "schema_version": "phase4_math_expansion_sources_v1",
        "seed": SEED,
        "upstream": "openai/gsm8k/main/train",
        "skip": args.skip,
        "rows": len(selected),
        "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_disjoint_from_phase1_phase3": True,
        "disjoint_from_original_phase4_allocation_by_deterministic_offset": True,
        "confirmation_candidates_read": False,
    }
    (args.output_dir / "source_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
