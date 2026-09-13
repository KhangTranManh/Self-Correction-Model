"""Build the Phase 3 source pool without touching any frozen P0 split.

The resulting pool contains 1,000 GSM8K train problems and every MBPP problem
from the non-test splits (374 train + 90 validation + 10 prompt = 474).  P0 uses
GSM8K test and MBPP full/test, so these 1,474 source problems are disjoint by
construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase3.yaml"
MBPP_PARQUET = (
    "hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/"
    "full/{split}/0000.parquet"
)
_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-0-9,.]+)")
_CALC_STEP_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
_ENTRY_POINT_RE = re.compile(r"def\s+(\w+)\s*\(")


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _gsm8k_answer(answer: str) -> str:
    match = _GSM8K_ANSWER_RE.search(answer)
    if not match:
        raise ValueError(f"GSM8K row has no final answer marker: {answer!r}")
    return match.group(1).replace(",", "")


def _gsm8k_rows(limit: int) -> list[dict]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    rows: list[dict] = []
    for index, row in enumerate(dataset.select(range(min(limit, len(dataset))))):
        raw_answer = row["answer"]
        rows.append(
            {
                "id": f"gsm8k_train_{index:04d}",
                "dataset": "gsm8k",
                "split": "train",
                "source_index": index,
                "domain": "math",
                "problem": row["question"],
                "ground_truth": _gsm8k_answer(raw_answer),
                "reference_solution": raw_answer,
                "reference_answer": _gsm8k_answer(raw_answer),
                "calc_steps": [
                    [expression.strip(), value.strip()]
                    for expression, value in _CALC_STEP_RE.findall(raw_answer)
                ],
            }
        )
    return rows


def _mbpp_rows(split_limits: dict[str, int]) -> list[dict]:
    if set(split_limits) - {"train", "validation", "prompt"}:
        raise ValueError("Only MBPP train/validation/prompt splits are allowed")

    rows: list[dict] = []
    for split, limit in split_limits.items():
        dataset = load_dataset(
            "parquet",
            data_files=MBPP_PARQUET.format(split=split),
            split="train",
        )
        for source_index, row in enumerate(
            dataset.select(range(min(int(limit), len(dataset))))
        ):
            match = _ENTRY_POINT_RE.search(row["code"])
            if not match:
                raise ValueError(
                    f"Cannot extract entry point for MBPP {split} row {source_index}"
                )
            tests = list(row["test_list"])
            if row.get("test_setup_code"):
                tests.insert(0, row["test_setup_code"])
            rows.append(
                {
                    "id": f"mbpp_{split}_{row['task_id']}",
                    "dataset": "mbpp",
                    "split": split,
                    "source_index": source_index,
                    "task_id": int(row["task_id"]),
                    "domain": "code",
                    "problem": row["text"],
                    "ground_truth": row["code"],
                    "entry_point": match.group(1),
                    "tests": tests,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing source pool"
    )
    args = parser.parse_args()

    config = _load_config()
    output_path = _resolve(config["paths"]["source_problems"])
    manifest_path = _resolve(config["paths"]["source_manifest"])
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it")

    rows = _gsm8k_rows(int(config["dataset"]["gsm8k_train_limit"]))
    rows.extend(_mbpp_rows(config["dataset"]["mbpp_splits"]))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate source IDs detected")
    if any(row["dataset"] == "gsm8k" and row["split"] != "train" for row in rows):
        raise RuntimeError("GSM8K non-train row entered the Phase 3 source pool")
    if any(row["dataset"] == "mbpp" and row["split"] == "test" for row in rows):
        raise RuntimeError("MBPP test row entered the Phase 3 source pool")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    output_path.write_text(payload, encoding="utf-8", newline="\n")
    counts = {
        "gsm8k": sum(row["dataset"] == "gsm8k" for row in rows),
        "mbpp": sum(row["dataset"] == "mbpp" for row in rows),
    }
    manifest = {
        "total": len(rows),
        "counts": counts,
        "p0_overlap_guard": {
            "gsm8k_allowed_splits": ["train"],
            "mbpp_allowed_splits": ["train", "validation", "prompt"],
            "frozen_p0_splits_excluded": ["gsm8k/test", "mbpp/full/test"],
        },
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_file": str(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
