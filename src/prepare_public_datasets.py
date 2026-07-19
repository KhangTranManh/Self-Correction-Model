"""Tai bo du lieu chuan GSM8K (toan) va MBPP (code) tu HuggingFace datasets,
convert sang dinh dang JSONL cua project -- ghi de data/problems/math.jsonl
va data/problems/code.jsonl (bo mau 3+3 truoc do qua de, model lam dung het).

Chay (khong can GPU, chi can internet de tai dataset):
    python -m src.prepare_public_datasets --math-limit 50 --code-limit 50
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent


def _extract_gsm8k_answer(answer_field: str) -> str:
    # GSM8K answer format: "... reasoning ... #### 42"
    match = re.search(r"####\s*([\-0-9,.]+)", answer_field)
    if not match:
        raise ValueError(f"Khong tim thay dap so cuoi trong: {answer_field!r}")
    return match.group(1).replace(",", "")


def _extract_entry_point(code: str) -> str | None:
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else None


def prepare_math(limit: int, out_path: Path) -> None:
    # "gsm8k" (khong namespace) la dataset script cu, huggingface_hub/datasets ban moi
    # khong con resolve duoc -> phai dung id chinh tac moi tren Hub (cung field question/answer).
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(limit, len(ds))))

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            record = {
                "id": f"gsm8k_{i:04d}",
                "question": row["question"],
                "reference_answer": _extract_gsm8k_answer(row["answer"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Da ghi {len(ds)} bai toan (GSM8K) vao {out_path}")


def prepare_code(limit: int, out_path: Path) -> None:
    # "mbpp" (khong namespace) cung la dataset script cu, cung ly do nhu gsm8k o tren.
    # load_dataset("Muennighoff/mbpp", split="test") bi CastError vi ban parquet auto-convert
    # cua HF gop nham 2 config (full/sanitized, khac schema) -> tro thang toi file parquet
    # cua dung config "full/test" (giu nguyen field task_id/text/code/test_list/test_setup_code).
    ds = load_dataset(
        "parquet",
        data_files=(
            "hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/full/test/0000.parquet"
        ),
        split="train",
    )
    ds = ds.select(range(min(limit, len(ds))))

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            entry_point = _extract_entry_point(row["code"])
            if entry_point is None:
                continue  # bo qua neu khong parse duoc ten ham tu code mau

            tests = list(row["test_list"])
            if row.get("test_setup_code"):
                tests = [row["test_setup_code"]] + tests

            record = {
                "id": f"mbpp_{row['task_id']}",
                "question": row["text"],
                "entry_point": entry_point,
                "tests": tests,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Da ghi {written} bai code (MBPP) vao {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-limit", type=int, default=50, help="So bai GSM8K se lay")
    parser.add_argument("--code-limit", type=int, default=50, help="So bai MBPP se lay")
    args = parser.parse_args()

    prepare_math(args.math_limit, ROOT_DIR / "data" / "problems" / "math.jsonl")
    prepare_code(args.code_limit, ROOT_DIR / "data" / "problems" / "code.jsonl")


if __name__ == "__main__":
    main()
