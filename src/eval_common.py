"""Phan dung chung cho 2 script eval (ban Unsloth va ban vLLM).

Ca 2 phai do TREN CUNG bo problem held-out va ghi CUNG dinh dang log, neu khong thi
so lieu 2 ban khong so sanh duoc voi nhau -- ma so sanh duoc chinh la ly do giu ca
2 ban. Nhu src/prompts.py, file nay khong import vllm/unsloth de ca 2 venv deu dung
duoc; no CO import datasets (can de tai GSM8K/MBPP).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.data.schema import Problem
from src.prepare_public_datasets import _extract_entry_point, _extract_gsm8k_answer


def load_held_out_problems(
    math_offset: int, math_limit: int, code_offset: int, code_limit: int
) -> list[Problem]:
    """Held-out = lay tu offset TRO DI tren split "test".

    prepare_public_datasets.py luon cat tu index 0, nen chi can offset qua khoi so
    luong da dung de train la dam bao khong trung -- khong can file split rieng.
    """
    problems: list[Problem] = []

    # "gsm8k"/"mbpp" khong namespace la dataset script cu, huggingface_hub/datasets ban
    # moi khong con resolve duoc -> dung id chinh tac moi.
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    end = min(math_offset + math_limit, len(gsm8k))
    for i in range(math_offset, end):
        row = gsm8k[i]
        problems.append(
            Problem(
                id=f"eval_gsm8k_{i:04d}",
                domain="math",
                question=row["question"],
                reference_answer=_extract_gsm8k_answer(row["answer"]),
            )
        )

    # Qua load_dataset(repo, split=...) se loi CastError: ban parquet auto-convert cua
    # HF gop nham 2 config (full/sanitized, khac schema) -> tro thang toi file parquet
    # cua dung config "full/test".
    mbpp = load_dataset(
        "parquet",
        data_files=(
            "hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/full/test/0000.parquet"
        ),
        split="train",
    )
    end = min(code_offset + code_limit, len(mbpp))
    for i in range(code_offset, end):
        row = mbpp[i]
        entry_point = _extract_entry_point(row["code"])
        if entry_point is None:
            continue
        tests = list(row["test_list"])
        if row.get("test_setup_code"):
            tests = [row["test_setup_code"]] + tests
        problems.append(
            Problem(
                id=f"eval_mbpp_{row['task_id']}",
                domain="code",
                question=row["text"],
                entry_point=entry_point,
                tests=tests,
            )
        )

    return problems


def open_log(log_file: str | None) -> Path | None:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    return log_path


def append_log(log_path: Path | None, **record) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_report(records: list[dict]) -> None:
    """Breakdown theo domain + tong.

    format_incomplete duoc tach RIENG khoi genuine_still_wrong: mot ban bi cat ngang
    giua <thinking> (chua toi "### Sua lai") khac han ve ban chat voi mot ban da viet
    xong dinh dang nhung sua sai. Gop 2 loai lai se doc sai nguyen nhan that.
    """
    domains = ["math", "code"]
    stats = {d: defaultdict(int) for d in domains}
    stats["total"] = defaultdict(int)

    for r in records:
        for key in (r["domain"], "total"):
            s = stats[key]
            s["n"] += 1
            if r.get("initial_passed"):
                s["initial_correct"] += 1
            else:
                s["initial_wrong"] += 1
                if r.get("format_incomplete"):
                    s["format_incomplete"] += 1
                elif r.get("second_passed"):
                    s["self_corrected"] += 1
                else:
                    s["genuine_still_wrong"] += 1

    hdr = (
        f"{'domain':<8}{'n':>4}{'init_ok':>9}{'init_wrong':>12}"
        f"{'fmt_incompl':>13}{'self_corr':>11}{'still_wrong':>13}"
    )
    print("\n=== KET QUA DANH GIA ===")
    print(hdr)
    print("-" * len(hdr))
    for key in domains + ["total"]:
        s = stats[key]
        print(
            f"{key:<8}{s['n']:>4}{s['initial_correct']:>9}{s['initial_wrong']:>12}"
            f"{s['format_incomplete']:>13}{s['self_corrected']:>11}{s['genuine_still_wrong']:>13}"
        )

    print()
    for key in domains + ["total"]:
        s = stats[key]
        if s["initial_wrong"]:
            rate = s["self_corrected"] / s["initial_wrong"] * 100
            print(
                f"Ty le tu sua THAT [{key:<5}]: {rate:5.1f}%"
                f"   ({s['self_corrected']}/{s['initial_wrong']})"
            )
