"""Danh gia xem model (sau SFT) co thuc su hoc duoc ky nang "biet no sai -> tu sua"
hay khong -- day la thuoc do CHINH cho muc tieu cua Phase 1, khong phai chi nhin
train_loss.

Chay tren mot bo problem HELD-OUT (KHONG nam trong tap da dung de train -- lay tiep
theo ngay sau offset da dung trong prepare_public_datasets.py), do 2 buoc:
  1. Model (co adapter) tu giai (attempt dau tien) -> verify khach quan.
  2. Neu SAI: dua chinh model prompt phan tu giong het luc train
     ("Ket qua kiem tra: SAI. Hay tu ra soat...") -> model tu sua -> verify lai.

Metric quan trong nhat: ty le tu sua thanh cong = self_corrected / initial_wrong.
Neu ty le nay thap (~0%), nghia la model chi hoc thuoc format (<thinking>, ### Phat
hien loi...) ma khong hoc duoc ky nang thuc su -- luc do KHONG nen scale du lieu len,
ma phai xem lai chat luong du lieu/cach train truoc.

YEU CAU PHAN CUNG: GPU compute capability >= 7.5 (xem ghi chu trong train_sft.py).

Chay tren may co GPU (dung dung adapter da train o outputs/phase1_lora):
    python -m src.evaluate_self_correction --math-offset 150 --math-limit 30 \
        --code-offset 150 --code-limit 30
"""
from __future__ import annotations

import argparse
import json
import re

# unsloth PHAI import truoc trl/transformers/peft/datasets (xem ghi chu trong
# train_sft.py) -- dat dau tien de tranh cac module con lai patch khong day du.
import unsloth  # noqa: F401

from datasets import load_dataset
from unsloth import FastLanguageModel

from src.config import load_config
from src.data.schema import Problem, VerifierResult
from src.generate_attempts import _build_prompt
from src.model_loading import load_model_and_tokenizer
from src.prepare_public_datasets import _extract_entry_point, _extract_gsm8k_answer
from src.verifier.code_verifier import CodeVerifier
from src.verifier.math_verifier import MathVerifier

# Phai giong het template trong build_dataset.py -- neu khong dong bo, model se gap
# prompt khac voi luc train, mat tac dung cua viec dua verifier_detail that vao.
_REFLECT_PROMPT_TEMPLATE = (
    "Kết quả kiểm tra: SAI.\n"
    "Chi tiết lỗi từ hệ thống kiểm tra: {verifier_detail}\n"
    "Hãy tự rà soát lại lời giải trên và sửa lại cho đúng."
)
_CORRECTION_RE = re.compile(r"###\s*Sửa lại\s*\n(.*)", flags=re.DOTALL)


def _load_held_out_problems(
    math_offset: int, math_limit: int, code_offset: int, code_limit: int
) -> list[Problem]:
    problems: list[Problem] = []

    # "gsm8k"/"mbpp" khong namespace la dataset script cu, huggingface_hub/datasets ban moi
    # tren server nay khong con resolve duoc -> dung id chinh tac moi (xem prepare_public_datasets.py
    # de biet chi tiet vi sao mbpp phai load truc tiep tu file parquet).
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

    # "Muennighoff/mbpp"/"google-research-datasets/mbpp" qua load_dataset(repo, split=...)
    # bi loi CastError vi ban parquet auto-convert cua HF gop nham 2 config (full/sanitized,
    # khac schema) lai voi nhau -> phai tro thang toi file parquet cua dung config "full/test".
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


def _append_log(log_path, **record) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-offset", type=int, default=150)
    parser.add_argument("--math-limit", type=int, default=30)
    parser.add_argument("--code-offset", type=int, default=150)
    parser.add_argument("--code-limit", type=int, default=30)
    parser.add_argument(
        "--log-file",
        type=str,
        default="outputs/eval_self_correction_log.jsonl",
        help="Ghi lai toan bo attempt_text/correction_text (bao gom <thinking>) de soat thu cong.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help=(
            "Nguon adapter: id repo tren HF Hub (vd Kxck/AGI_v1) hoac duong dan local. "
            "Mac dinh: doc tu paths.lora_out_dir trong phase1.yaml (thu muc train local)."
        ),
    )
    parser.add_argument(
        "--reflect-role",
        type=str,
        default="tool",
        choices=["user", "tool"],
        help=(
            "Role bao boc verifier_detail o buoc phan tu. Mac dinh 'tool' (da doi tu 'user' "
            "sau khi do thuc te qua paper 2606.05976 (The Self-Correction Illusion): tool giup "
            "ty le tu sua TOAN tang 0%% -> 33.3%% tren adapter cu (Kxck/AGI_v1, van con train "
            "voi role 'user'). build_dataset.py cung da doi sang 'tool' cho vong train tiep "
            "theo -- dung 'user' o day chi de doi chieu nguoc lai voi adapter cu neu can.",
        ),
    )
    args = parser.parse_args()

    cfg = load_config()
    adapter_dir = args.adapter or cfg.path("lora_out_dir")
    log_path = None
    if args.log_file:
        from pathlib import Path

        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    # Unsloth doc thang tu thu muc adapter da luu (tu tim base model qua
    # adapter_config.json), khong can load base + PeftModel rieng nhu truoc.
    model, tokenizer = load_model_and_tokenizer(
        cfg, cfg.small_model["max_seq_length"], model_name_override=str(adapter_dir)
    )
    FastLanguageModel.for_inference(model)

    verifiers = {
        "math": MathVerifier(),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }

    problems = _load_held_out_problems(
        args.math_offset, args.math_limit, args.code_offset, args.code_limit
    )
    gen_cfg = cfg.generation

    stats = {"total": 0, "initial_correct": 0, "initial_wrong": 0, "self_corrected": 0, "still_wrong": 0}

    for problem in problems:
        stats["total"] += 1
        verifier = verifiers[problem.domain]

        prompt = _build_prompt(problem)
        messages = [{"role": "user", "content": prompt}]
        attempt_text = _generate(model, tokenizer, messages, gen_cfg["max_new_tokens"])

        first_result = verifier.verify(problem, attempt_text)
        if first_result.passed:
            stats["initial_correct"] += 1
            print(f"[{problem.id}] initial: CORRECT")
            if log_path:
                _append_log(
                    log_path,
                    problem_id=problem.id,
                    domain=problem.domain,
                    attempt_text=attempt_text,
                    initial_passed=True,
                    verifier_detail=first_result.detail,
                )
            continue

        stats["initial_wrong"] += 1

        messages = messages + [
            {"role": "assistant", "content": attempt_text},
            {
                "role": args.reflect_role,
                "content": _REFLECT_PROMPT_TEMPLATE.format(verifier_detail=first_result.detail),
            },
        ]
        correction_max_new_tokens = gen_cfg.get("max_new_tokens_correction", gen_cfg["max_new_tokens"])
        correction_text = _generate(model, tokenizer, messages, correction_max_new_tokens)

        match = _CORRECTION_RE.search(correction_text)
        if match:
            corrected_solution = match.group(1).strip()
            second_result = verifier.verify(problem, corrected_solution)
        else:
            # Khong tim thay "### Sua lai" -> generation bi cat ngang giua chung (het
            # max_new_tokens) TRUOC KHI model hoan tat dinh dang, khong phai da co cau
            # tra loi cuoi cung. KHONG fallback dung ca correction_text tho de verify --
            # da phat hien thuc te fallback nay lam verifier vo tinh bat trung so/code
            # nhap con dang do dang (vd lay "so cuoi cung xuat hien trong text" hoac
            # "doan code Python dau tien" xuat hien som trong <thinking> chua hoan tat),
            # tao ra "tu sua thanh cong" gia. Coi thang la chua sua duoc.
            corrected_solution = None
            second_result = VerifierResult(
                passed=False,
                detail="Chua hoan tat dinh dang '### Sua lai' (co the bi cat vi het max_new_tokens)",
            )

        if second_result.passed:
            stats["self_corrected"] += 1
            print(f"[{problem.id}] initial: WRONG -> self-corrected: CORRECT")
        else:
            stats["still_wrong"] += 1
            print(f"[{problem.id}] initial: WRONG -> self-corrected: STILL WRONG ({second_result.detail})")

        if log_path:
            _append_log(
                log_path,
                problem_id=problem.id,
                domain=problem.domain,
                attempt_text=attempt_text,
                initial_passed=False,
                verifier_detail=first_result.detail,
                correction_text=correction_text,
                corrected_solution=corrected_solution,
                format_incomplete=match is None,
                second_passed=second_result.passed,
                second_detail=second_result.detail,
                reflect_role=args.reflect_role,
            )

    print("\n=== KET QUA DANH GIA ===")
    print(stats)
    if stats["initial_wrong"] > 0:
        rate = stats["self_corrected"] / stats["initial_wrong"] * 100
        print(f"Ty le tu sua thanh cong (tren cac case sai luc dau): {rate:.1f}%")


if __name__ == "__main__":
    main()
