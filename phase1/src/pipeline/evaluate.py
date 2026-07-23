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
    python -m src.pipeline.evaluate --math-offset 150 --math-limit 30 \
        --code-offset 150 --code-limit 30
"""
from __future__ import annotations

import argparse

# unsloth PHAI import truoc trl/transformers/peft/datasets (xem ghi chu trong
# train_sft.py) -- dat dau tien de tranh cac module con lai patch khong day du.
import unsloth  # noqa: F401

from unsloth import FastLanguageModel

from src.config import load_config
from src.core.schema import VerifierResult
# Bo de held-out + dinh dang log dung chung voi ban vLLM -- neu moi ban tu tai de
# rieng thi 2 con so khong con so sanh duoc voi nhau.
from src.pipeline.eval_common import append_log, load_held_out_problems, open_log, print_report
from src.core.model_loading import load_model_and_tokenizer
from src.data.verifiers.code import CodeVerifier
from src.data.verifiers.math import MathVerifier

# Lay tu src/prompts.py -- KHONG khai bao lai o day. build_dataset.py (luc train) va
# file nay (luc eval) doc cung mot hang so, nen chung khong the lech nhau nua.
# Truoc day _build_prompt duoc import tu generate_attempts.py, ma file do import vllm
# o top-level -> ep moi truong chay eval (Unsloth) phai cai ca vLLM. Doi sang
# src.core.prompts (khong co dependency nang) de 2 venv doc lap hoan toan.
from src.core.prompts import CORRECTION_RE as _CORRECTION_RE
from src.core.prompts import REFLECT_ROLES, build_reflect_message
from src.core.prompts import build_prompt as _build_prompt



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
        choices=list(REFLECT_ROLES),
        help=(
            "Role bao boc verifier_detail o buoc phan tu. Mac dinh 'tool' (da doi tu 'user' "
            "sau khi do thuc te qua paper 2606.05976 (The Self-Correction Illusion): tool giup "
            "ty le tu sua TOAN tang 0%% -> 33.3%% tren adapter cu (Kxck/AGI_v1, van con train "
            "voi role 'user'). build_dataset.py cung da doi sang 'tool' cho vong train tiep "
            "theo -- dung 'user' o day chi de doi chieu nguoc lai voi adapter cu neu can.",
        ),
    )
    parser.add_argument(
        "--math-localize-steps", action="store_true",
        help="Xem giai thich cung ten trong evaluate_vllm.py: verifier_detail cua "
             "TOAN chi ra buoc tinh nghi van thay vi noi thang dap so dung.",
    )
    args = parser.parse_args()

    cfg = load_config()
    adapter_dir = args.adapter or cfg.path("lora_out_dir")
    log_path = open_log(args.log_file)

    # Unsloth doc thang tu thu muc adapter da luu (tu tim base model qua
    # adapter_config.json), khong can load base + PeftModel rieng nhu truoc.
    model, tokenizer = load_model_and_tokenizer(
        cfg, cfg.small_model["max_seq_length"], model_name_override=str(adapter_dir)
    )
    FastLanguageModel.for_inference(model)

    verifiers = {
        "math": MathVerifier(localize_steps=args.math_localize_steps),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }

    problems = load_held_out_problems(
        args.math_offset, args.math_limit, args.code_offset, args.code_limit
    )
    gen_cfg = cfg.generation

    records: list[dict] = []

    for problem in problems:
        verifier = verifiers[problem.domain]

        prompt = _build_prompt(problem)
        messages = [{"role": "user", "content": prompt}]
        attempt_text = _generate(model, tokenizer, messages, gen_cfg["max_new_tokens"])

        first_result = verifier.verify(problem, attempt_text)
        if first_result.passed:
            print(f"[{problem.id}] initial: CORRECT")
            rec = dict(
                problem_id=problem.id,
                domain=problem.domain,
                attempt_text=attempt_text,
                initial_passed=True,
                verifier_detail=first_result.detail,
            )
            records.append(rec)
            append_log(log_path, **rec)
            continue


        messages = messages + [
            {"role": "assistant", "content": attempt_text},
            build_reflect_message(first_result.detail, args.reflect_role),
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
            print(f"[{problem.id}] initial: WRONG -> self-corrected: CORRECT")
        else:
            print(f"[{problem.id}] initial: WRONG -> self-corrected: STILL WRONG ({second_result.detail})")

        rec = dict(
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
            math_localize_steps=args.math_localize_steps,
        )
        records.append(rec)
        append_log(log_path, **rec)

    print_report(records)
    if log_path:
        print(f"\nLog chi tiet: {log_path}")


if __name__ == "__main__":
    main()
