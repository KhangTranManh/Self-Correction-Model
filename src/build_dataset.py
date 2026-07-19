"""Orchestration Phase 1:
  attempts.jsonl (that, tu generate_attempts.py)
    -> verifier khach quan (math/code)
    -> neu SAI: goi DeepSeek sinh critique + correction
    -> verifier lai correction
    -> chi giu mau neu correction DA DUOC XAC NHAN DUNG
    -> ghi ra data/processed/phase1_sft.jsonl (dinh dang chat/ChatML)

Chay (khong can GPU, chi can .env co DEEPSEEK_API_KEY/DEEPSEEK_MODEL va da co attempts.jsonl):
    python -m src.build_dataset
"""
from __future__ import annotations

import json

from src.config import load_config
from src.data.problem_sources import load_code_problems, load_math_problems
from src.data.schema import Problem
from src.deepseek_client import DeepSeekClient
from src.verifier.code_verifier import CodeVerifier
from src.verifier.math_verifier import MathVerifier

_SYSTEM_PROMPT = "Bạn luôn kiểm tra lại lời giải của mình trước khi chốt câu trả lời cuối."
# QUAN TRONG: phai co verifier_detail THAT trong prompt phan tu -- neu khong, model
# hoc cach "doan mo" 1 loi nghe hop ly thay vi doc loi that va sua (da phat hien qua
# evaluate_self_correction.py: model tu bia ra loi gia, sua sai cho, van fail).
_REFLECT_PROMPT_TEMPLATE = (
    "Kết quả kiểm tra: SAI.\n"
    "Chi tiết lỗi từ hệ thống kiểm tra: {verifier_detail}\n"
    "Hãy tự rà soát lại lời giải trên và sửa lại cho đúng."
)
# Role "tool" (khong phai "user") cho tin nhan phan tu -- theo paper 2606.05976
# (The Self-Correction Illusion): verifier_detail von la output that cua 1 tool
# (checker), dua duoi role "tool" giup model addressable hoa claim sai tot hon.
# Da do thuc te qua evaluate_self_correction.py --reflect-role: tool giup ty le
# tu sua TOAN tang 0% -> 33.3% (khong doi voi code, vi code von da addressable
# qua traceback). Xem note.txt muc 7.
_REFLECT_ROLE = "tool"


def _load_attempts(path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_chat_record(problem: Problem, attempt_text: str, correction: dict, verifier_detail: str) -> dict:
    # reasoning: chain-of-thought THAT SU cua DeepSeek (reasoning_content tu API),
    # khong phai text tu bia -- day model nho hoc CACH suy luan tung buoc de tim
    # ra loi, khong chi hoc thuoc format 3 phan ben duoi.
    reasoning = correction.get("reasoning", "").strip()
    thinking_block = f"<thinking>\n{reasoning}\n</thinking>\n\n" if reasoning else ""

    assistant_reflection = (
        f"{thinking_block}"
        f"### Phát hiện lỗi\n{correction['error_location']}\n\n"
        f"### Nguyên nhân\n{correction['error_reason']}\n\n"
        f"### Sửa lại\n{correction['corrected_solution']}"
    )
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": problem.question},
            {"role": "assistant", "content": attempt_text},
            {
                "role": _REFLECT_ROLE,
                "content": _REFLECT_PROMPT_TEMPLATE.format(verifier_detail=verifier_detail),
            },
            {"role": "assistant", "content": assistant_reflection},
        ]
    }


def main() -> None:
    cfg = load_config()

    problems: list[Problem] = load_math_problems(cfg.path("problems_math")) + load_code_problems(
        cfg.path("problems_code")
    )
    problems_by_id = {p.id: p for p in problems}

    attempts = _load_attempts(cfg.path("attempts_out"))

    verifiers = {
        "math": MathVerifier(),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }
    deepseek = DeepSeekClient(cfg.deepseek)

    stats = {"total": 0, "already_correct": 0, "sent_to_deepseek": 0, "kept": 0, "discarded": 0}

    out_path = cfg.path("sft_dataset_out")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for attempt in attempts:
            stats["total"] += 1
            problem = problems_by_id[attempt["problem_id"]]
            verifier = verifiers[problem.domain]

            first_result = verifier.verify(problem, attempt["text"])
            if first_result.passed:
                stats["already_correct"] += 1
                continue

            stats["sent_to_deepseek"] += 1
            try:
                correction = deepseek.critique_and_correct(
                    problem, attempt["text"], first_result.detail
                )
            except RuntimeError as e:
                print(f"[SKIP] {problem.id}: DeepSeek loi - {e}")
                stats["discarded"] += 1
                continue

            second_result = verifier.verify(problem, correction["corrected_solution"])
            if not second_result.passed:
                print(
                    f"[DISCARD] {problem.id}: correction cua DeepSeek van sai - {second_result.detail}"
                )
                stats["discarded"] += 1
                continue

            record = _to_chat_record(problem, attempt["text"], correction, first_result.detail)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    print(f"Da ghi dataset SFT vao: {out_path}")
    print(f"Thong ke: {stats}")


if __name__ == "__main__":
    main()
