"""Orchestration Phase 1:
  attempts.jsonl (that, tu generate_attempts.py)
    -> verifier khach quan (math/code)
    -> neu SAI: goi DeepSeek sinh critique + correction
    -> verifier lai correction
    -> chi giu mau neu correction DA DUOC XAC NHAN DUNG
    -> ghi ra data/processed/phase1_sft.jsonl (dinh dang chat/ChatML)

RESUMABLE: ghi lai MOI problem_id da xu ly (bat ke ket qua: already_correct/kept/
discarded) vao data/processed/build_dataset_seen_ids.txt. Lan chay sau chi xu ly
attempt co problem_id CHUA co trong file nay -- tranh goi lai DeepSeek (ton tien
that) cho nhung case da biet ket qua tu truoc, kho co ich khi scale them du lieu.
Neu muon lam lai tu dau, xoa file ledger nay + phase1_sft.jsonl truoc khi chay.

Chay (khong can GPU, chi can .env co DEEPSEEK_API_KEY/DEEPSEEK_MODEL va da co attempts.jsonl):
    python -m src.pipeline.build_dataset
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from src.config import load_config
from src.data.problem_sources import load_code_problems, load_math_problems
from src.core.schema import Problem
from src.llm.api_client import DeepSeekClient
# QUAN TRONG: phai co verifier_detail THAT trong prompt phan tu -- neu khong, model
# hoc cach "doan mo" 1 loi nghe hop ly thay vi doc loi that va sua (da phat hien qua
# evaluate_self_correction.py: model tu bia ra loi gia, sua sai cho, van fail).
# Template nay dung chung voi ca 2 script eval qua src/prompts.py -- KHONG copy lai
# gia tri vao day, lech 1 byte la du de pha vo su khop nhau giua train va infer.
from src.core.prompts import build_reflect_message
from src.core.run_lock import single_instance
from src.data.verifiers.code import CodeVerifier
from src.data.verifiers.math import MathVerifier

_SYSTEM_PROMPT = "Bạn luôn kiểm tra lại lời giải của mình trước khi chốt câu trả lời cuối."
# Role "tool" (khong phai "user") cho tin nhan phan tu -- theo paper 2606.05976
# (The Self-Correction Illusion): verifier_detail von la output that cua 1 tool
# (checker), dua duoi role "tool" giup model addressable hoa claim sai tot hon.
# Da do thuc te qua evaluate_self_correction.py --reflect-role: tool giup ty le
# tu sua TOAN tang 0% -> 33.3% (khong doi voi code, vi code von da addressable
# qua traceback). Xem note.txt muc 7.
_REFLECT_ROLE = "tool"
# TAT theo mac dinh: giu nguyen hanh vi cu ("dap so sai: dung phai la X") cho du
# lieu train hien co. Bat co CHU DICH cho vong train THU NGHIEM sau khi Giai
# doan 3 (probe re bang eval-time, khong retrain -- xem evaluate_vllm.py
# --math-localize-steps) cho thay dinh vi buoc giup ich that, tranh doi hanh vi
# du lieu train am tham truoc khi co bang chung. Xem locate_wrong_step trong
# src/data/verifiers/math.py.
_MATH_LOCALIZE_STEPS = False


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
        # problem_id: khong dung de train (train_sft.py chi doc "messages"), chi
        # de build_dataset.py/cong cu khac sau nay biet mau nay tu problem nao.
        "problem_id": problem.id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": problem.question},
            {"role": "assistant", "content": attempt_text},
            # Qua build_reflect_message() de khung role o luc TRAIN dung y het luc
            # EVAL -- ke ca truong hop "memory" (phai la system + <memory>, khong
            # phai role "memory"). Xem src/prompts.py.
            build_reflect_message(verifier_detail, _REFLECT_ROLE),
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
        "math": MathVerifier(localize_steps=_MATH_LOCALIZE_STEPS),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }
    deepseek = DeepSeekClient(cfg.deepseek)

    stats = {"total": 0, "already_correct": 0, "sent_to_deepseek": 0, "kept": 0, "discarded": 0, "skipped_seen": 0}

    out_path = cfg.path("sft_dataset_out")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids_path = out_path.parent / "build_dataset_seen_ids.txt"

    seen_ids: set[str] = set()
    if seen_ids_path.exists():
        seen_ids = {line.strip() for line in seen_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"[build_dataset] Da co {len(seen_ids)} problem_id xu ly tu truoc -- bo qua, chi xu ly moi.")

    todo = [a for a in attempts if a["problem_id"] not in seen_ids]
    stats["skipped_seen"] = len(attempts) - len(todo)

    # SONG SONG HOA: buoc nay bi chan boi do TRE MANG, khong phai CPU -- moi bai la
    # 1 lan goi API toi model reasoning voi max_tokens=8000, thuc te 30-90 giay. Chay
    # tuan tu tren ~1300 bai la khoang 9-15 tieng, trong khi GPU nam khong (buoc nay
    # khong dung GPU). Voi thread pool, thoi gian gan nhu chia deu cho so worker.
    #
    # deepseek.max_workers trong phase1.yaml. Dat vua phai: qua cao thi dinh rate
    # limit cua provider, va moi request deu ton token that.
    max_workers = int(cfg.raw.get("deepseek", {}).get("max_workers", 8))
    lock = threading.Lock()
    done_count = 0
    total_todo = len(todo)
    print(f"[build_dataset] {total_todo} attempt can xu ly, {max_workers} worker song song.")

    def process(attempt: dict) -> None:
        """Xu ly 1 attempt. Moi ghi file/stats deu nam trong lock."""
        nonlocal done_count
        problem = problems_by_id[attempt["problem_id"]]
        verifier = verifiers[problem.domain]

        first_result = verifier.verify(problem, attempt["text"])

        correction = None
        second_result = None
        if not first_result.passed:
            try:
                correction = deepseek.critique_and_correct(
                    problem, attempt["text"], first_result.detail
                )
                second_result = verifier.verify(problem, correction["corrected_solution"])
            except Exception as e:  # noqa: BLE001
                # Bat rong: trong thread pool, mot ngoai le khong bat se lam pool.map
                # nem lai va giet toan bo cac worker con lai. Mot bai hong chi duoc
                # phep lam mat chinh no.
                print(f"[SKIP] {problem.id}: {type(e).__name__} - {e}")
                correction = None

        with lock:
            done_count += 1
            stats["total"] += 1
            # Ghi seen_id cho MOI attempt da xu ly, bat ke ket qua -- day la co so
            # cua tinh resumable. Ghi trong lock + flush ngay de neu bi ngat giua
            # chung thi phan da lam khong bi lam lai.
            seen_f.write(attempt["problem_id"] + "\n")
            seen_f.flush()

            if first_result.passed:
                stats["already_correct"] += 1
            elif correction is None:
                stats["sent_to_deepseek"] += 1
                stats["discarded"] += 1
            elif not second_result.passed:
                stats["sent_to_deepseek"] += 1
                stats["discarded"] += 1
                print(f"[DISCARD] {problem.id}: correction van sai - {second_result.detail}")
            else:
                stats["sent_to_deepseek"] += 1
                record = _to_chat_record(
                    problem, attempt["text"], correction, first_result.detail
                )
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                stats["kept"] += 1

            if done_count % 25 == 0 or done_count == total_todo:
                print(
                    f"[build_dataset] {done_count}/{total_todo} | kept={stats['kept']} "
                    f"already_correct={stats['already_correct']} discarded={stats['discarded']}",
                    flush=True,
                )

    # Khoa: 2 instance cung ghi phase1_sft.jsonl se tao ban ghi trung va ton tien API
    # gap doi cho cung mot bai. Da xay ra that -- xem src/run_lock.py.
    with single_instance("build_dataset", out_path.parent):
        with open(out_path, "a", encoding="utf-8") as out_f, open(
            seen_ids_path, "a", encoding="utf-8"
        ) as seen_f:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(process, todo))

    print(f"Da ghi dataset SFT vao: {out_path}")
    print(f"Thong ke: {stats}")


if __name__ == "__main__":
    main()
