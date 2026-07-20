"""Nguon DUY NHAT cho moi prompt template dung chung giua cac stage cua pipeline.

VI SAO CO FILE NAY: _REFLECT_PROMPT_TEMPLATE truoc day duoc copy o CA
build_dataset.py (sinh du lieu train) LAN evaluate_self_correction.py (eval). Hai
ban copy do BAT BUOC phai giong nhau tung byte -- lech nhau la tai tao dung lop bug
ma viec nhung verifier_detail that vao sinh ra de sua (model gap prompt luc infer
khac voi luc train -> quay lai bia chan doan). Gom ve mot cho de dieu do duoc bao
dam boi cau truc code, khong con phu thuoc vao viec nguoi sua nho cap nhat ca 2 noi.

File nay co CHU DICH khong import gi nang (khong vllm, khong unsloth, khong
datasets): ca 2 moi truong -- venv vLLM (generate/eval) va venv Unsloth (train) --
deu phai import duoc no.
"""
from __future__ import annotations

import re

from src.data.schema import Problem

# Prompt cho luot giai dau tien (model tu giai, chua co phan hoi nao).
PROMPT_TEMPLATES = {
    "math": (
        "Giải bài toán sau từng bước, sau đó ghi rõ đáp số cuối cùng theo định dạng "
        "'Đáp số: <giá trị>'.\n\nBài toán: {question}"
    ),
    "code": (
        "{question}\n\nChỉ trả về code Python hoàn chỉnh trong 1 code block "
        "(```python ... ```), không giải thích thêm."
    ),
}

# Prompt cho luot PHAN TU (sau khi verifier da khang dinh loi giai tren la sai).
# {verifier_detail} la thong bao loi THAT tu sympy/traceback -- bat buoc phai co,
# xem instructionAI/data_pipeline.md muc "verifier_detail grounding fix".
REFLECT_PROMPT_TEMPLATE = (
    "Kết quả kiểm tra: SAI.\n"
    "Chi tiết lỗi từ hệ thống kiểm tra: {verifier_detail}\n"
    "Hãy tự rà soát lại lời giải trên và sửa lại cho đúng."
)

# Tach phan loi giai da sua ra khoi toan bo output (bo <thinking> + 2 muc dau).
CORRECTION_RE = re.compile(r"###\s*Sửa lại\s*\n(.*)", flags=re.DOTALL)


def build_prompt(problem: Problem) -> str:
    return PROMPT_TEMPLATES[problem.domain].format(question=problem.question)
