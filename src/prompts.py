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

# Cac khung (role) co the boc verifier_detail o luot phan tu. Theo paper 2606.05976,
# hieu qua PHU THUOC DOMAIN: <memory> manh nhat o toan, user manh nhat o logic,
# tool manh nhat o BBH-LD. Xem REFLECT_ROLES ben duoi va build_reflect_message().
REFLECT_ROLES = ("user", "tool", "memory")


def build_prompt(problem: Problem) -> str:
    return PROMPT_TEMPLATES[problem.domain].format(question=problem.question)


def build_reflect_message(verifier_detail: str, role: str) -> dict:
    """Dung message cho luot phan tu theo khung `role`.

    CANH BAO QUAN TRONG ve role "memory":
    Chat template cua Qwen2.5 KHONG biet role "memory". Dat {"role": "memory"} thi
    template NUOT IM LANG ca message -- khong raise, khong warning, thong bao loi
    bien mat hoan toan khoi prompt. Model se duoc yeu cau "tu sua" ma khong he biet
    sai o dau, so lieu sup do, va KHONG co gi chi ra nguyen nhan. Da verify truc tiep
    tren tokenizer.
    Vi vay "memory" duoc hien thuc dung nhu paper lam: role "system" + noi dung boc
    trong <memory>...</memory>.

    Ghi chu ve role "tool" (khung dang dung mac dinh): tren template Qwen no KHONG
    phai mot role token rieng -- no render thanh
        <|im_start|>user\\n<tool_response>...</tool_response><|im_end|>
    tuc la user message co them lop boc XML. Doi chieu voi thang H0-H4 muc 3.3 cua
    paper, day la nac H3 (lop boc cu phap, chua co role tag), khong phai H4. Paper
    do duoc: lop boc dong gop 17-23pp, con role tag dong gop THEM ~30pp.
    """
    content = REFLECT_PROMPT_TEMPLATE.format(verifier_detail=verifier_detail)
    if role == "memory":
        return {"role": "system", "content": f"<memory>\n{content}\n</memory>"}
    return {"role": role, "content": content}
