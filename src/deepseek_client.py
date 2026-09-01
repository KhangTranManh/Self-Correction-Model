"""Wrapper goi DeepSeek V4 Pro (API OpenAI-compatible) de sinh critique + correction
cho cac attempt da bi verifier (khach quan) xac nhan la SAI.

DeepSeek chi duoc goi SAU KHI verifier da chay va bao fail — tiet kiem token,
va dam bao model chi hoc tu nhung case that su can sua.
"""
from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from src.config import DeepSeekConfig
from src.data.schema import Problem

_SYSTEM_PROMPT = (
    "Ban la mot chuyen gia cham bai nghiem khac va chinh xac. Nhiem vu cua ban la "
    "phan tich mot loi giai SAI (da duoc xac nhan sai boi chuong trinh kiem tra khach quan), "
    "chi ra CHINH XAC buoc suy luan nao sai, giai thich nguyen nhan goc, va dua ra loi giai da sua. "
    "Luon tra loi bang JSON hop le, khong them van ban ngoai JSON."
)

_USER_TEMPLATE = """Đề bài:
{question}

Lời giải của một học sinh (ĐÃ ĐƯỢC XÁC NHẬN LÀ SAI bởi chương trình kiểm tra khách quan):
{attempt_text}

Kết quả kiểm tra khách quan (verifier):
{verifier_detail}

Nhiệm vụ:
1. Chỉ ra CHÍNH XÁC bước suy luận / dòng code nào sai (không đoán chung chung, không nói "có thể sai ở đâu đó").
2. Giải thích rõ nguyên nhân gốc của lỗi (root cause), không chỉ mô tả hiện tượng.
3. Đưa ra lời giải đã sửa, đầy đủ, đúng với đề bài, trong trường "corrected_solution":
   - Nếu là bài TOÁN: trình bày các bước, và BẮT BUỘC kết thúc bằng dòng riêng "Đáp số: <giá trị>".
   - Nếu là bài CODE: chỉ trả về code Python thuần (không fence markdown, không giải thích xen trong code).

Trả về đúng định dạng JSON sau, không thêm bất kỳ text nào khác:
{{
  "error_location": "...",
  "error_reason": "...",
  "corrected_solution": "..."
}}"""


def _extract_json(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Khong parse duoc JSON tu response DeepSeek: {raw_text[:500]}")


class DeepSeekClient:
    def __init__(self, cfg: DeepSeekConfig):
        self.cfg = cfg
        # The SDK default can wait roughly ten minutes on a dead upstream
        # connection and also performs its own retries. Keep one bounded request
        # timeout and let the explicit retry loop below own retry behavior.
        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=300.0,
            max_retries=0,
        )

    def critique_and_correct(
        self, problem: Problem, attempt_text: str, verifier_detail: str
    ) -> dict:
        user_prompt = _USER_TEMPLATE.format(
            question=problem.question,
            attempt_text=attempt_text,
            verifier_detail=verifier_detail,
        )

        last_error: Exception | None = None
        for attempt_no in range(self.cfg.max_retries):
            choice = None
            raw_text = None
            try:
                # Dat request ben trong khoi try: loi mang/rate-limit/5xx tu API
                # cung phai duoc retry, khong duoc lam crash ca dataset builder.
                response = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )

                # response.choices co the la None/rong (loi tam thoi tu API/proxy,
                # da gap thuc te voi vilao.ai) -- coi day la loi can thu lai, khong
                # de TypeError thoat ra ngoai vong retry lam crash ca build_dataset.py.
                if not response.choices:
                    raise ValueError(
                        f"Response khong co choices (response.choices={response.choices!r})"
                    )
                choice = response.choices[0]
                raw_text = choice.message.content
                # deepseek-v4-pro la model reasoning: tra ve them reasoning_content
                # (chain-of-thought that su, tach biet voi content JSON cuoi cung).
                # Dung lam "Thinking" trace de day model nho hoc cach suy luan tung
                # buoc, khong chi hoc thuoc format critique.
                reasoning = getattr(choice.message, "reasoning_content", None) or ""

                if choice.finish_reason == "length":
                    raise ValueError(
                        f"Response bi cat cut do het max_tokens (finish_reason=length, "
                        f"completion_tokens={response.usage.completion_tokens})"
                    )
                parsed = _extract_json(raw_text)
                for key in ("error_location", "error_reason", "corrected_solution"):
                    if key not in parsed:
                        raise ValueError(f"Thieu key '{key}' trong response DeepSeek")
                parsed["reasoning"] = reasoning
                return parsed
            except Exception as e:
                # Bao gom JSON/schema loi, timeout, rate-limit va HTTP 5xx. Khong
                # in request/config de tranh lo API key; chi in loai loi + preview
                # response model (neu da co).
                finish_reason = choice.finish_reason if choice is not None else "N/A"
                preview = raw_text[:200] if raw_text else "N/A"
                print(
                    f"[deepseek_client] Lan thu {attempt_no + 1}/{self.cfg.max_retries} "
                    f"that bai ({type(e).__name__}): {e} "
                    f"(finish_reason={finish_reason}, content_preview={preview!r})"
                )
                last_error = e
                if attempt_no + 1 < self.cfg.max_retries:
                    time.sleep(min(2**attempt_no, 8))
                continue

        raise RuntimeError(f"DeepSeek tra ve JSON khong hop le sau {self.cfg.max_retries} lan thu: {last_error}")
