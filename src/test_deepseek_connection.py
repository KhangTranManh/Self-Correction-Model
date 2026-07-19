"""Kiem tra nhanh: .env co dung khong, DeepSeek API co goi duoc khong, JSON tra ve
co dung schema khong. KHONG can GPU -- nen chay truoc tien, truoc khi dong may
GPU thue theo gio de tranh phat hien loi cau hinh sau khi da ton tien.

Chay:
    python -m src.test_deepseek_connection
"""
from __future__ import annotations

from src.config import load_config
from src.data.schema import Problem
from src.deepseek_client import DeepSeekClient

_DUMMY_PROBLEM = Problem(
    id="smoke_test_0001",
    domain="math",
    question="Tinh 2 + 2.",
    reference_answer="4",
)
_DUMMY_WRONG_ATTEMPT = "2 + 2 = 5. Đáp số: 5"
_DUMMY_VERIFIER_DETAIL = "Dap so sai: model dua ra '5', dung phai la '4'"


def main() -> None:
    cfg = load_config()
    print(f"[OK] .env hop le -- model = {cfg.deepseek.model}, base_url = {cfg.deepseek.base_url}")

    client = DeepSeekClient(cfg.deepseek)
    print("Dang goi DeepSeek...")
    result = client.critique_and_correct(_DUMMY_PROBLEM, _DUMMY_WRONG_ATTEMPT, _DUMMY_VERIFIER_DETAIL)

    for key in ("error_location", "error_reason", "corrected_solution"):
        assert key in result, f"Thieu key '{key}' trong response"

    print("[OK] DeepSeek tra ve dung schema JSON:")
    print(f"  error_location:      {result['error_location']}")
    print(f"  error_reason:        {result['error_reason']}")
    print(f"  corrected_solution:  {result['corrected_solution']}")
    print("\nKET NOI DEEPSEEK HOAT DONG BINH THUONG -- co the yen tam chay pipeline that.")


if __name__ == "__main__":
    main()
