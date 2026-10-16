# Phase 14 — soát `rescued` (mục 1b) — **bản do trợ lý gán, chưa phải người độc lập**

Mẫu: `rescued_audit_sample.csv` (seed 20261015; tối đa 50 lần chấm loại 2 mỗi mô hình — Self_Correction chỉ có 29).
Tiêu chí: số `rescued` đọc được có đúng là đáp số cuối judge muốn nói không. *Nhầm* = rõ ràng sai; *mơ hồ* = judge đưa
nhiều giá trị / biểu thức / nhãn mâu thuẫn, hoặc 300 ký tự cuối không đủ để kiểm. Ngưỡng ghi trước (`RUN_STATUS.md`
2026-10-15, ii): < 5% → độ nhạy chính; 5–15% → cảnh báo; > 15% → phụ lục. Chi tiết từng dòng: `rescued_audit_assistant.csv`.

| Mô hình | n | Nhầm | Mơ hồ | Tỷ lệ nhầm | Tỷ lệ nhầm + mơ hồ (thận trọng) | Quyết định (bản thận trọng) |
|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | 50 | 2 | 3 | 4.0% | 10.0% | báo cáo kèm cảnh báo |
| Mistral-7B | 50 | 1 | 5 | 2.0% | 12.0% | báo cáo kèm cảnh báo |
| Self_Correction | 29 | 1 | 6 | 3.4% | 24.1% | chỉ để phụ lục |
| Qwen2.5-7B-Instruct | 50 | 0 | 0 | 0.0% | 0.0% | dùng làm độ nhạy chính |
| **Tổng** | 179 | 4 | 14 | 2.2% | 10.1% | báo cáo kèm cảnh báo |

Nhận xét:
- Mọi lỗi rõ ràng đến từ đáp số không phải số nguyên đơn (căn, phân số, hai giá trị): `rescued` lấy số đầu tiên.
- Qwen (mô hình mà `rescued` thay đổi kết luận nhiều nhất) không có dòng nhầm hay mơ hồ nào trong 50 → kết luận "Qwen
  lên 72–88% một thứ tự khi đọc bằng `rescued`" đứng được ở mức độ nhạy chính.
- Self_Correction có ít lần chấm loại 2 (29) và nhiều dòng mơ hồ → `rescued` của mô hình này chỉ để phụ lục.
- Cần một người độc lập gán lại (R1) trước khi ghi "đã kiểm bằng tay" trong paper; hiện paper ghi "assistant-annotated".
