# Phase 14 — trạng thái cuối trước khi tắt máy

Viết lúc 2026-10-14 05:46 (local) bởi finalize.py. Lý do chốt: **both runs finished; guardian made the final sync**.

## Đồng bộ

- Dòng đồng bộ cuối: | 2026-10-14 05:35 | final sync complete; guardian stops |
- Hidden state (F:\AGI_data\phase14_hidden): Run finished on the host: **yes**. Files: 57, local and verified: 57, total 19.4 GB.
- Phân tích có ở local: probe_report.json, report.json, report_tables.md
- Stage judge 1536 token đã đồng bộ: 4/4 (outputs/phase14_remote/outputs/phase14_v1/main/llama31_8b/asdiv/judge_base_1536, outputs/phase14_remote/outputs/phase14_v1/main/llama31_8b/asdiv/judge_base_c_1536, outputs/phase14_remote/outputs/phase14_v1/main/llama31_8b/math13/judge_base_1536, outputs/phase14_remote/outputs/phase14_v1/main/llama31_8b/math13/judge_base_c_1536)

## Script hậu xử lý (local)

- `phase14_decomposition.py`: ok
- `phase14_truncation.py`: ok
- `phase14_catalog.py`: ok

## Nếu thiếu

Mọi stage chạy lại được từ audit log trên một host mới (`phase14/scripts/ops/run_complete.sh`, `ops/run_long_judge.sh`); dữ liệu đã có ở local được mô tả trong `phase14/DATA_CATALOG.md`.

## Tóm tắt kết quả

Bản tổng kết đầy đủ (mọi số cuối, 4 mô hình, claim được / không được viết): [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md).

### Tóm tắt nhanh

- **Endpoint (không đổi):** E1 skill index 4/4 → tổng quát; E2 log-prob 1/4 → có điều kiện.
- **Kiểm tra pipeline:** judge `base` tái tạo đúng tuyệt đối số báo cáo; `base_c` chạy lại trong ±0,01; Self_Correction
  với lời giải mới tái lập Phase 13 (`paper/PHASE14_PRELIMINARY.md` mục 1).
- **L1:** cả 20/20 ô có skill index KTC > 0; hướng bias ổn định theo mô hình (Llama, Mistral, Qwen nghiêng vị trí đầu;
  Self_Correction nghiêng vị trí sau). Khoảng cách một thứ tự chủ yếu do không có lựa chọn hợp lệ, bias 0–2,5 điểm.
- **L2:** endpoint 1/4; ô mô tả 8/16 có mức tăng ≥ 3 điểm (không đồng nhất); liều–đáp ứng không ủng hộ (ρ = −0,27).
- **Trần judge 1536 (Llama):** khoảng cách không đổi (14,3 → 14,5; 19,4 → 19,6); nâng trần không cứu được — các lần
  bị cắt phần lớn là lặp vòng.
- **L3 (probe, cả 4 mô hình, `main/analysis/probe_report.json`):** probe cặp vượt đặc trưng bề mặt ở **mọi** hướng
  train/test với Self_Correction (AUC 0,75–0,84 vs 0,62–0,72); ở Llama vượt trên cặp ASDiv↔MATH nhưng không ở mọi hướng;
  ở Mistral và Qwen đặc trưng bề mặt cao hơn (Qwen bề mặt AUC 0,90–0,95 — nhiều khả năng do lỗi định dạng đáp số làm
  "parse được hay không" thành tín hiệu mạnh). Probe bị đảo khi đổi thứ tự ở 22–72% cặp: không mô hình nào có probe
  không bị thứ tự ảnh hưởng.
- **Cần làm sau:** cập nhật số cuối vào `paper/PHASE14_REPORT.md`, `paper/REPORT_SPEC.md`, `paper/CHECKLIST_RESULTS.md`;
  bảng độ nhạy "đọc lựa chọn bằng `rescued`"; quyết định cách viết điểm Qwen vs Self_Correction và tiêu đề; viết báo
  cáo theo mẫu .docx; commit. GPU có thể tắt.
