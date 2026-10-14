# Phase 14 — đánh giá sơ bộ (2026-10-14 ~02:00, trước bước phân tích chính thức trên host)

**Mô tả, chưa phải kết quả cuối.** 20 ô (4 mô hình × 4 dataset + bridge + 2 bộ hồi quy) đã đồng bộ về local,
trừ bridge của Qwen-Instruct (đang chạy). Số được tính tại local bằng đúng `phase14/scripts/metrics.cell`
(như `analyze.py`), file riêng [`results/phase14_preliminary.md`](results/phase14_preliminary.md); ô endpoint
lấy nguyên từ báo cáo đã mở lúc 03:55 UTC 13/10. Không ô nào thay đổi E1/E2. Mọi đánh giá dưới đây theo
**quy tắc diễn giải ghi trước** trong `phase14/RUN_STATUS.md` (claim khóa L1–L3). Parser chuẩn (`lenient`),
judge `base` trừ khi ghi khác; [a; b] = KTC 95% bootstrap theo bài.

## 1. Kiểm tra pipeline — đạt

| Kiểm tra | Kết quả |
|---|---|
| Judge `base` Self_Correction, output gốc của báo cáo (GSM8K 561, SVAMP 960) | **trùng tuyệt đối** mọi số in trong báo cáo (skill 0,215 / 0,248; tie-break 0,601 / 0,609; …; chênh 0,000) |
| `base_c` chạy lại trên các cặp Phase 13 | trong ±0,01 so với Phase 13 (ngưỡng ±0,03): một thứ tự 0,550 vs 0,549, tie-break 0,568 vs 0,568, coverage 0,497 vs 0,490 |
| Self_Correction lời giải **mới** (seed Phase 14) vs Phase 13/12 | GSM8K skill 0,191 [0,095; 0,286] (báo cáo 0,215), tie-break 0,590 (0,601); SVAMP skill 0,237 [0,151; 0,323] (0,248) → **"tái lập với mẫu mới"** |
| Qwen-Instruct GSM8K vs tham chiếu T1 (tie-break 0,629) | 0,651 [0,613; 0,689], chứa 0,629 → khớp |

## 2. L1 — một thứ tự trông như đoán mò, skill index dương rõ

**Cả 20/20 ô có KTC skill index nằm trên 0** → 16 ô mô tả ghi là "phù hợp với E1" (đếm riêng, không cộng vào
4/4 endpoint). Không ô nào là phản ví dụ.

| Mô hình | Skill index (4 dataset) | Một thứ tự | Chọn vị trí 2 | Bịa |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0,47–0,61 | 45–64% | **34–42%** (nghiêng A) | 22–49% |
| Llama-3.1-8B | 0,31–0,36 | 50–57% | 41–44% (nghiêng A) | 19–29% |
| Mistral-7B | 0,25–0,34 | 47–51% | 42–45% (nghiêng A) | 29–31% |
| Self_Correction | **0,16–0,24** | 49–56% | **52–56% (nghiêng B)** | 10–20% |

- **Hướng bias ổn định theo mô hình, trên cả 4 dataset:** ba mô hình chưa fine-tune nghiêng vị trí đầu,
  Self_Correction nghiêng vị trí sau — đúng như T6 và phù hợp (không chứng minh) với dữ liệu SFT "câu sau là
  đúng" (D1b).
- **Phân rã Shapley** (`results/phase14_decomposition.md`, khám phá): khoảng cách tới R do thiên lệch vị trí
  chỉ 0,0–2,5 điểm ở mọi mô hình; phần còn lại là không có lựa chọn hợp lệ — Mistral gần như toàn bịa thật
  (18–20 điểm), Llama chia bị cắt (6–10) và bịa (8–14), Qwen lớn nhất (20–43 điểm, MATH: bị cắt 14,5 + bịa
  28,7), Self_Correction nhỏ nhất (6–12). Câu chữ đã chốt ("chủ yếu vì không có lựa chọn hợp lệ, ít hơn nhiều
  do thiên lệch vị trí") **đứng vững trên cả 4 mô hình**.

## 3. L2 — trung bình log-prob qua hai thứ tự thêm ít

- Endpoint: 1/4 (giữ nguyên "có điều kiện").
- Ô mô tả: mức tăng ≥ 3 điểm với KTC > 0 ở **8/16**: Llama GSM8K **+10,9** [5,8; 16,2], Llama bridge +3,7,
  Self_Correction ASDiv +5,8, SVAMP +6,9, GSM8K-báo cáo **+8,2** [3,6; 12,8], SVAMP-báo cáo +5,7, Qwen ASDiv
  +6,0, Qwen SVAMP +4,4. → theo quy tắc: **"không đồng nhất giữa mô hình và dataset"**; E2 vẫn có điều kiện.
- **Phần bỏ bias** ≥ 3 điểm với KTC > 0 ở 4 ô — Self_Correction ASDiv +4,6 [1,0; 8,0], SVAMP +3,9 [0,5; 7,4],
  SVAMP-báo cáo +6,3 [2,9; 9,7], Mistral SVAMP +3,1 [1,1; 5,0] → báo cáo là **ngoại lệ** của "bỏ bias ≈ 0".
  Ở các ô khác, mức tăng chủ yếu là phần "đọc hai lần".
- Liều–đáp ứng trên toàn 16 ô lưới: Spearman ρ = −0,27 (p = 0,32) → **vẫn không được ủng hộ**. Qwen có bias
  log-prob lớn nhất (log-odds 3,1–5,3) mà phần bỏ bias ≈ 0.
- **Số thay cho dòng sai cũ của Bảng 3:** Self_Correction trên chính các cặp GSM8K của báo cáo, đo đúng cách
  (log-prob, judge `base_c`): **+8,2 điểm [3,6; 12,8]**, trong đó bỏ bias +2,2 [−1,0; 5,4] và đọc hai lần +5,9.
  Con số "+7,1" cũ (tie-break − một thứ tự) gần về độ lớn nhưng là đại lượng khác.

## 4. L3 — hidden state

Chưa có cho các ô mới: probe chạy trên host sau bước `--add-comparison` (~03:30). Hidden state của mọi ô đã
tải về F: (đối chiếu SHA-256).

## 5. Đối chiếu với tài liệu cũ

| Phát hiện cũ | Phase 14 | Đánh giá |
|---|---|---|
| Phase 13: Self_Correction GSM8K skill +0,215, một thứ tự 52,3%, tie-break 0,601 | lời giải mới 0,191 / 52,2% / 0,590 | tái lập |
| T6: Self_Correction nghiêng vị trí 2 trên bài của chính nó; Instruct nghiêng vị trí 1 | đúng trên cả 4 dataset (52–56% vs 34–42%) | củng cố, mở rộng sang ASDiv/MATH/SVAMP |
| T1/T3: Instruct tự chấm GSM8K tie-break 0,63–0,66, không hơn Self_Correction đáng kể | tie-break 0,651 vs 0,590; **skill index 0,485 vs 0,191**; trên cả 4 dataset Qwen 0,47–0,61 vs 0,16–0,24 | **cần viết lại thận trọng** (mục 6) |
| D1: Instruct chấm cặp của Self_Correction tốt hơn hẳn (0,77) | bridge: Llama 0,729, skill 0,545 (so với 0,60–0,65 trên bài của chính nó); Mistral 0,595; Qwen chưa có | "chấm cặp của Self_Correction tốt hơn bài mình" đúng với Llama, không với Mistral → mô tả theo từng mô hình |
| B5/F3: judge đóng góp vài điểm so với ngẫu nhiên; self-check ≈ vote@3; vote@5 tốt nhất | judge-hoặc-ngẫu-nhiên hơn ngẫu nhiên 1,5–4 điểm; self-check trong ±2,5 điểm của vote@3; vote@5 tốt nhất ở cả 20 ô | củng cố |
| E2 sửa: tie-break − một thứ tự là thành phần bịa, không so được với Ren | log-prob: endpoint 0,6–5, ô mô tả đến 11 điểm, phần bỏ bias phần lớn ≈ 0 | gần Ren ở đa số ô, có ngoại lệ |

## 6. Điểm cần chú ý trước khi viết

1. **Mẹ vs con (Qwen-Instruct vs Self_Correction).** Skill index của mô hình gốc lớn gấp 2–3 lần ở cả 4 dataset,
   trong khi tie-break chỉ hơn 0,04–0,10. Chênh lệch giữa hai thước đo đến từ việc Qwen bịa nhiều hơn (22–49%
   so với 10–20%), mà mục 7 cho thấy một phần là do viết đáp số thành câu. T1 kết luận "không ủng hộ SFT làm
   giảm kỹ năng" dựa trên tie-break; skill index (thước đo không bị thiên lệch vị trí) nghiêng về hướng ngược
   lại. Theo quy tắc khóa: **không tạo claim mới**; báo cáo cả hai thước đo, ghi rõ tập cặp khác nhau (mỗi mô
   hình chấm lời giải của chính nó, Qwen giải đúng nhiều hơn) và để câu "SFT làm giảm khả năng phân xử" ở dạng
   câu hỏi mở.
2. **Đo "bịa" bị lẫn lỗi định dạng.** Trong các lần chấm bị tính là bịa, cách đọc `rescued` cho thấy đáp số
   thật ra khớp một trong hai lời giải ở **Qwen 20–43%**, Mistral 6–19%, Llama 5–15%, Self_Correction 2–7%.
   Quy tắc chọn đã đăng ký (parser chuẩn) giữ nguyên cho số chính; nên thêm một bảng độ nhạy "lựa chọn đọc
   bằng `rescued`" (CPU, không cần GPU) — dự kiến nâng một thứ tự và tie-break của Qwen nhiều nhất.
3. **Self_Correction có skill index thấp nhất ở mọi dataset** nhưng cũng bịa ít nhất và tie-break ngang các
   mô hình khác (0,57–0,60).
4. Ô `gsm8k13_report`/`svamp12_report` nhập judge `base` từ báo cáo nên không có cờ chạm trần — các bảng bị
   cắt bỏ qua hai ô này.

## 7. Còn chờ

Bridge của Qwen; `analyze.py --add-comparison`, kiểm tra hồi quy chính thức, probe cho mọi ô, bảng báo cáo
(~03:30); kiểm tra độ nhạy trần judge 1536 token cho Llama (~05:30). Khi xong: chạy lại
`phase14_preliminary.py`, `phase14_decomposition.py`, `phase14_truncation.py`, `phase14_catalog.py`.
