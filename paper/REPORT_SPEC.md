# Đặc tả số liệu cho báo cáo "Position bias che khuất khả năng tự phân xử"

Cập nhật 2026-10-13 (thêm kết quả endpoint Phase 14). Tài liệu này khớp bản mẫu báo cáo (`Bản mẫu báo cáo Position bias che khuất
khả năng tự phân xử.docx`) với một bộ định nghĩa và thông số duy nhất cho cả bốn mô hình. Mọi con
số trong báo cáo phải lấy theo đây. Chi tiết từng kiểm tra nằm trong
[`CHECKLIST_RESULTS.md`](CHECKLIST_RESULTS.md); thiết kế Phase 14 nằm trong
[`phase14/docs/PREREGISTRATION.md`](../phase14/docs/PREREGISTRATION.md).

## 0. Hai lớp số liệu

| Lớp | Mục trong báo cáo | Nguồn | Mô hình |
|---|---|---|---|
| **I. Phân tích sâu một mô hình** | 4.1–4.5, 5, Phụ lục A–E | Phase 9–13 + kiểm tra paper (A–F, T1–T6) | Self_Correction_v1 (judge base, base_c, p12, p13b); Qwen2.5-7B-Instruct ở 4.5 |
| **II. Lưới tổng quát hóa** | 4.6, Bảng 3 (phần "chúng tôi"), Hình 2, một câu trong Tóm tắt | Phase 14 (endpoint đã mở 2026-10-13 03:55 UTC; lần chạy hoàn thiện lưới đang chạy) | Self_Correction_v1, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 |

Quy tắc:
- Lớp II chạy lại **cả bốn mô hình** bằng cùng một giao thức (lời giải mới, seed Phase 14). Số của
  Self_Correction và Instruct ở 4.6 vì vậy sẽ khác đôi chút so với 4.1–4.5; không trộn hai lớp trong
  cùng một bảng.
- Hai lớp được nối bằng kiểm tra hồi quy (`phase14/scripts/regression.py`): Phase 14 phải tái tạo
  đúng số của lớp I từ chính output Phase 13.

## 1. Thông số cố định (cả bốn mô hình)

| Thông số | Giá trị | Ghi chú |
|---|---|---|
| Prompt giải (s1) | Prompt Phase 1 tiếng Việt, kết thúc "Đáp số" | import nguyên văn, không gõ lại |
| Prompt giải mù (s2–s5) | `blind_resolve` của Phase 7/8 | |
| Số lời giải | 5, temperature 0,7, seed theo từng request | |
| Cặp được chấm | (s1, s2) và (s1, s3) có đáp số khác nhau; cả hai thứ tự (ab: s1 là A; ba: s1 là B) | |
| Phân tích chính | Cặp đúng-một (đúng một lời giải đúng) | |
| Judge | Greedy. `base`: prompt Phase 9 (viết phần kiểm tra, đáp số tự do). `base_c`: prompt Phase 13 ép dòng "Verdict: Solution A/B" | |
| Trần token | **768 là chính** (cả lời giải lẫn judge); 1536 là phân tích độ nhạy | 4.5 phải dùng 768 cho mọi dòng |
| Context | 4096 | |
| Phần cứng | Phase 9–13: V100, FP16, vLLM 0.7.0. Kiểm tra paper: RTX 3090, FP16. **Phase 14: Tesla V100-SXM2 32 GB, FP16, vLLM 0.7.0, Transformers 4.48.1** | chunk vLLM: 64 (Llama, endpoint), 256 (Mistral và lần chạy hoàn thiện); seed theo request và judge greedy không đổi |
| Llama-3.1-8B-Instruct | Trọng số `NousResearch/Meta-Llama-3.1-8B-Instruct` @ d10aef79 (sha256 4 shard trùng repo Meta @ 0e9e39f2); tokenizer + chat template `RedHatAI/...-FP8-dynamic` @ 442e7f52 (trùng blob Meta) | repo Meta còn chờ duyệt quyền; ghi trong Phụ lục F |
| GSM8K | **519 bài (sau audit) là chính**, 561 bài báo cáo kèm | danh sách 42 bài: `phase14/data/gsm8k13_audit_excluded.json` |
| SVAMP | 960 bài (Phase 12) | |
| ASDiv, MATH 1–3 | `phase14/data/sources_v1/` (đã freeze, không trùng phase trước) | chỉ ở lớp II |
| Khoảng tin cậy | Bootstrap theo bài, 10.000 lần, 95% | |
| So sánh nhiều cặp | Holm | |
| Compute | Số lượt gọi mô hình (chấm hai thứ tự = 2 lượt); token là thước đo phụ | |

## 2. Cách đọc đáp số (thống nhất tên gọi)

Các file cũ dùng hai bộ tên khác nhau. Báo cáo dùng tên ở cột đầu.

| Tên trong báo cáo | Định nghĩa | Tên ở Phase 14 | Tên ở kiểm tra T3–T6 | Vai trò |
|---|---|---|---|---|
| **parser chuẩn** | Bộ đọc Phase 1: `\boxed{}` cuối, nếu không có thì dòng "Đáp số/Answer" cuối, nếu không có thì số cuối cùng | `lenient` | "strict" | **chính** (mọi số Phase 9–13 dùng cách này) |
| **rescued** | Parser chuẩn, nhưng nếu kết quả không phải số thì lấy số đầu tiên trên dòng "Đáp số" cuối | `rescued` (thêm 2026-10-12) | "lenient" | độ nhạy, **bắt buộc báo cáo** khi một mô hình có >2% đáp án bị đọc sai do định dạng |
| **boxed-only** | Chỉ chấp nhận `\boxed{}` hoặc dòng có nhãn | `strict` | — | độ nhạy |

Lý do có `rescued`: Instruct viết đáp số thành câu ("Đáp số: Mỗi trẻ nhận được 9 trứng.") ở 29/561
lời giải đầu, bị parser chuẩn chấm sai (Self_Correction: 4/561). Llama và Mistral trả lời prompt
tiếng Việt nên có thể gặp nhiều hơn.

## 3. Định nghĩa thước đo (một bộ cho mọi mô hình)

Tất cả tính trên cặp đúng-một, hai thứ tự, trừ khi ghi khác. "Lần chấm" = một thứ tự của một cặp.

| Thước đo | Định nghĩa | Đơn vị | Dùng ở |
|---|---|---|---|
| Accuracy một thứ tự | Tỷ lệ lần chấm chọn lời giải đúng (bịa = sai) | lần chấm | 4.1, B4 |
| **Skill index** | P(chọn A \| A đúng) − P(chọn A \| B đúng). Thiên lệch vị trí thuần tự triệt tiêu | lần chấm | 4.1, 4.6 (E1) |
| Tỷ lệ chọn vị trí 2 | Trong các lần chọn A hoặc B (bỏ bịa), tỷ lệ chọn B. 50% = không lệch | lần chấm | 4.1, 4.5, 4.6 |
| Bias log-odds | Trung bình (m_ab + m_ba)/2, m = log P(A) − log P(B) ở token verdict; > 0 nghiêng A | cặp, chỉ `base_c` | 4.6, Hình 2 |
| Tỷ lệ bịa | Lần chấm có đáp số không khớp lời giải nào | lần chấm | 4.1 |
| Nhất quán / coverage | Hai thứ tự chọn cùng một lời giải / tỷ lệ cặp như vậy | cặp | 4.2 |
| Đúng khi nhất quán | Đúng trong các cặp nhất quán | cặp | 4.2 |
| Đúng / toàn bộ cặp | Nhất quán và đúng / mọi cặp đúng-một (kiểu JudgeBench) | cặp | 4.2 |
| Tie-break | Nhất quán đúng = 1, nhất quán sai = 0, còn lại 0,5 | cặp | 4.2, 4.4, 4.5 |
| **Mức tăng khử bias (log-prob)** | Accuracy của sign(m_ab − m_ba) trừ accuracy một thứ tự của chính các log-prob đó, cùng cặp. Tách thành *phần bỏ bias* và *phần hai lần đọc* | cặp, chỉ `base_c` | **Bảng 3**, 4.6 (E2) |
| Tie-break − một thứ tự | Bằng đúng 0,5 × [không thứ tự nào chọn đúng và có bịa]. **Không phải mức tăng khử bias**, chỉ báo cáo kèm, không so với Ren et al. | cặp | Phụ lục E |
| Verdict ép | Prompt `base_c` + "Verdict: Solution", không viết phần kiểm tra; đọc log P(A), log P(B) ở lớp cuối và qua logit lens | lần chấm | 4.3, Hình 1 |
| Probe cặp | Hidden state token cuối của prompt judge; StandardScaler + logistic L2, C = 0,01; train SVAMP → test GSM8K; layer chọn bằng CV 5 fold theo bài trên tập train | lần chấm | 4.3 |
| Probe pointwise | Hidden state token cuối của prompt hỏi "đáp số của lời giải này có đúng không"; cùng probe | lời giải | 4.3, 5 |
| Baseline bề mặt | Logistic trên độ dài, số từ, số dòng, số lượng số, log độ lớn đáp số, đáp số nguyên, parse được, âm (cặp: A, B, B − A; pointwise: một lời giải) | như probe tương ứng | 4.3 |
| Baseline likelihood | Chọn lời giải có log-prob (tổng, và trung bình theo token) cao hơn dưới prompt giải | cặp | 4.3 |
| Khớp lần giải greedy | Chọn lời giải trùng một lần giải greedy mới (tốn thêm 1 lượt) | cặp | Phụ lục |
| Chiến lược | Giữ s1; ngẫu nhiên s1/s2 (kỳ vọng); judge hoặc ngẫu nhiên; self-check (không nhất quán → vote@3); vote@3; vote@5 | bài | 5 |

Không trộn đơn vị trong cùng một cột: AUC pointwise (mức lời giải) khác accuracy chọn cặp.

## 4. Cái gì đã có cho từng mô hình

| Thước đo | Self_Correction | Qwen2.5-7B-Instruct | Llama-3.1-8B | Mistral-7B |
|---|---|---|---|---|
| Bảng 1, skill index, bịa (GSM8K, SVAMP) | có (Phase 13, 12) | GSM8K: có, trên lời giải của chính nó (T1/T3) | Phase 14 | Phase 14 |
| Bảng 2 (tie-break, coverage…) | có | GSM8K: có (T1/T3) | Phase 14 | Phase 14 |
| Mức tăng khử bias log-prob (Bảng 3) | lần chạy hoàn thiện | lần chạy hoàn thiện | **có** (ASDiv, MATH) | **có** (ASDiv, MATH) |
| Probe cặp / pointwise / bề mặt / likelihood | có (C2–C4, F2, F2b) | lần chạy hoàn thiện | **có** | **có** |
| Logit lens, verdict ép | có cho base_c, p13b (C5) | lần chạy hoàn thiện | **có** (verdict ép) | **có** (verdict ép) |
| DPO judge (p12, p13b) | có | không | không | không |
| ASDiv, MATH 1–3 | lần chạy hoàn thiện | lần chạy hoàn thiện | **có (endpoint, mở 03:55 UTC)** | **có (endpoint)** |
| GSM8K, SVAMP (giao thức Phase 14) | lần chạy hoàn thiện | lần chạy hoàn thiện | lần chạy hoàn thiện | lần chạy hoàn thiện |
| Chiến lược mục 5 | có | GSM8K: có (T2) | **có** (ASDiv, MATH) | **có** (ASDiv, MATH) |

"Lần chạy hoàn thiện" = `phase14/scripts/ops/run_complete.sh` (bắt đầu 2026-10-13 05:58 UTC, cùng host):
điền đủ lưới 4 × 4, bộ bridge và chạy lại base_c cho kiểm tra hồi quy; các ô mới được thêm bằng
`analyze.py --add-comparison`, **không tính lại E1/E2 đã mở**.

## 5. Từng mục của bản mẫu: số dùng và chỗ cần sửa

### Tóm tắt
- "skill index +0,21 [0,13; 0,30]", "tie-break 0,60": giữ. Bản 519 bài: skill +0,207, tie-break 0,596.
- **Xóa** "Mức tăng khi khử bias lớn hơn đáng kể so với báo cáo trước đó" cho tới khi có E2 Phase 14
  (xem 4.2 dưới).
- "chỉ nhỉnh hơn đặc trưng bề mặt": đúng cho probe cặp (AUC 0,737 so với 0,742), **sai cho probe
  pointwise** (0,838 so với 0,722). Viết: "probe cặp chỉ ngang đặc trưng bề mặt; probe theo từng lời
  giải vượt rõ (AUC 0,84 so với 0,72)".
- "đáp số bịa giảm từ 17% xuống 2%": 16,7% (base) → 2,1% (p13b), nhưng riêng prompt ép verdict đã
  còn 4,9% (base_c). Viết kèm ý này.
- **Câu tổng quát [chờ kết quả] → điền:** "Trên hai họ mô hình khác (Llama-3.1-8B, Mistral-7B, chưa
  fine-tune) và hai dataset mới (ASDiv, MATH 1–3), một thứ tự cũng trông như đoán mò (50–57%) trong
  khi skill index dương rõ ở cả 4/4 ô (+0,32 đến +0,36); nhưng trung bình xác suất qua hai thứ tự chỉ
  thêm 0,6–5 điểm (đạt ngưỡng ở 1/4 ô), gần mức của Ren et al."
- **Xóa hẳn** câu "Mức tăng khi khử bias lớn hơn đáng kể so với báo cáo trước đó": Phase 14 cho thấy
  ngược lại (Bảng 3 dưới).

### 3. Thiết lập
Bổ sung theo mục 1–3 ở trên: Llama-3.1-8B-Instruct và Mistral-7B-Instruct-v0.3, ASDiv và MATH 1–3,
trần 768 (1536 độ nhạy), ba cách đọc đáp số, bias log-odds và mức tăng log-prob.

### 4.1 Bảng 1 và bảng skill index
Số hiện tại là GSM8K 561 bài, parser chuẩn: giữ, thêm cột hoặc chú thích 519 bài.

| Judge | GSM8K 561: skill [KTC] | GSM8K 519: skill | Single-order 561 / 519 | Bịa 561 / 519 |
|---|---|---:|---:|---:|
| base | 0,215 [0,128; 0,304] | 0,207 | 52,3% / 51,7% | 16,7% / 16,8% |
| p12 | 0,319 [0,236; 0,402] | 0,306 | 61,8% / 61,4% | 9,9% / 9,8% |
| base_c | 0,135 [0,045; 0,223] | 0,133 | 54,9% / 54,6% | 4,9% / 5,0% |
| p13b | 0,316 [0,222; 0,408] | 0,317 | 64,1% / 64,2% | 2,1% / 2,0% |

KTC cho bản 519 bài chưa tính: chạy lại `paper/scripts/a1_sensitivity.py` có bootstrap (CPU).

### 4.2 Bảng 2 và Bảng 3
- Bảng 2: giữ. Bản 519: tie-break base 0,596, p12 0,649, base_c 0,566, p13b 0,651.
- **Bảng 3 phải sửa.** Các dòng "Chúng tôi … +7,1 / +8,5" là hiệu tie-break − một thứ tự, tức là thành
  phần đáp số bịa, không phải mức tăng khử bias; và không cùng thang với Ren et al. (trắc nghiệm,
  không có lựa chọn bịa). Giữ hai dòng Ren et al.; thay các dòng "chúng tôi" bằng mức tăng log-prob
  của cả bốn mô hình (Phase 14, E2), tách phần bỏ bias và phần hai lần đọc. Xóa câu "Mức tăng không
  co lại khi tăng số ứng viên". Có thể nêu trong Phụ lục E: judge gần như không bịa (p13b) chỉ có
  hiệu +1,0, gần +0,5–0,6 của Ren et al.
- **Bảng 3 mới (Phase 14, judge `base_c`, log-prob ở token verdict, cùng thang Ren et al.):**

| Nghiên cứu / mô hình | Dataset | Một thứ tự | Trung bình hai thứ tự | Mức tăng [KTC] | Phần bỏ bias / hai lần đọc | Bias log-odds | Coverage log-prob |
|---|---|---:|---:|---|---|---:|---:|
| Ren et al., PaLM-2 L (4 ứng viên) | TruthfulQA | 58,26 | 58,87 | +0,6 | — | — | — |
| Ren et al., PaLM-2 L (4 ứng viên) | TL;DR | 70,20 | 70,70 | +0,5 | — | — | — |
| Llama-3.1-8B | ASDiv | 61,8 | 62,4 | +0,6 [−3,7; +4,8] | −0,8 / +1,4 | 1,65 | 80% |
| Llama-3.1-8B | MATH 1–3 | 62,4 | 67,4 | **+5,0 [+0,9; +9,3]** | +1,4 / +3,7 | 1,35 | 50% |
| Mistral-7B | ASDiv | 71,7 | 72,4 | +0,7 [−2,3; +3,6] | −0,3 / +0,9 | 1,89 | 69% |
| Mistral-7B | MATH 1–3 | 62,7 | 66,0 | +3,3 [−0,7; +7,3] | +1,2 / +2,1 | 2,40 | 87% |
| Self_Correction, Qwen-Instruct | 4 dataset | lần chạy hoàn thiện | | | | | |

  Thêm cột "verdict ép ngay" (không viết phần kiểm tra): Llama 67,2→66,8 (ASDiv), 65,0→69,8 (MATH);
  Mistral 64,7→67,0, 62,2→66,5. Đọc: bias về vị trí A rất lớn trong log-prob (log-odds 1,3–2,4),
  nhưng **phần bỏ bias chỉ −0,8 đến +1,4 điểm**; phần lớn mức tăng (khi có) đến từ việc đọc hai lần.
  E2 đạt ngưỡng (≥ 3 điểm, KTC > 0) ở 1/4 ô → "có điều kiện". Cột "tie-break − một thứ tự" vẫn đúng
  bằng phần bịa (Llama ASDiv 0,073 = 0,073, kiểm tra trong code), giữ ở Phụ lục E.

### 4.3 Bên trong mô hình
Sửa bảng để mỗi dòng ghi rõ đơn vị:

| Bộ chọn (GSM8K) | Đơn vị | Accuracy | AUC |
|---|---|---:|---:|
| Bề mặt, dạng cặp | lần chấm | 64,2% | 0,742 |
| Log-prob cao hơn (tổng / trung bình) | cặp | 61,1% / 60,8% | — |
| Probe cặp (train SVAMP) | lần chấm | 67,9–68,6% | 0,734–0,737 |
| Probe cặp + bề mặt | lần chấm | 68,9–69,1% | 0,754–0,759 |
| Bề mặt, dạng pointwise | lời giải | 72,2% (chọn cặp đúng 62,5%) | 0,722 |
| Probe pointwise | lời giải | 79,3% (chọn cặp đúng 74,3%) | 0,838 |
| Probe pointwise + bề mặt | lời giải | 80,0% (chọn cặp đúng 75,7%) | 0,849 |

Tỷ lệ cơ sở để đọc accuracy pointwise: 65,0% lời giải GSM8K là đúng. Giữ hai câu: probe cặp bị
thứ tự lật ở ~44% cặp; logit lens chỉ giải mã được verdict từ ~layer 22 trong khi probe đã đọc được
từ layer 16–21.

**Phase 14 (khám phá; train một dataset, test dataset kia; layer chọn bằng CV trên tập train):**

| Mô hình, train → test | Bề mặt (cặp) acc / AUC | Log-prob | Probe cặp acc / AUC | Probe + bề mặt | Probe pointwise acc / AUC | Probe bị lật |
|---|---|---:|---|---|---|---:|
| Llama, ASDiv → MATH | 64,1% / 0,690 | 50,7% | 66,8% / 0,722 (L20) | 66,5% / 0,720 | 74,7% / 0,803 | 38,7% |
| Llama, MATH → ASDiv | 62,0% / 0,641 | 62,8% | 65,9% / 0,715 (L15) | 66,1% / 0,717 | 66,4% / 0,722 | 39,8% |
| Mistral, ASDiv → MATH | **72,5% / 0,804** | 62,8% | 69,3% / 0,760 (L20) | 70,5% / 0,773 | 75,8% / 0,817 | 40,2% |
| Mistral, MATH → ASDiv | **80,8% / 0,876** | 66,7% | 72,5% / 0,819 (L14) | 73,5% / 0,825 | 72,6% / 0,803 | 24,9% |

- Llama lặp lại kết quả Self_Correction: probe cặp ~66–67% (AUC 0,72), vượt bề mặt một chút, vượt
  log-prob; probe pointwise mạnh hơn (AUC 0,80 theo cặp; 0,80–0,81 theo từng lời giải).
- **Mistral thì ngược lại: đặc trưng bề mặt vượt probe** (AUC 0,80–0,88 so với 0,76–0,82): lời giải sai
  của Mistral thường lộ ra ở bề mặt (dài, lan man, lẫn tiếng Anh, từ chối). Viết: "tín hiệu trong
  hidden state vượt đặc trưng bề mặt ở Self_Correction và Llama, không ở Mistral".
- Probe bị thứ tự lật ở 25–40% cặp (Self_Correction: 44%): không mô hình nào có probe "không bị thứ tự
  ảnh hưởng".

### 4.4 DPO
Giữ bảng. Thêm dòng p13b − base_c: +8,3 [+4,1; +12,5], Holm p = 0,0008. Bản 519 bài: p12 − base
+5,4 (Holm 0,016), p13b − base +5,5 (Holm 0,043), p13b − base_c +8,5 (Holm 0,0008), p13b − p12
+0,2 (0,96).

### 4.5 Mô hình gốc
Bảng hiện tại trộn trần 768 (dòng Self_Correction) với 1536 (dòng Instruct). Dùng 768 cho mọi dòng;
chuyển 1536 xuống Phụ lục C. Cột "hướng bias" lấy từ T6 (tỷ lệ chọn vị trí 1, parser chuẩn).

| Judge → lời giải của | Cặp | Tie-break (parser chuẩn / rescued) | Chọn vị trí 1 [KTC] | Hướng |
|---|---:|---|---|---|
| Self_Correction → của chính nó | 288 / 279 | 0,599 / 0,599 | 44,6% [40,8; 48,4] | nghiêng vị trí 2 |
| Instruct → của chính nó | 264 / 133 | 0,629 / 0,605 | 63,3% [58,7; 67,8] | nghiêng vị trí 1 |
| Instruct → của Self_Correction | 288 / 279 | 0,774 / 0,789 | 51,5% [48,6; 54,4] | cân bằng |
| Self_Correction → của Instruct | 264 / 133 | 0,642 / 0,564 | 48,5% [44,7; 52,5] | cân bằng |

Phụ lục C (1536 token): Instruct → của chính nó 0,655 / 0,658, chọn vị trí 1 65,2% [61,1; 69,4];
Instruct → của Self_Correction 0,819 / 0,839. Câu thêm vào 4.5: cả hai mô hình chỉ thiên lệch vị
trí khi chấm lời giải của **chính mình**; trên lời giải của mô hình kia thì gần 50/50. Số cặp đổi khi
dùng rescued vì nhiều cặp "bất đồng" của Instruct chỉ khác cách viết đáp số.

### 4.6 Lưới tổng quát hóa (Phase 14)
- Hàng: Self_Correction, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3.
  Bỏ hàng "Họ thứ ba (Gemma hoặc Phi)": Gemma đã loại; Phi-3.5-mini chỉ thay thế nếu một mô hình
  trượt smoke test.
- Cột: GSM8K (519, kèm 561), SVAMP, ASDiv, MATH 1–3. **Mọi ô lấy từ lần chạy Phase 14**, kể cả hai
  hàng Qwen; không ghi "đã có" bằng số Phase 13.
- Mỗi ô: skill index [KTC] (E1), mức tăng khử bias log-prob [KTC] (E2), tỷ lệ chọn vị trí 2 và bias
  log-odds, coverage.
- Quy tắc tổng quát (đã đăng ký trước): một claim là tổng quát nếu đạt ở ≥ 3/4 ô của hai mô hình mới
  trên ASDiv và MATH; 1–2 ô thì ghi là có điều kiện.
- **Hình 2 sửa trục:** trục tung là *phần bỏ bias* của mức tăng log-prob, không phải "mức tăng
  tie-break" (vì hiệu tie-break chỉ đo đáp số bịa). Trục hoành là |bias log-odds|.

**Kết quả endpoint (mở một lần, 2026-10-13 03:55 UTC, lock main 5b782972…; judge `base`, parser chuẩn):**

| Ô (endpoint) | Skill index [KTC] (E1) | Một thứ tự | Chọn vị trí 2 | Bịa | Tie-break [KTC] | Đúng khi nhất quán / coverage | E2 |
|---|---|---:|---:|---:|---|---|---|
| Llama · ASDiv (274 cặp) | **+0,321 [0,230; 0,413]** ✅ | 57,3% | 44,3% | 19,3% | 0,646 [0,605; 0,687] | 79,4% / 49,6% | +0,6 ❌ |
| Llama · MATH 1–3 (434) | **+0,362 [0,291; 0,428]** ✅ | 53,0% | 43,9% | 26,7% | 0,628 [0,597; 0,659] | 80,0% / 42,6% | **+5,0** ✅ |
| Mistral · ASDiv (558) | **+0,335 [0,269; 0,400]** ✅ | 51,3% | 45,4% | 29,2% | 0,616 [0,586; 0,645] | 76,8% / 43,2% | +0,7 ❌ |
| Mistral · MATH 1–3 (331) | **+0,344 [0,261; 0,431]** ✅ | 50,5% | 41,9% | 31,4% | 0,609 [0,574; 0,644] | 78,1% / 38,7% | +3,3 ❌ |

- **E1: 4/4 → "tổng quát".** **E2: 1/4 → "có điều kiện".** Dự đoán "phần bỏ bias tăng theo |bias|":
  không ủng hộ (Spearman ρ = −0,20, 4 ô, chỉ mô tả) → Hình 2 có 4 điểm nằm ngang quanh 0.
- Hướng bias: cả hai mô hình mới nghiêng **vị trí 1 (A)** (chọn vị trí 2: 42–45%; log-odds +1,3 đến
  +2,4), giống Qwen-Instruct khi tự chấm (4.5), ngược Self_Correction (nghiêng vị trí 2).
- Mức "đoán mò" một thứ tự (50–57%) và tie-break (0,61–0,65) cùng cỡ Self_Correction trên GSM8K
  (52,3%; 0,601): mảnh 1–2 của claim trung tâm **lặp lại trên hai họ mô hình khác**.
- Judge `base_c` (ép verdict) trên hai mô hình mới (mô tả): Mistral ASDiv skill +0,511, tie-break
  0,749 (bịa chỉ 3,5%); Llama MATH skill +0,159, tie-break 0,573. Prompt ép verdict giảm bịa mạnh
  (Mistral 29% → 3–4%) nhưng tác động lên kỹ năng không đồng nhất.
- Các ô còn lại của lưới (Self_Correction và Qwen-Instruct ở cả 4 dataset; Llama và Mistral ở GSM8K,
  SVAMP) đến từ lần chạy hoàn thiện, được ghi là **mô tả**, không thay đổi E1/E2. **Claim đã khóa
  (L1–L3) và bảng quy tắc diễn giải từng tình huống** ghi trong `phase14/RUN_STATUS.md` trước khi
  chấm các ô mới; tóm tắt ở [`PHASE14_REPORT.md`](summary/sources/PHASE14_REPORT.md) mục 5.
- **Coverage của E2:** mọi lần thiếu log-prob là lần không ghi dòng "Verdict: Solution" (Llama chủ yếu
  do chạm trần 768: 740/840 ở MATH; Mistral do không theo định dạng). Vị trí `verdict` không lấp được;
  thước đo phủ 100% là vị trí `direct` (cột "verdict ép ngay" ở Bảng 3), cho cùng kết luận.
- **Khám phá, post-hoc (không phải endpoint):** phân rã Shapley khoảng cách một thứ tự thành **ba phần —
  thiên lệch vị trí, bị cắt ở trần token, bịa thật** (`paper/scripts/phase14_decomposition.py` →
  `paper/results/phase14_decomposition.md`, KTC bootstrap theo bài; mốc R = accuracy của chính judge nếu không
  bias, không bị cắt, không bịa). Câu chữ: **"Một thứ tự trông như đoán mò chủ yếu vì judge không đưa ra một
  lựa chọn hợp lệ giữa hai lời giải (bịa đáp số, hoặc bị cắt ở trần token), ít hơn nhiều do thiên lệch vị
  trí."** Tỷ lệ (judge `base`): Llama — bị cắt 6–10 điểm, bịa thật 8–14, bias 0,2–0,6; Mistral — bịa thật ~20,
  bị cắt 1–3, bias 0,1–0,5. Kết cục thứ ba của `base_c` gọi là "không có lựa chọn hợp lệ", không gộp với bịa.
  Lần bị cắt phần lớn là lặp vòng (Llama 65–91%), xem `paper/results/phase14_truncation.md` (4 nhãn, 4 mô hình).
  Kiểm tra độ nhạy trần judge 1536 token cho Llama đang xếp hàng; quy tắc đọc ghi trước trong
  `phase14/RUN_STATUS.md`. Đặt ở mục khám phá / phụ lục, kèm KTC.

### 5. Thảo luận
Giữ bảng. Bản 519: giữ s1 65,5%, ngẫu nhiên 65,2%, judge-or-random p12 69,5%, self-check p12 71,7%,
vote@3 71,9%, vote@5 77,1%. Có thể thêm dòng Instruct làm judge trên lời giải Self_Correction:
self-check 74,0% (+2,0 [+0,5; +3,4] so với vote@3, ~3,1 lượt); trên lời giải của chính Instruct chỉ
ngang vote@3 (87,2% so với 87,0%). Dòng probe pointwise: 79,3% (+1,8 [+0,2; +3,4] so với vote@5); bề
mặt pointwise không vượt (78,1%, +0,5 [−0,7; +1,8]). Đánh dấu khám phá.

### 6. Giới hạn
Thêm: (1) thiên lệch vị trí phụ thuộc cả judge lẫn tập cặp (4.5), nên so sánh giữa cột chỉ mang tính
mô tả; (2) parser chuẩn chấm sai đáp án viết thành câu, ảnh hưởng mạnh với mô hình chưa fine-tune
(rescued là kiểm tra); (3) Phase 14 dùng prompt tiếng Việt cho mọi mô hình; (4) mức tăng log-prob
chỉ có cho judge `base_c` (prompt ép verdict); (5) endpoint Phase 14 chỉ gồm hai mô hình mới trên
ASDiv và MATH (cắt theo thời gian GPU, sửa đổi ghi trước khi sinh); các ô còn lại của lưới là mô tả,
sinh sau khi mở endpoint; (6) Mistral giải toán yếu bằng prompt tiếng Việt (toy smoke 5/12; giữ s1 trên
MATH 15,8%) nên tập cặp của nó khác về độ khó; (7) Llama dùng bản sao trọng số đã chứng minh trùng
byte với Meta, không tải trực tiếp từ repo Meta; (8) coverage log-prob của E2 chỉ 50–87% số cặp (cặp
không có log-prob A/B ở cả hai thứ tự bị loại).

### Phụ lục
- C: T3–T6 với cả ba cách đọc đáp số.
- D: Hình 1 = `paper/figures/c5_logit_lens.png` (base_c, p13b; Phase 14 thêm bốn mô hình).
- E: bốn ứng viên (E1: chọn vị trí 1 36,4%, mỗi lần chấm ngang ngẫu nhiên 44,9% so với 45,5%) và
  bảng chứng minh vì sao hiệu tie-break − một thứ tự là thành phần đáp số bịa.

## 6. Việc còn mở

| Việc | Ai / khi nào | Ảnh hưởng |
|---|---|---|
| ~~Chạy Phase 14 endpoint~~ — xong 2026-10-13 (E1 4/4, E2 1/4) | — | 4.6, Bảng 3, Tóm tắt đã điền |
| Lần chạy hoàn thiện lưới 4 × 4 + bridge + regression base_c | GPU (đang chạy từ 05:58 UTC) | các ô mô tả của 4.6, Bảng 3 cho Self_Correction và Qwen, 4.3 cho hai mô hình gốc |
| Tải hidden state về `D:\AGI\outputs\phase14_hidden\` + `HIDDEN_STATE_REPORT.md` | cần chạy ngoài auto mode | lưu trữ, phân tích lại sau |
| KTC cho mọi số 519 bài | CPU, chạy lại `a1_sensitivity.py` có bootstrap | 4.1, 4.2, 4.4 |
| File 513 dòng train của Self_Correction_v1 | người dùng lấy từ máy train/HF | câu về nguồn gốc bias ở 4.5 (hiện ghi "phù hợp với") |
| Xác nhận F2 trên tập mới | Phase 14 (pointwise probe trên ASDiv/MATH) | mục 5 |
