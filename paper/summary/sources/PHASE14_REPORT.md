# Phase 14 — báo cáo tổng hợp (bản tạm, 2026-10-13)

Kết quả endpoint đã mở một lần và đã khóa. Các ô còn lại của lưới 4 × 4 đang chạy (lần chạy hoàn
thiện) và sẽ được điền vào mục 6 theo đúng quy tắc ở mục 5. Nhật ký đầy đủ và mọi sửa đổi:
[`phase14/RUN_STATUS.md`](../../../phase14/RUN_STATUS.md). Định nghĩa thước đo: [`REPORT_SPEC.md`](../../REPORT_SPEC.md).

## 1. Câu hỏi và thiết kế

Phase 11–13 cho thấy, trên một checkpoint (Self_Correction_v1, fine-tune từ Qwen2.5-7B), lựa chọn
của judge giữa hai lời giải mâu thuẫn của chính nó là hỗn hợp của position bias, một kỹ năng thật
nhưng yếu, và đáp số bịa. Phase 14 hỏi điều đó có giữ ở họ mô hình khác và dataset khác không.

| Hạng mục | Giá trị |
|---|---|
| Mô hình endpoint | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 (chưa fine-tune) |
| Dataset endpoint | ASDiv (822 bài), MATH level 1–3 (785 bài); mới, không trùng phase trước |
| Giao thức | Phase 13 giữ nguyên: 5 lời giải (temperature 0,7), cặp (s1, s2), (s1, s3) bất đồng, judge greedy hai thứ tự; `base` (prompt Phase 9) và `base_c` (ép "Verdict: Solution A/B", có log-prob A/B) |
| Endpoint | **E1**: skill index (judge `base`) có KTC > 0. **E2**: mức tăng khi lấy trung bình log-prob qua hai thứ tự (judge `base_c`) ≥ 3 điểm và KTC > 0 |
| Quy tắc tổng quát | ≥ 3/4 ô → "tổng quát"; 1–2 → "có điều kiện" |
| Phần cứng | Tesla V100-SXM2 32 GB, FP16, vLLM 0.7.0, Transformers 4.48.1 |

## 2. Những gì đã chạy (2026-10-12 → 13)

- Smoke test cả 4 mô hình (vLLM và đường hidden state của Transformers) — đạt kỹ thuật.
- Pilot 4 mô hình × 2 dataset × 100 bài → kích thước main 822 / 785.
- Main, cắt theo thời gian GPU (sửa đổi ghi trước khi sinh): hai mô hình mới × ASDiv, MATH, kèm hidden
  state (answer, judge, direct). **Mở endpoint một lần lúc 03:55 UTC.**
- Probe khám phá cho cả hai mô hình.
- Sửa đổi đáng chú ý: Llama dùng trọng số NousResearch (sha256 trùng Meta) + tokenizer RedHatAI (trùng
  Meta); giữ Mistral dù toy 5/12; Mistral chạy chunk vLLM 256 (Llama 64).

## 3. Kết quả endpoint

| Ô | Cặp đúng-một | Một thứ tự | Skill index [KTC] (E1) | Chọn vị trí 2 | Bịa | Tie-break | Log-prob: một thứ tự → trung bình (E2) |
|---|---:|---:|---|---:|---:|---:|---|
| Llama · ASDiv | 274 | 57,3% | **+0,321 [0,230; 0,413]** ✅ | 44,3% | 19,3% | 0,646 | 61,8 → 62,4: +0,6 [−3,7; +4,8] ❌ |
| Llama · MATH 1–3 | 434 | 53,0% | **+0,362 [0,291; 0,428]** ✅ | 43,9% | 26,7% | 0,628 | 62,4 → 67,4: **+5,0 [+0,9; +9,3]** ✅ |
| Mistral · ASDiv | 558 | 51,3% | **+0,335 [0,269; 0,400]** ✅ | 45,4% | 29,2% | 0,616 | 71,7 → 72,4: +0,7 [−2,3; +3,6] ❌ |
| Mistral · MATH 1–3 | 331 | 50,5% | **+0,344 [0,261; 0,431]** ✅ | 41,9% | 31,4% | 0,609 | 62,7 → 66,0: +3,3 [−0,7; +7,3] ❌ |

- **E1: 4/4 → tổng quát.** Một thứ tự trông như đoán mò (50–57%), nhưng judge phản ứng rõ với việc lời
  giải nào đúng — cùng hình ảnh với Self_Correction trên GSM8K (52,3%; skill +0,215).
- **E2: 1/4 → có điều kiện.** Bias về vị trí A rất lớn trong log-prob (log-odds +1,3 đến +2,4), nhưng
  phần mức tăng do *bỏ bias* chỉ −0,8 đến +1,4 điểm; phần còn lại đến từ việc đọc hai lần. Dự đoán
  "phần bỏ bias tăng theo |bias|" không được ủng hộ (Spearman ρ = −0,20, 4 ô, mô tả).
- **So với Ren et al. 2023** (+0,5 / +0,6 điểm): đo đúng thang, khử bias cũng giúp rất ít ở đây.
- Cả hai mô hình mới nghiêng **vị trí 1**; Self_Correction nghiêng vị trí 2; Qwen-Instruct (tự chấm)
  nghiêng vị trí 1.

## 4. Kết quả bổ sung đã có

**Coverage log-prob (vì sao E2 chỉ phủ 50–87% cặp).** Không lần chấm nào thiếu A/B trong top-20; mọi lần
thiếu là lần không có dòng "Verdict: Solution":

| Ô | Lần chấm `base_c` | Thiếu log-prob | Trong đó chạm trần 768 |
|---|---:|---:|---:|
| Llama · ASDiv | 860 | 143 (17%) | 122 |
| Llama · MATH | 2042 | 840 (41%) | 740 |
| Mistral · ASDiv | 1866 | 349 (19%) | ≈ 0 (không ghi verdict theo định dạng) |
| Mistral · MATH | 2754 | 148 (5%) | 8 |

Vị trí `verdict` (teacher-forced) cũng cần dòng verdict có sẵn nên không lấp được các lần này. Thước
đo kiểu Ren et al. **phủ 100% cặp** đã có ở vị trí `direct` (ép verdict ngay sau prompt, không viết
kiểm tra): Llama 67,2 → 66,8 (ASDiv), 65,0 → 69,8 (MATH); Mistral 64,7 → 67,0, 62,2 → 66,5 — cùng kết
luận: trung bình hai thứ tự thêm −0,4 đến +4,8 điểm.

**Probe (khám phá, train một dataset → test dataset kia):**

| Mô hình | Probe cặp AUC | Bề mặt AUC | Probe pointwise AUC (theo lời giải) | Probe bị thứ tự lật |
|---|---|---|---|---|
| Llama | 0,715–0,722 | 0,641–0,690 | 0,80–0,81 | 39–40% |
| Mistral | 0,760–0,819 | **0,804–0,876** | 0,77–0,81 | 25–40% |

Llama lặp lại Self_Correction (hidden state vượt bề mặt một chút; probe pointwise mạnh hơn probe cặp).
Mistral thì không: đặc trưng bề mặt vượt hidden state — lời giải sai của nó lộ ra ở bề mặt.

**Giá trị thực tế (mô tả):** self-check (không nhất quán → vote@3) ngang vote@3 ở cả 4 ô (Llama ASDiv
83,9 vs 84,3; MATH 52,1 vs 51,2; Mistral 69,3 vs 70,8; 26,1 vs 26,9); vote@5 luôn cao nhất.

**Khám phá, post-hoc — phân rã khoảng cách một thứ tự giữa bias và bịa** (không phải endpoint; claim
đã khóa vẫn dựa trên E1, E2). Script [`scripts/phase14_decomposition.py`](scripts/phase14_decomposition.py)
(CPU), kết quả `paper/results/phase14_decomposition.md`.

- *Mô hình mỗi ô:* bịa với xác suất π; nếu không, chọn A với xác suất σ(β + δ) khi A đúng và σ(β − δ)
  khi B đúng (β = thiên lệch về A, δ = kỹ năng), khớp đúng bảng 2 × 3 của ô.
- *Mốc tham chiếu:* **R = σ(δ), accuracy của chính judge đó nếu không thiên lệch vị trí và không bịa.**
  "Phần mất" luôn tính so với R, không so với 100%.
- *Gán:* giá trị Shapley — tác dụng của gỡ bias và gỡ bịa lấy trung bình qua cả hai thứ tự gỡ, nên
  không phụ thuộc lựa chọn thứ tự; hai phần cộng đúng bằng R − accuracy một thứ tự.
- *KTC:* bootstrap theo bài, 2000 lần.

**Kết cục thứ ba (ngoài chọn A, chọn B) được định nghĩa riêng cho từng judge, không gộp:**
- `base` — **bịa đáp số**: đáp số cuối của judge không khớp lời giải nào.
- `base_c` — **không đưa ra lựa chọn hợp lệ**: không có dòng "Verdict: Solution A/B" *và* đáp số cuối cũng
  không khớp lời giải nào. Đây không phải "bịa" theo nghĩa của `base`: phần lớn là lần chấm bị cắt ở trần
  768 token trước khi viết dòng verdict (bảng nguồn lựa chọn dưới).

| Ô | Judge | Một thứ tự [KTC] | R [KTC] | Khoảng cách [KTC] | Phần do bias [KTC] | Phần do kết cục thứ ba [KTC] |
|---|---|---|---|---|---|---|
| Llama · ASDiv | base | 57,3 [52,2; 62,2] | 71,5 [66,0; 76,7] | 14,3 [11,4; 17,3] | 0,4 [0,1; 1,0] | **13,8 [11,0; 16,7]** |
| Llama · MATH | base | 53,0 [49,1; 56,9] | 72,3 [68,2; 76,5] | 19,4 [17,0; 22,1] | 0,2 [0,0; 0,5] | **19,2 [16,8; 21,8]** |
| Mistral · ASDiv | base | 51,3 [47,5; 55,2] | 72,6 [68,3; 76,7] | 21,3 [18,8; 23,8] | 0,1 [0,0; 0,4] | **21,1 [18,7; 23,7]** |
| Mistral · MATH | base | 50,5 [45,9; 54,9] | 73,9 [68,4; 79,2] | 23,5 [20,1; 27,1] | 0,5 [0,1; 1,1] | **23,0 [19,7; 26,4]** |
| Llama · ASDiv | base_c | 58,8 [54,3; 63,1] | 63,7 [58,8; 68,5] | 5,0 [3,5; 6,8] | 1,1 [0,5; 2,1] | 3,8 [2,6; 5,3] |
| Llama · MATH | base_c | 47,9 [44,3; 51,5] | 61,5 [57,2; 65,7] | 13,6 [11,5; 16,0] | 0,7 [0,3; 1,3] | 13,0 [10,9; 15,2] |
| Mistral · ASDiv | base_c | 73,5 [70,3; 77,0] | 77,2 [74,0; 80,7] | 3,8 [2,7; 5,0] | 1,1 [0,6; 1,8] | 2,6 [1,7; 3,7] |
| Mistral · MATH | base_c | 62,7 [58,1; 67,1] | 67,0 [61,6; 72,4] | 4,3 [2,8; 6,3] | **2,9 [1,7; 4,5]** | 1,4 [0,6; 2,4] |

Chú thích bảng: đơn vị là điểm phần trăm accuracy một thứ tự; số đứng trước là **ước lượng điểm**,
[a; b] là **KTC 95%** bootstrap theo bài. Phần do bias + phần do kết cục thứ ba = khoảng cách (ước lượng
điểm). "Kết cục thứ ba" = bịa đáp số với `base`, không có lựa chọn hợp lệ với `base_c`.

**Nguồn của mỗi lựa chọn một thứ tự** (tỷ lệ lần chấm trên cặp đúng-một):

| Ô | `base`: khớp đáp số | `base`: bịa đáp số (chạm trần 768 / không chạm) | `base_c`: dòng Verdict | `base_c`: không có Verdict, khớp đáp số | `base_c`: không lựa chọn hợp lệ (chạm trần / không chạm) |
|---|---:|---|---:|---:|---|
| Llama · ASDiv | 80,7% | 8,8% / 10,6% | 88,3% | 5,7% | 5,8% / 0,2% |
| Llama · MATH | 73,3% | 14,3% / 12,4% | 66,2% | 12,7% | 19,7% / 1,4% |
| Mistral · ASDiv | 70,8% | 1,2% / 28,0% | 74,4% | 22,1% | 0,0% / 3,5% |
| Mistral · MATH | 68,6% | 4,2% / 27,2% | 90,3% | 7,4% | 0,0% / 2,3% |

So với Phase 13 (Self_Correction, `base_c` thiếu dòng Verdict 19,3%): ở hai mô hình mới, 6–34% lần chấm
`base_c` không có dòng Verdict; phần lớn trong số đó vẫn đọc được lựa chọn qua đáp số, phần còn lại (2–21%)
là "không có lựa chọn hợp lệ", hầu hết do trần 768 ở Llama.

**Đọc (khám phá; giới hạn cho cách viết L1, không đổi claim):**
- Với judge tự do (`base`), accuracy của chính judge nếu không bias và không bịa là ~72–74%; quan sát
  50–57%. **Gần như toàn bộ khoảng cách 14–24 điểm là do bịa đáp số; phần do position bias chỉ
  0,1–0,5 điểm.** Lý do: một thiên lệch vị trí vừa phải (β ≈ 0,2–0,4 so với δ ≈ 1) chỉ làm giảm
  accuracy ở bậc hai, σ(δ) − ½[σ(δ+β) + σ(δ−β)] — cùng lý do E2 thấy trung bình hai thứ tự thêm rất ít.
  Ở Mistral, bịa hầu như không do trần token (27–28% lần chấm bịa mà không chạm trần); ở Llama khoảng
  một nửa số lần bịa đi kèm chạm trần.
- Với judge ép verdict (`base_c`), β lớn hơn (0,5–0,9) và phần do bias là 0,7–2,9 điểm. Ở Mistral MATH,
  phần do bias **2,9 điểm** chiếm phần lớn **khoảng cách 4,3 điểm** (cả hai là ước lượng điểm; phần còn lại
  1,4 điểm là không có lựa chọn hợp lệ). Ở Llama MATH, khoảng cách 13,6 điểm chủ yếu là **không có lựa chọn
  hợp lệ** (13,0 điểm), tức là lần chấm bị cắt ở trần 768 — một giới hạn của trần token, không phải bịa.
  Ép verdict cũng làm R thấp hơn ở ba trên bốn ô (kỹ năng thể hiện kém hơn khi chỉ được chọn A/B).
- Bảng loại cặp cũ (khóa vị trí / bịa / sai nhất quán so với 100%) đã bị thay: nó gán theo thứ tự cố định
  và dùng mốc 100%, nên đã quy nhầm 6–10 điểm cho bias. Bảng loại cặp vẫn có ở mục 2 của file kết quả
  như mô tả, không dùng để gán.

**Ba thành phần: bias / bị cắt ở trần / bịa thật** (Shapley trung bình qua cả 6 thứ tự gỡ; kết cục thứ
ba tách theo việc lần chấm có chạm trần 768 token hay không; ba phần cộng đúng bằng khoảng cách; số đứng
trước là ước lượng điểm, [a; b] là KTC 95% bootstrap theo bài; đơn vị: điểm phần trăm accuracy một thứ
tự). Endpoint (ASDiv, MATH) và ô mô tả đã có (Llama GSM8K, SVAMP từ lần chạy hoàn thiện):

| Ô | Judge | Khoảng cách | Do bias | Do bị cắt | Do bịa thật | Tỷ lệ lần chấm bị cắt / bịa thật |
|---|---|---|---|---|---|---|
| Llama · ASDiv | base | 14,3 [11,5; 17,4] | 0,4 [0,1; 1,1] | 6,2 [4,3; 8,3] | 7,6 [5,7; 9,7] | 8,8% / 10,6% |
| Llama · MATH | base | 19,4 [16,9; 22,0] | 0,2 [0,0; 0,5] | 10,3 [8,4; 12,3] | 8,9 [7,1; 11,0] | 14,3% / 12,4% |
| Llama · GSM8K (mô tả) | base | 21,3 [17,6; 25,3] | 0,6 [0,2; 1,5] | 6,3 [4,6; 8,3] | 14,3 [11,6; 17,5] | 8,9% / 20,3% |
| Llama · SVAMP (mô tả) | base | 16,7 [14,0; 19,7] | 0,6 [0,2; 1,3] | 6,8 [5,1; 8,7] | 9,2 [7,2; 11,4] | 9,4% / 12,7% |
| Mistral · ASDiv | base | 21,3 [18,8; 24,0] | 0,1 [0,0; 0,4] | 0,8 [0,4; 1,3] | 20,3 [17,9; 22,9] | 1,2% / 28,0% |
| Mistral · MATH | base | 23,5 [20,1; 27,1] | 0,5 [0,1; 1,2] | 3,1 [1,8; 4,6] | 20,0 [16,7; 23,2] | 4,2% / 27,2% |
| Llama · ASDiv | base_c | 5,0 [3,5; 6,7] | 1,1 [0,5; 2,1] | 3,7 [2,5; 5,1] | 0,1 [0,0; 0,3] | 5,8% / 0,2% |
| Llama · MATH | base_c | 13,6 [11,5; 16,0] | 0,7 [0,3; 1,3] | 12,1 [10,0; 14,4] | 0,9 [0,4; 1,5] | 19,7% / 1,4% |
| Mistral · ASDiv | base_c | 3,8 [2,7; 5,0] | 1,1 [0,6; 1,8] | 0,0 | 2,6 [1,7; 3,7] | 0,0% / 3,5% |
| Mistral · MATH | base_c | 4,3 [2,8; 6,2] | 2,9 [1,7; 4,5] | 0,0 | 1,4 [0,6; 2,4] | 0,0% / 2,3% |

Với `base_c`, "bịa thật" đọc là "không có lựa chọn hợp lệ dù đã viết xong".

**Câu chữ cho báo cáo (khám phá, đi kèm KTC):** "Một thứ tự trông như đoán mò chủ yếu vì judge không đưa
ra một lựa chọn hợp lệ giữa hai lời giải — bịa ra đáp số thứ ba, hoặc bị cắt ở trần token trước khi kết
luận — và ít hơn nhiều do thiên lệch vị trí." Tỷ lệ theo mô hình, judge tự do (`base`):
- **Llama-3.1-8B:** khoảng cách 14–21 điểm chia gần đều giữa **bị cắt (6–10 điểm)** và **bịa thật (8–14
  điểm)**; bias 0,2–0,6 điểm.
- **Mistral-7B:** gần như toàn bộ là **bịa thật (20 điểm)**; bị cắt 1–3 điểm; bias 0,1–0,5 điểm.
- Với judge ép verdict, Llama mất chủ yếu vì **bị cắt** (4–12 điểm), Mistral chủ yếu vì **bias** (1–3 điểm,
  khoảng cách nhỏ 4 điểm).
- Self_Correction và Qwen-Instruct: điền khi lần chạy hoàn thiện xong (cùng script).

**Lần chấm bị cắt đang làm gì** (script [`scripts/phase14_truncation.py`](scripts/phase14_truncation.py),
gán nhãn tự động cho **mọi output chạm trần của cả 4 mô hình**, cả lời giải s1–s5 lẫn lời chấm;
kết quả đầy đủ `paper/results/phase14_truncation.md`). Bốn nhãn, xét theo thứ tự c → b → d → a:
**(a)** dài nhưng đúng hướng, chưa xong; **(b)** lặp vòng (lặp nguyên văn ≥ 3 lần, hoặc liệt kê theo khuôn
≥ 5 dòng); **(c)** lạc đề / rác / lẫn ngôn ngữ (chuỗi ký tự lặp ≥ 30, chữ ngoài Latin/Việt, hoặc đổi
Việt ↔ Anh giữa bài; viết toàn tiếng Anh không tính); **(d)** bỏ hai lời giải, tự giải lại lan man (chỉ cho
lời chấm). Tỷ lệ trên số lần chạm trần, lời chấm `base`:

| Mô hình (split) | Dataset | Chạm trần | (a) đúng hướng (đang kết luận) | (b) lặp vòng | (c) rác / lẫn | (d) tự giải lại |
|---|---|---:|---|---:|---:|---:|
| Llama (main) | ASDiv | 18,1% | 20,5% (4,5%) | **70,5%** | 0,0% | 9,0% |
| Llama (main) | MATH | 31,9% | 24,9% (6,3%) | **65,0%** | 0,3% | 9,8% |
| Llama (main) | GSM8K | 14,7% | 21,2% (11,9%) | **75,4%** | 0,0% | 3,4% |
| Llama (main) | SVAMP | 17,9% | 6,2% (1,9%) | **91,0%** | 0,0% | 2,9% |
| Mistral (main) | ASDiv | 3,5% | 21,5% (6,2%) | 43,1% | 6,2% | 29,2% |
| Mistral (main) | MATH | 16,9% | 25,6% (5,2%) | 33,5% | 2,8% | **38,1%** |
| Self_Correction (pilot) | MATH | 14,7% | 0,0% | **77,1%** | 5,7% | 17,1% |
| Qwen-Instruct (pilot) | MATH | 29,2% | **40,3%** (11,3%) | 35,5% | 3,2% | 21,0% |

**Đọc:** với Llama và Self_Correction, phần lớn lần bị cắt (65–91%) là **lặp vòng** — mô hình không
"sắp xong", nó mắc kẹt; chỉ 2–12% bị cắt đúng lúc đang kết luận. Mistral bị cắt ít hơn, và khi bị cắt thì
lặp vòng (33–43%) hoặc bỏ hai lời giải để tự giải lại (29–38%). Qwen-Instruct là mô hình duy nhất mà phần
lớn lần bị cắt là **dài nhưng đúng hướng** (40%). Hệ quả dự đoán (kiểm tra bằng judge 1536 token, đang
xếp hàng): nâng trần sẽ chỉ cứu được phần nhỏ (a) của Llama; phần (b) lặp vòng sẽ tiếp tục lặp tới trần mới.
Gán nhãn là heuristic: hai lần sửa sau khi kiểm tra tay ví dụ (ghi trong script), nên dùng như mô tả.

Chạy lại khi có ô mới: `python paper/scripts/phase14_decomposition.py` và
`python paper/scripts/phase14_truncation.py`.

## 5. Claim đã khóa và quy tắc diễn giải các ô mô tả

Ghi trong `RUN_STATUS.md` trước khi bất kỳ ô mới nào được chấm. Tóm tắt:

- **L1** Một thứ tự trông như đoán mò trong khi skill index dương rõ — E1 tổng quát (4/4).
- **L2** Trung bình log-prob qua hai thứ tự thêm ít, như Ren et al. — E2 có điều kiện (1/4); phần bỏ
  bias ≈ 0 dù bias lớn; dự đoán liều–đáp ứng không được ủng hộ.
- **L3** Hidden state mang tín hiệu đúng/sai, vượt bề mặt ở Self_Correction và Llama, không ở Mistral.

Các ô mô tả chỉ **củng cố** hoặc **giới hạn** L1–L3, không đổi phán quyết E1/E2, không cộng vào số ô
endpoint, không tạo claim mới, và được báo cáo đủ. Ví dụ: skill index KTC > 0 → "phù hợp với E1"; KTC
chứa 0 → "không phân biệt được với 0 ở ô này" (giới hạn); KTC < 0 → nêu là phản ví dụ ở tóm tắt và giới
hạn, L1 vẫn ghi "tổng quát trên hai mô hình mới, ASDiv/MATH" và không viết là phổ quát.

## 6. Đang chạy / chưa có

| Phần | Trạng thái (bản đồng bộ cuối 06:22 UTC) |
|---|---|
| Llama, Mistral × GSM8K, SVAMP + bridge | đang chạy (Llama GSM8K: lời giải + `base` xong) |
| Self_Correction, Qwen-Instruct × 4 dataset; regression `base_c`; bridge Qwen | chờ |
| `analyze --add-comparison`, regression, probe, bảng | sau cùng (ước tính ~19:00 UTC) |
| Hidden state thô về `D:\AGI\outputs\phase14_hidden\` + `HIDDEN_STATE_REPORT.md` | đang tải (~4,7 MB/s), tự cập nhật |
| **Kiểm tra độ nhạy trần judge 1536 token** (Llama, ASDiv + MATH, `base` và `base_c`; `ops/judge_long.py`, `ops/run_long_judge.sh`) | xếp hàng trên host, tự chạy ngay sau lần hoàn thiện; ghi ra stage `judge_*_1536`; `phase14_decomposition.py` tự đọc khi có |

## 7. Giới hạn đã biết

Prompt tiếng Việt cho mọi mô hình; Mistral giải toán yếu (toy 5/12, giữ s1 trên MATH 15,8%) nên tập
cặp của nó khó hơn; endpoint chỉ hai mô hình × hai dataset; E2 phụ thuộc mô hình có ghi dòng verdict
(coverage 50–87%; vị trí `direct` phủ 100% cho cùng kết luận); bias đo trên prompt Phase 9/13, một lần
chạy mỗi ô; Llama từ bản sao trọng số đã chứng minh trùng byte với Meta.
