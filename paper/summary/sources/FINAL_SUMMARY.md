# Tổng kết cuối: tự phân xử giữa hai lời giải mâu thuẫn ở mô hình 7B

Cập nhật 2026-10-14. Gộp toàn bộ kết quả Phase 9–13, các kiểm tra paper (A–F, T1–T6) và Phase 14 (4 mô hình ×
4 dataset). Đây là file chốt để viết báo cáo; chi tiết từng phần ở các file được dẫn. Số liệu Phase 14 lấy từ
báo cáo chính thức trên host (`outputs/phase14_remote/outputs/phase14_v1/main/analysis/`), đã đối chiếu khớp với
bản tính lại ở local. KTC = khoảng tin cậy 95% bootstrap theo bài; số đứng trước là ước lượng điểm.

## 0. Kết luận ngắn

1. **Khi một mô hình 7B phân xử giữa hai lời giải mâu thuẫn của chính nó, đo theo một thứ tự trình bày thì
   trông gần như đoán mò (≈ 45–64% với parser chuẩn), nhưng skill index dương rõ ở mọi ô đã đo** — 4/4 ô
   endpoint (Llama-3.1-8B, Mistral-7B × ASDiv, MATH 1–3) và 16/16 ô mô tả còn lại (cả Self_Correction và
   Qwen2.5-7B-Instruct, cả GSM8K và SVAMP). Kỹ năng phân xử có thật và tổng quát qua họ mô hình.
2. **Điều làm một thứ tự trông như đoán mò chủ yếu không phải position bias**, mà là judge không đưa ra một lựa
   chọn hợp lệ giữa hai lời giải (bịa đáp số thứ ba, hoặc bị cắt ở trần token khi đang lặp vòng); thiên lệch
   vị trí chỉ chiếm 0–2,5 điểm. Một phần đáng kể của "bịa" lại là **lỗi đọc đáp số** (mô hình viết đáp số thành
   câu): đọc lại bằng `rescued`, Qwen-Instruct lên 72–88% một thứ tự.
3. **Khử bias bằng trung bình xác suất qua hai thứ tự thêm ít** (E2 có điều kiện, 1/4 endpoint), gần kết quả
   Ren et al. 2023; ở các ô mô tả mức tăng không đồng nhất (0–11 điểm), phần do bỏ bias phần lớn ≈ 0.
4. **Hidden state mang tín hiệu đúng/sai** ở mọi mô hình, nhưng chỉ vượt đặc trưng bề mặt một cách nhất quán
   ở Self_Correction; probe không miễn nhiễm với thứ tự (bị lật ở 22–72% cặp).
5. **Ở mức hệ thống**, tự phân xử chỉ ngang vote@3 cùng chi phí và thua vote@5 ở cả 20 ô.

Hệ quả cho tiêu đề "Position bias che khuất khả năng tự phân xử": phần "kỹ năng bị che khuất" đứng vững, phần
"do position bias" thì không — đề xuất đổi thành dạng "Đo một thứ tự che khuất khả năng tự phân xử" hoặc
"… vì judge không đưa ra lựa chọn hợp lệ, không chủ yếu vì thiên lệch vị trí".

## 1. Thiết lập

| Thành phần | Chi tiết |
|---|---|
| Mô hình | **Self_Correction_v1** (Qwen2.5-7B-Instruct + SFT tự sửa, Phase 1); **Qwen2.5-7B-Instruct** (mô hình gốc); **Llama-3.1-8B-Instruct** (trọng số NousResearch trùng byte Meta, tokenizer Meta); **Mistral-7B-Instruct-v0.3**. Judge LoRA p12, p13b (DPO) chỉ ở Phase 12–13 |
| Dữ liệu | GSM8K test 561 bài (519 sau audit), SVAMP 960 (Phase 12–13); ASDiv 822, MATH level 1–3 785 (Phase 14, mới, không trùng) |
| Lời giải | 5 lời giải/bài, temperature 0,7, không phạt lặp, dùng 768 token đầu |
| Cặp | (s1, s2), (s1, s3) có đáp số khác nhau; chấm cả hai thứ tự; phân tích trên cặp đúng-một |
| Judge | greedy, trần 768; `base` (viết phần kiểm tra, đáp số tự do), `base_c` (ép dòng "Verdict: Solution A/B", có log-prob) |
| Nhãn | verifier SymPy tất định; parser chuẩn là chính, `strict` và `rescued` là độ nhạy |
| Phần cứng | Phase 9–13 V100; kiểm tra paper RTX 3090; Phase 14 V100 32 GB, FP16, vLLM 0.7.0 + Transformers (hidden state) |
| Thống kê | bootstrap theo bài 10.000 lần; Holm khi so nhiều cặp; endpoint đăng ký trước, mở một lần |

Định nghĩa thước đo: [`REPORT_SPEC.md`](../../REPORT_SPEC.md) mục 3; vị trí và lược đồ dữ liệu:
[`../../../phase14/DATA_CATALOG.md`](../../../phase14/DATA_CATALOG.md).

## 2. Toàn vẹn và tái lập

- **A1 nhiễm dữ liệu:** 42/561 bài GSM8K từng xuất hiện trong một lần đánh giá cũ (không trong dữ liệu train);
  loại ra không đổi kết luận (bản 519 bài báo cáo kèm).
- **Kiểm tra hồi quy Phase 14:** judge `base` trên output gốc của báo cáo tái tạo **chính xác tuyệt đối** mọi số
  (skill 0,215 / 0,248; tie-break 0,601 / 0,609); `base_c` chạy lại trong ±0,01 (ngưỡng ±0,03) — "reproduced".
- **Tái lập với mẫu mới:** Self_Correction với lời giải mới cho GSM8K skill 0,191 [0,095; 0,283] (báo cáo 0,215),
  tie-break 0,590 (0,601); SVAMP 0,237 (0,248).
- **Tái lập D1:** Qwen-Instruct chấm lại các cặp Phase 13 (bridge) cho tie-break **0,774** [0,741; 0,808] — trùng
  số D1 cũ (0,774).
- **3090 so với V100 (D0):** 95% lựa chọn trùng.
- Thiết kế Phase 14 đăng ký trước; mọi sửa đổi ghi trước bước bị ảnh hưởng (`phase14/RUN_STATUS.md`); khóa thực
  thi pilot / main / completion.

## 3. Lớp I — phân tích sâu Self_Correction (Phase 9–13 + kiểm tra paper)

| Kiểm tra | Kết quả chính |
|---|---|
| B1 phân rã theo vị trí (judge chưa train, GSM8K) | một thứ tự 52,3%, **skill index +0,215 [0,128; 0,304]**; chọn vị trí 2 53,7%, bịa 16,7% |
| B2 ba con số | tie-break 0,601; đúng khi nhất quán 71,0% ở coverage 47,9% |
| B3 / 4.4 DPO | p12 − base +5,2 điểm tie-break (Holm 0,016); p13b − base_c +8,3 (Holm 0,0008); p13b − p12 −0,2 (n.s.); bịa 16,7% → 2,1% |
| B5 / F3 | judge đóng góp ~+4 điểm so với chọn ngẫu nhiên; trần phân xử hai ứng viên 77,9% ≈ vote@5 77,5% (43% cặp bất đồng sai cả hai) |
| C1–C4 probe | probe cặp 65–69% (AUC ~0,74), không đổi qua 4 judge (DPO không thêm tín hiệu); bị thứ tự lật ~44% cặp |
| C2b, C3 | bề mặt AUC 0,742, probe + bề mặt 0,759; log-prob cao hơn 61%; khớp lần giải greedy 81% (tốn thêm lượt) |
| C5 logit lens | verdict giải mã được từ ~layer 22; probe đã đọc được từ layer 16–21 |
| F2 / F2b (khám phá) | probe pointwise AUC 0,84 (bề mặt pointwise 0,72); vote@5 có trọng số probe 79,3% (+1,8 [+0,2; +3,4]) |
| D1–D3 nguồn bias | Instruct gốc không lệch vị trí trên cặp của Self_Correction (D1, Holm 0,016); thiên lệch theo vị trí, không theo nhãn (D2); nhắc lại đề không đổi đáng kể (D3) |
| D1b | dữ liệu SFT có tiên nghiệm "câu sau là đúng" 513/513 (file train đối chiếu 2026-10-16) — **phù hợp với**, không chứng minh, thiên lệch vị trí 2 của Self_Correction (file train 513 dòng chưa xác nhận) |
| T1–T6 | Instruct tự chấm bài mình: tie-break 0,63–0,66, nghiêng vị trí 1 (63–65%); không mô hình nào lệch vị trí trên bài của mô hình kia; "chấm bài người khác dễ hơn" chỉ đúng một chiều |
| E1 bốn ứng viên | nghiêng vị trí đầu (36%), mỗi lần chấm ngang ngẫu nhiên |

## 4. Phase 14 — endpoint (mở một lần, 2026-10-13 03:55 UTC)

| Ô | Skill index [KTC] — E1 | Log-prob trung bình hai thứ tự [KTC] — E2 |
|---|---|---|
| Llama · ASDiv | **+0,321 [0,230; 0,413]** ✅ | +0,6 điểm [−3,7; +4,8] |
| Llama · MATH 1–3 | **+0,362 [0,291; 0,428]** ✅ | **+5,0 [+0,9; +9,3]** ✅ |
| Mistral · ASDiv | **+0,335 [0,269; 0,400]** ✅ | +0,7 [−2,3; +3,6] |
| Mistral · MATH 1–3 | **+0,344 [0,261; 0,431]** ✅ | +3,3 [−0,7; +7,3] |

**E1 4/4 → tổng quát. E2 1/4 → có điều kiện.** Dự đoán liều–đáp ứng (phần bỏ bias tăng theo |bias|) không được
ủng hộ: ρ = −0,20 (4 ô), −0,27 (16 ô lưới, p = 0,32).

## 5. Lưới 4 × 4 — judge `base`, parser chuẩn

| Mô hình | Dataset | Loại | Cặp | s1 đúng | Một thứ tự | Skill index [KTC] | Chọn vị trí 2 | Bịa | Tie-break [KTC] | Đúng khi nhất quán / coverage |
|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| Llama-3.1-8B | ASDiv | **endpoint** | 274 | 79,6% | 57,3% | 0,321 [0,230; 0,413] | 44,3% | 19,3% | 0,646 [0,605; 0,687] | 79,4% / 49,6% |
| Llama-3.1-8B | MATH 1–3 | **endpoint** | 434 | 44,2% | 53,0% | 0,362 [0,291; 0,428] | 43,9% | 26,7% | 0,628 [0,597; 0,659] | 80,0% / 42,6% |
| Llama-3.1-8B | GSM8K | mô tả | 264 | 74,3% | 50,2% | 0,307 [0,211; 0,403] | 40,9% | 29,2% | 0,604 [0,565; 0,643] | 78,4% / 36,7% |
| Llama-3.1-8B | SVAMP | mô tả | 389 | 75,7% | 56,4% | 0,357 [0,278; 0,434] | 41,7% | 22,1% | 0,647 [0,613; 0,678] | 80,6% / 47,8% |
| Mistral-7B | ASDiv | **endpoint** | 558 | 54,6% | 51,3% | 0,335 [0,269; 0,400] | 45,4% | 29,2% | 0,616 [0,586; 0,645] | 76,8% / 43,2% |
| Mistral-7B | MATH 1–3 | **endpoint** | 331 | 15,8% | 50,5% | 0,344 [0,261; 0,431] | 41,9% | 31,4% | 0,609 [0,574; 0,644] | 78,1% / 38,7% |
| Mistral-7B | GSM8K | mô tả | 411 | 43,1% | 47,4% | 0,255 [0,180; 0,332] | 41,5% | 28,7% | 0,575 [0,543; 0,607] | 69,1% / 39,4% |
| Mistral-7B | SVAMP | mô tả | 681 | 51,8% | 47,4% | 0,254 [0,194; 0,313] | 44,3% | 29,2% | 0,584 [0,559; 0,610] | 69,6% / 43,0% |
| Self_Correction | ASDiv | mô tả | 311 | 79,6% | 55,9% | 0,222 [0,127; 0,313] | 52,2% | 10,9% | 0,603 [0,558; 0,647] | 67,8% / 57,9% |
| Self_Correction | MATH 1–3 | mô tả | 452 | 50,2% | 49,0% | 0,155 [0,078; 0,229] | 55,7% | 19,6% | 0,573 [0,538; 0,608] | 64,0% / 52,2% |
| Self_Correction | GSM8K | mô tả | 277 | 65,8% | 52,2% | 0,191 [0,095; 0,283] | 55,4% | 13,7% | 0,590 [0,547; 0,633] | 67,6% / 51,3% |
| Self_Correction | SVAMP | mô tả | 371 | 82,7% | 56,3% | 0,237 [0,149; 0,324] | 54,8% | 10,0% | 0,598 [0,557; 0,640] | 65,7% / 62,8% |
| Qwen2.5-7B-Instruct | ASDiv | mô tả | 256 | 85,6% | 63,5% | 0,609 [0,532; 0,682] | 36,0% | 25,6% | 0,701 [0,667; 0,736] | 94,8% / 44,9% |
| Qwen2.5-7B-Instruct | MATH 1–3 | mô tả | 380 | 55,9% | 45,1% | 0,466 [0,405; 0,526] | 41,3% | 49,1% | 0,609 [0,584; 0,636] | 91,9% / 26,1% |
| Qwen2.5-7B-Instruct | GSM8K | mô tả | 241 | 81,1% | 56,4% | 0,485 [0,399; 0,567] | 41,9% | 31,1% | 0,651 [0,613; 0,689] | 87,6% / 40,2% |
| Qwen2.5-7B-Instruct | SVAMP | mô tả | 358 | 85,0% | 63,1% | 0,575 [0,514; 0,638] | 33,9% | 22,2% | 0,679 [0,647; 0,709] | 90,0% / 44,7% |

GSM8K sau audit (519 bài): Llama skill 0,320 / tie-break 0,604; Mistral 0,249 / 0,570; Self_Correction 0,186 /
0,585; Qwen 0,487 / 0,646.

**Bridge** (mỗi mô hình chấm lại đúng 288 cặp Phase 13 của Self_Correction; Self_Correction tự chấm: skill 0,215,
tie-break 0,601): Qwen-Instruct skill 0,653, tie-break **0,774**; Llama 0,545 / **0,729**; Mistral 0,302 / 0,595.

**Đọc:**
- Hướng thiên lệch ổn định theo mô hình trên cả 4 dataset: ba mô hình chưa fine-tune nghiêng vị trí đầu (chọn vị
  trí 2: 34–45%), Self_Correction nghiêng vị trí sau (52–56%).
- Thứ hạng skill index: Qwen-Instruct (0,47–0,61) > Llama (0,31–0,36) > Mistral (0,25–0,34) > Self_Correction
  (0,16–0,24). Mô hình đã SFT tự sửa có skill index thấp nhất ở mọi dataset nhưng cũng bịa ít nhất (10–20%).

## 6. Bảng 3 — trung bình log-prob qua hai thứ tự (cùng thang với Ren et al.)

Judge `base_c`, log-prob ở token verdict; mức tăng = accuracy của sign(m_ab − m_ba) − accuracy một thứ tự của
chính các log-prob đó. Ren et al. 2023 (PaLM-2 L, 4 ứng viên): TruthfulQA +0,6; TL;DR +0,5.

| Mô hình | Dataset | Coverage | Bias log-odds (> 0 = A) | Một thứ tự → trung bình | Mức tăng [KTC] | Phần bỏ bias [KTC] |
|---|---|---:|---:|---|---|---|
| Llama | ASDiv · MATH · GSM8K · SVAMP | 50–80% | 1,3–2,6 | 54–67% → 62–69% | +0,6 · **+5,0** · **+10,9** · +2,7 | −0,8 · +1,4 · +2,7 · −0,4 (đều KTC chứa 0) |
| Mistral | ASDiv · MATH · GSM8K · SVAMP | 69–87% | 1,4–2,4 | 63–72% → 66–72% | +0,7 · +3,3 · +0,1 · +2,8 | −0,3 · +1,2 · −1,9 · **+3,1 [1,2; 5,1]** |
| Self_Correction | ASDiv · MATH · GSM8K · SVAMP | 66–74% | 1,3–1,5 | 52–58% → 55–63% | **+5,8** · +2,3 · +2,5 · **+6,9** | **+4,6 [1,1; 8,1]** · +1,1 · +0,3 · **+3,9 [0,5; 7,4]** |
| Self_Correction | cặp GSM8K của báo cáo (Phase 13) | 70% | 1,42 | 56,7% → 64,9% | **+8,2 [3,6; 12,8]** | +2,2 [−1,0; 5,3] |
| Qwen-Instruct | ASDiv · MATH · GSM8K · SVAMP | 84–91% | **3,1–5,3** | 55–59% → 57–63% | **+6,0** · +1,4 · −0,2 · **+4,4** | +2,7 · −0,8 · −0,2 · −0,2 |

Số in đậm: KTC của mức tăng nằm trên 0. **+8,2 điểm là số đúng cho dòng Self_Correction của Bảng 3 cũ** (dòng cũ
"+7,1" là hiệu tie-break − một thứ tự, tức thành phần bịa, không cùng thang Ren et al.). Đọc: đa số ô gần Ren et
al.; ngoại lệ lớn nhất là Llama GSM8K và Self_Correction, và ở đó phần lớn mức tăng đến từ việc đọc hai lần chứ
không từ bỏ bias (trừ Self_Correction ASDiv, SVAMP). Qwen có bias log-prob lớn nhất nhưng phần bỏ bias ≈ 0.
Verdict ép ngay sau prompt (không viết phần kiểm tra, phủ 100% cặp) cho cùng bức tranh: mức tăng −0,4 đến
+7,7 điểm.

## 7. Một thứ tự mất điểm vì đâu (khám phá, post-hoc)

**Phân rã Shapley ba thành phần** (`results/phase14_decomposition.md`): mốc R = accuracy của chính judge nếu không
thiên lệch vị trí, không bị cắt, không bịa; khoảng cách R − một thứ tự chia cho bias / bị cắt ở trần / bịa thật
(trung bình qua 6 thứ tự gỡ; ba phần cộng đúng bằng khoảng cách). Judge `base`:

| Mô hình | Khoảng cách | Do bias | Do bị cắt ở trần | Do bịa thật (đã viết xong) |
|---|---|---|---|---|
| Llama | 14–21 điểm | 0,2–0,6 | 6–10 | 8–14 |
| Mistral | 19–24 | 0,1–0,5 | 0,6–3,1 | **18–20** |
| Self_Correction | 6–12 | 0,0–0,2 | 0,5–3,6 | 5–8 |
| Qwen-Instruct | 20–43 | 0,2–2,5 | 2–15 | **15–29** |

**Lần bị cắt đang làm gì** (`results/phase14_truncation.md`, gán nhãn tự động cho mọi output chạm trần của 4 mô
hình): Llama và Self_Correction phần lớn là **lặp vòng** (65–97% số lần bị cắt); Mistral lặp vòng 30–58% hoặc
**bỏ hai lời giải để tự giải lại** (27–51%); Qwen-Instruct thường **dài nhưng đúng hướng** (25–67%).

**Nâng trần judge lên 1536 token (Llama, ASDiv + MATH):** khoảng cách không đổi (14,3 → 14,5; 19,4 → 19,6;
`base_c` MATH 13,6 → 13,6); tỷ lệ bị cắt giảm một nửa nhưng các lần đó viết tiếp mà vẫn không chọn được lời giải.
Theo quy tắc ghi trước: nâng trần không cứu được.

**Câu chữ:** "Một thứ tự trông như đoán mò chủ yếu vì judge không đưa ra một lựa chọn hợp lệ giữa hai lời giải
(bịa đáp số, hoặc bị cắt ở trần token khi đang lặp vòng), ít hơn nhiều do thiên lệch vị trí." Lý do bias tốn ít:
một thiên lệch vị trí đối xứng chỉ làm giảm accuracy ở bậc hai.

## 8. Độ nhạy: đọc lựa chọn của judge bằng `rescued` (khám phá, mới)

Cặp và nhãn đúng/sai giữ nguyên; chỉ đổi cách đọc đáp số cuối của judge (`results/phase14_rescued_picks.md`):

| Mô hình | Một thứ tự | Skill index | Bịa | Tie-break |
|---|---|---|---|---|
| Self_Correction | 49–56% → 51–61% | 0,16–0,24 → 0,17–0,31 | gần như không đổi | 0,57–0,60 → 0,58–0,64 |
| Llama | 50–57% → 56–60% | 0,31–0,36 → 0,34–0,38 | −2 đến −6 điểm | 0,60–0,65 → 0,65–0,67 |
| Mistral | 47–51% → 56–65% | 0,25–0,34 → 0,38–0,52 | −5 đến −9 điểm | 0,58–0,62 → 0,65–0,71 |
| **Qwen-Instruct** | 45–64% → **72–88%** | 0,47–0,61 → **0,63–0,83** | −15 đến −26 điểm | 0,61–0,70 → **0,79–0,89** |

**Đọc:** với Qwen-Instruct, phần lớn "bịa" là đáp số đúng một trong hai lời giải nhưng viết thành câu; đọc lại
thì mô hình gốc **không còn trông như đoán mò**. Kết luận chính (parser chuẩn, đăng ký trước) không đổi, nhưng phải
báo cáo kèm: mức "đoán mò" của một thứ tự phụ thuộc mạnh vào cách đọc đáp số với mô hình chưa fine-tune.
Probe bề mặt AUC 0,90–0,95 của Qwen (mục 9) nhiều khả năng cũng do lỗi định dạng này.

## 9. Hidden state (khám phá)

| Mô hình | Probe cặp vs bề mặt (AUC), train → test | Probe theo từng lời giải (AUC) | Probe bị thứ tự lật |
|---|---|---|---|
| Self_Correction | **vượt ở mọi hướng**: ASDiv→MATH 0,763 vs 0,620; MATH→ASDiv 0,802 vs 0,654; SVAMP→GSM8K 0,814 vs 0,690 | 0,78–0,86 | 28–45% |
| Llama | vượt trên ASDiv↔MATH (0,722 vs 0,690; 0,715 vs 0,641), không ở SVAMP→GSM8K (0,689 vs 0,735) | 0,69–0,83 | 39–50% |
| Mistral | không vượt (0,760 vs 0,804; 0,819 vs 0,876; 0,811 vs 0,836) | 0,69–0,90 | 22–50% |
| Qwen-Instruct | không vượt (bề mặt 0,91–0,95) | 0,77–0,97 | 23–72% |

Log-prob cao hơn chỉ 48–69%. **Đọc:** tín hiệu đúng/sai có trong hidden state ở cả 4 mô hình, probe theo từng lời
giải mạnh hơn probe cặp; "vượt đặc trưng bề mặt" chỉ đứng được với Self_Correction (và Llama trên cặp endpoint);
không mô hình nào có probe không bị thứ tự ảnh hưởng.

## 10. Giá trị thực tế

Ở cả 20 ô: judge-hoặc-ngẫu-nhiên hơn chọn ngẫu nhiên s1/s2 khoảng 1,5–4 điểm; self-check (không nhất quán → vote@3)
nằm trong ±2,5 điểm của vote@3 với chi phí tương đương; **vote@5 tốt nhất ở mọi ô**. Trần của phân xử hai ứng viên
gần vote@5 vì nhiều cặp bất đồng sai cả hai. Khám phá: vote@5 có trọng số theo probe theo từng lời giải 79,3%
(+1,8 so với vote@5, Self_Correction GSM8K) — cần xác nhận trên tập mới.

## 11. Claim: được viết và không được viết

| Viết | Không viết |
|---|---|
| Đo một thứ tự, phân xử giữa hai lời giải của chính mình trông gần đoán mò; skill index dương rõ ở 20/20 ô, 4 mô hình, 4 dataset (E1 tổng quát) | "Position bias là nguyên nhân chính khiến kỹ năng trông như đoán mò" |
| Phần mất chủ yếu do không có lựa chọn hợp lệ (bịa, bị cắt khi lặp vòng), một phần là lỗi đọc đáp số; bias 0–2,5 điểm | "SFT tự sửa gây ra position bias" (chỉ "phù hợp với" dữ liệu SFT) |
| Hướng thiên lệch là đặc trưng ổn định của từng mô hình (gốc: vị trí đầu; SFT tự sửa: vị trí sau) | "Chấm bài người khác dễ hơn" như quy luật chung |
| Trung bình log-prob hai thứ tự thêm ít ở đa số ô (E2 có điều kiện), gần Ren et al. | "Khử bias giúp nhiều hơn Ren et al." |
| DPO đảo thứ tự cải thiện cách thể hiện (tie-break +5, bịa 17% → 2%), không đổi tín hiệu probe | "Hidden state không bị thứ tự ảnh hưởng" |
| Hidden state mang tín hiệu đúng/sai; vượt bề mặt nhất quán ở Self_Correction | "Hidden state vượt đặc trưng bề mặt" cho mọi mô hình |
| Tự phân xử ngang vote@3 cùng chi phí, thua vote@5 | "Tự phân xử vượt bỏ phiếu" |

**Câu hỏi mở (không thành claim):** mô hình gốc so với mô hình SFT tự sửa — skill index của Qwen-Instruct gấp 2–3 lần
Self_Correction ở cả 4 dataset, và khoảng cách còn rộng hơn khi đọc bằng `rescued` (tie-break 0,79–0,89 so với
0,58–0,64); T1 trước đây (tie-break, parser chuẩn) không thấy khác biệt rõ. Mỗi mô hình chấm lời giải của chính nó
(tập cặp khác nhau, Qwen giải đúng nhiều hơn), nên "SFT làm giảm khả năng phân xử" vẫn là giả thuyết cần một thiết
kế cùng tập cặp.

## 12. Giới hạn

- Endpoint chỉ gồm 2 mô hình × 2 dataset; các ô còn lại là mô tả, sinh sau khi mở endpoint.
- Prompt tiếng Việt cho mọi mô hình; Mistral giải toán yếu (giữ s1 trên MATH 15,8%) và thường trả lời bằng tiếng Anh.
- Đọc đáp số: parser chuẩn chấm sai đáp án viết thành câu; ảnh hưởng mạnh tới Qwen-Instruct và Mistral (mục 8).
- Judge greedy, không phạt lặp: tỷ lệ lặp vòng và phần bị cắt phụ thuộc cách giải mã.
- Log-prob chỉ có cho judge `base_c`; coverage 50–94% (lần chấm không có dòng verdict bị loại).
- Mỗi mô hình chấm lời giải của chính nó, nên so sánh giữa mô hình là mô tả.
- Một lần chạy mỗi ô; batch vLLM 64/256 có thể đổi kết quả FP16 ở mức nhỏ.
- Llama tải từ bản sao trùng byte (repo Meta còn chờ duyệt).
- Phần lớn phân tích ngoài E1/E2 là post-hoc.

## 13. Dữ liệu và tái lập (không cần GPU)

- Dữ liệu sinh ra: `outputs/phase14_remote/outputs/phase14_v1/` (đồng bộ có kiểm tra SHA-256); hidden state:
  `F:\AGI_data\phase14_hidden` (57 file, 19,4 GB; `D:\AGI\outputs\phase14_hidden` là junction).
- Danh mục, lược đồ, SHA-256: [`../../../phase14/DATA_CATALOG.md`](../../../phase14/DATA_CATALOG.md),
  [`../../../phase14/DATA_MANIFEST.json`](../../../phase14/DATA_MANIFEST.json).
- Nạp mọi thứ: `paper/scripts/phase14_data.py`. Tính lại: `phase14_decomposition.py`, `phase14_truncation.py`,
  `phase14_preliminary.py`, `phase14_probe_local.py`, `phase14_catalog.py`.
- Nhật ký và sửa đổi: [`../../../phase14/RUN_STATUS.md`](../../../phase14/RUN_STATUS.md); trạng thái cuối:
  [`../../../phase14/FINAL_STATUS.md`](../../../phase14/FINAL_STATUS.md).

## 14. Việc còn mở

- Chuyển các số ở đây vào [`PHASE14_REPORT.md`](PHASE14_REPORT.md), [`REPORT_SPEC.md`](../../REPORT_SPEC.md),
  [`CHECKLIST_RESULTS.md`](../../CHECKLIST_RESULTS.md) (các file đó còn mang số tạm của từng giai đoạn).
- Quyết định tiêu đề và cách viết câu hỏi mở mẹ–con; nếu muốn trả lời: cho cả hai mô hình chấm cùng một tập cặp.
- Xác nhận F2 (probe-weighted vote) trên tập mới.
- Viết báo cáo theo bản mẫu .docx; commit `phase14/` và `paper/`.
