# Việc còn lại trước khi nộp paper

Cập nhật 2026-10-15. Tiêu đề làm việc: *Why Self-Arbitration Looks Like Guessing: A Diagnostic Study of Small Language Models*.

Mọi việc dưới đây chỉ cần **CPU và output đã lưu**. Không mở job GPU mới.
Mọi phân tích mới là **khám phá (post-hoc)**: ghi quy tắc vào `phase14/RUN_STATUS.md` trước khi chạy.

Ký hiệu: `[x]` xong · `[ ]` cần làm · `(tùy chọn)` làm nếu còn thời gian.
## Cần bạn xem xét (để trống, không tự quyết)

| # | Việc | Vì sao cần bạn | Tôi đã chuẩn bị |
|---|---|---|---|
| R1 | ~~Gán `rescued` bởi người độc lập~~ **Đã điền bởi trợ lý 2026-10-16 theo yêu cầu của bạn** (cột `verdict_ok` = yes/no/unclear + `annotator` + `note`: 161 / 4 / 14) | paper vẫn ghi "assistant-annotated"; nếu muốn có thêm một người gán độc lập thì tự làm sau | `paper/results/rescued_audit_sample.csv` |
| R2 | Mục 7 — chốt **tiêu đề** | quyết định tác giả | đề xuất ở mục 7 |
| R3 | Mục 7 — chốt cách viết **mẹ–con** (Qwen vs Self_Correction) | claim nhạy cảm | đề xuất: câu hỏi mở trong thảo luận |
| R4 | Mục 8 — người đọc, endorse, nộp arXiv | cần tài khoản và người thật | — |
| R5 | ~~File 513 dòng train~~ **Đã đối chiếu 2026-10-16**: tìm thấy ngay trong repo `data/processed/phase1_sft_train.jsonl` (bản train; `phase1_sft.jsonl` cùng 513 dòng nhưng còn vết `<thinking>`) | xong; câu "phù hợp với" giữ nguyên, thêm số đếm 513/513 | `paper/CHECKLIST_RESULTS.md` D1b |
| R6 | Tên tác giả, đơn vị, email, lời cảm ơn, nguồn tài trợ GPU | thông tin cá nhân | ô trống trong bản thảo |  **Bạn tự điền** (điền tên/đơn vị/email/cảm ơn/tài trợ vào các ô `\authorinfo{}` trong `main.tex`).
| R7 | Trích dẫn *Try Again, Don't Look Back*, V1, CAPS (trong bản mẫu .docx) | tìm không ra nguồn; nếu bạn có link thì thêm lại | đã bỏ khỏi bản thảo |  **Bạn tự làm**: ghi phiên bản arXiv của *Judging the Judges* (và thêm lại 3 trích dẫn nếu có link).
| R8 | Hội nghị của Huang et al. (ICLR 2024?) và phiên bản *Judging the Judges* cần dẫn | nguồn phụ không thống nhất | ghi arXiv ID |  **Bạn tự làm**: xác nhận hội nghị của Huang et al. (ICLR 2024?).
| R9 | Biên dịch `main.tex` ra PDF và soát bố cục bảng/hình | máy này không có LaTeX (pdflatex/tectonic) | lệnh trong `paper/README.md`; Overleaf cũng được |

## Tiến độ hôm nay (2026-10-15, cập nhật liên tục)

| Giờ | Việc |
|---|---|
| 05:05 | Mục 0 xong (dữ liệu đủ; amendment đã ghi). |
| 05:10 | Bắt đầu `paper/scripts/phase14_paper_tables.py` (mục 1a, 1b-mẫu, 1c, 2a–2c, 3, 5) — đang chạy, ~2 giờ CPU. |
| 05:20 | Mục 6: đã kiểm tra 13/17 trích dẫn trên arXiv (bảng mục 6). |
| 05:30 | Mục 6: kiểm tra xong toàn bộ; 3 trích dẫn không tìm được nguồn → R7. Thêm mục "Cần bạn xem xét" (R1–R8). |
| 05:35 | Mục 4: script `phase14_f2_oos.py` đang chạy. Bắt đầu bản thảo tiếng Anh `paper/manuscript/`. |
| 05:35 | Bản thảo v3 tiếng Anh: `paper/manuscript/main.tex` + `refs.bib` (23 mục, đều đã kiểm tra). Đủ mọi phần; chỗ đỏ `\pending{}` chờ kết quả mục 1, 3, 5; chỗ xanh `\authorinfo{}` = R6. |
| 05:36 | Mục 1a, 1c, 1d, 2a–2c, 3a–3d, 5 xong: bảng `.md` + 3 hình từ `phase14_render_tables.py`. Ghi nhận: π_A ≠ π_B ở Qwen (2c). F2 (mục 4) đang chạy: Llama ASDiv→MATH +3,4 [+0,9; +5,9], MATH→ASDiv +1,7 [+0,2; +3,3]. |
| 05:39 | Mục 1b (bản trợ lý) xong. Hình đổi nhãn sang tiếng Anh. Bắt đầu điền `\pending{}` trong bản thảo. |
| 05:41 | Mục 6, 8 xong (bản thảo đủ số, phụ lục A–F, README). Mục 7: tầng claim đã áp dụng; tiêu đề/SFT để R2–R3. Thêm R9 (biên dịch PDF). Còn: mục 4 (F2 đang chạy). |
| 06:12 | Mục 4 (F2) xong: xác nhận ngoài mẫu cho Llama và Mistral, không cho Self_Correction và Qwen; đã viết vào bản thảo. **Mọi mục không cần bạn đã xong**; còn lại R1–R9. |
| 2026-10-16 | R1 (bản trợ lý) và R5 (513/513) xong. Gộp tổng kết: `paper/summary/SUMMARY_ALL.md`; 4 file gốc chuyển vào `paper/summary/sources/` (FINAL_SUMMARY, PHASE14_PRELIMINARY, PHASE14_REPORT, FINAL_STATUS); đường dẫn đã sửa. |
| 2026-10-17 | Cập nhật tài liệu trước khi đẩy GitHub: `README.md` gốc (trạng thái, bảng Phase 14, bố cục repo), `phase14/README.md`, `paper/README.md` (gói arXiv). Gói nộp: `paper/arxiv/` + `paper/arxiv_submission.zip`. |

Mỗi mục có: **Định nghĩa** (làm chính xác cái gì), **Đầu vào** (file đã có), **Đầu ra** (file sẽ ghi), và
**Ghi chú** (khác với kế hoạch ban đầu, hoặc phần đã có sẵn). Số liệu tham chiếu: [`summary/SUMMARY_ALL.md`](summary/SUMMARY_ALL.md).

**Quy ước dùng chung cho mọi mục**
- *Ô:* một cặp mô hình × dataset. **Lưới chính = 16 ô** (4 mô hình × ASDiv, MATH 1–3, GSM8K, SVAMP). Thêm 3 ô
  bridge (Llama, Mistral, Qwen chấm cặp Phase 13) và 2 ô hồi quy (`gsm8k13_report`, `svamp12_report`) — tổng 21;
  bảng chính chỉ dùng 16 ô lưới, bridge và hồi quy để ở phụ lục.
- *Lần chấm:* một thứ tự (`ab` hoặc `ba`) của một cặp đúng-một. Mọi tỷ lệ "theo lần chấm" lấy mẫu số này.
- *Parser chuẩn* = `lenient` (chính, đăng ký trước); `rescued` = độ nhạy.
- *KTC:* bootstrap theo bài, 2000 lần cho phân tích khám phá (10.000 cho endpoint, đã có).
- *Nạp dữ liệu:* `from paper.scripts import phase14_data as P` (`P.samples`, `P.judgments`, `P.one_right_pairs`,
  `P.correct`, `P.hidden`, `P.report()`). Không đọc file trực tiếp.

---

## 0. Kiểm tra trước khi chạy

- [x] **Output đủ trường cho mọi lần chấm.** Đã kiểm tra 2026-10-15: 42 stage judge (4 mô hình × 4 dataset +
  bridge, `base` và `base_c`) đều có `output` đầy đủ, `hit_token_cap`, `generated_tokens`; mọi stage lời giải có
  `generated_tokens`. **Ngoại lệ:** `original_solver/gsm8k13_report` và `svamp12_report` (judge `base` nhập từ Phase
  13/12) không có cờ chạm trần và số token → các phân tích theo trần (loại 3–4) bỏ qua hai ô này.
- [x] **Hidden state vị trí `answer`** có đủ 4 mô hình × 4 dataset (+ bridge) trên `F:\AGI_data\phase14_hidden`
  (SHA-256 đã đối chiếu). Chỉ có s1–s3 (theo thiết kế), không có s4–s5.
- [x] **Amendment trong `RUN_STATUS.md`** — đã ghi 2026-10-15 trước khi chạy (mục "amendment before the paper
  analyses"). Nội dung (ghi trước khi chạy mục 1–4): (i) bảng 7 loại và thứ tự ưu tiên của
  mục 1; (ii) quy trình và ngưỡng gán tay của 1b; (iii) cận dưới/trên của 2a và hàm link của 2b; (iv) tiêu chí
  của mục 4; (v) quy tắc diễn giải: mọi kết quả ở đây là khám phá, không đổi E1/E2 và claim khóa L1–L3.

---

## 1. Phân loại lựa chọn không hợp lệ (ưu tiên cao nhất)

**Định nghĩa.** Mỗi lần chấm thuộc **đúng một** loại, xét theo thứ tự 1 → 7 (dừng ở loại đầu tiên khớp):

| # | Loại | Quy tắc chính xác | Nguồn có sẵn |
|---|---|---|---|
| 1 | Chọn hợp lệ | `pick()` (parser chuẩn) trả về A hoặc B. Với `base_c`: dòng Verdict cuối, nếu không có thì đáp số cuối khớp A/B | `phase13.scripts.constrained.pick` |
| 2 | Lỗi định dạng | Không thuộc 1, nhưng đáp số đọc bằng `rescued` khớp đúng một lời giải | `phase14.scripts.answers.parse(…, "rescued")` |
| 3 | Bị cắt khi lặp vòng | Không thuộc 1–2, `hit_token_cap = true`, bộ gán nhãn cho nhãn **b** (lặp vòng) | `paper/scripts/phase14_truncation.py`, hàm `label()` |
| 4 | Bị cắt khi đang đúng hướng | Không thuộc 1–2, chạm trần, nhãn khác b (a, c, d) | như trên |
| 5 | Tự giải lại | Không chạm trần, `label(text, judgment=True) == "d"` (60% cuối không nhắc lời giải A/B, có bước giải riêng), đáp số không khớp | như trên |
| 6 | Đáp số thứ ba khác | Không chạm trần, đọc được một số nhưng không khớp lời giải nào, không thuộc 5 | `parse()` khác `None` |
| 7 | Không có đáp số | Không chạm trần, không đọc được số nào (`parse()` và `rescued` đều `None`) | |

Ghi chú về thứ tự: loại 2 xét **trước** chạm trần, vì một lần chấm bị cắt nhưng đã viết đáp số dạng câu vẫn là
lỗi định dạng; loại 7 chỉ áp cho lần chấm **không** chạm trần (bị cắt mà chưa có số đã thuộc 3–4).

- [x] **1a.** *Xong 05:36 — gộp vào `paper/scripts/phase14_paper_tables.py` (hàm `classify`), bảng ở
  `paper/results/phase14_invalid_types.md` (KTC trong `phase14_paper_tables.json`). Kết quả: lần chấm hợp lệ
  51–90% với `base`, 79–99% với `base_c` (16 ô lưới).* Kế hoạch gốc: script mới `paper/scripts/phase14_invalid_types.py`: chạy cho 16 ô lưới + bridge, cả `base` và
  `base_c` (hai ô hồi quy chỉ được loại 1, 2, 5–7). **Đầu ra:** `paper/results/phase14_invalid_types.{json,md}` —
  tỷ lệ 7 loại theo lần chấm, mỗi ô × judge, kèm KTC.
  *Đã có một phần:* tỷ lệ chạm trần và 4 nhãn của lần bị cắt (`results/phase14_truncation.md`); ảnh hưởng của
  `rescued` lên một thứ tự / skill / tie-break (`results/phase14_rescued_picks.md`). Mục 1a gộp hai thứ này
  thành một phân loại loại trừ lẫn nhau.
- [x] **1b.** *Xong 05:39 (bản trợ lý gán; bản người độc lập = R1) — `paper/results/phase14_rescued_audit.md`
  + `rescued_audit_assistant.csv`. 179 dòng: nhầm rõ 4 (2,2%), nhầm + mơ hồ 18 (10,1%). Bản thận trọng: Qwen 0% →
  độ nhạy chính; Llama 10%, Mistral 12% → cảnh báo; Self_Correction 24% (n = 29) → phụ lục. Lỗi đều do đáp số không
  phải số nguyên đơn (căn, phân số, hai giá trị).* Kiểm tra `rescued` bằng tay. **Định nghĩa:** mỗi mô hình lấy ngẫu nhiên 50 lần chấm loại 2 (seed
  cố định, chia đều các dataset), xuất ra `paper/results/rescued_audit_sample.csv` (đoạn cuối 300 ký tự, số
  `rescued` đọc được, hai đáp số A/B). Người gán ghi "đúng" nếu số đó đúng là lựa chọn judge muốn nói. **Ngưỡng
  ghi trước:** tỷ lệ đọc nhầm < 5% → dùng `rescued` như độ nhạy chính; 5–15% → báo cáo kèm cảnh báo; > 15% → chỉ
  để phụ lục. Tôi chuẩn bị file CSV; phần gán nhãn cần người làm (~200 dòng, ~1 giờ).
  *Ghi chú:* tỷ lệ loại 2 dự kiến (từ kiểm tra trước) — Qwen 20–43% số lần "bịa", Mistral 6–19%, Llama 5–15%,
  Self_Correction 2–7%.
- [x] **1c.** *Xong 05:36 — `paper/results/phase14_decomp4.md`. 16 ô lưới, judge `base`: khoảng cách R − một thứ tự
  6,4–43,3 điểm; bias 0,0–2,5; lỗi định dạng 0,2–22,8 (lớn nhất ở Qwen); bị cắt 0,5–13,9; không lựa chọn 2,6–16,4.
  Với `base_c` khoảng cách chỉ còn 2,1–13,6.* Phân rã Shapley với các loại mới thay cho "bịa": các yếu tố là **bias**, **lỗi định dạng (2)**,
  **bị cắt (3+4)**, **không lựa chọn dù viết xong (5+6+7)** — 4 yếu tố, trung bình qua 24 thứ tự gỡ. Mốc R = σ(δ)
  như cũ; δ, β ước lượng chỉ từ lần chấm loại 1. KTC bootstrap theo bài. Mở rộng hàm `shapley3()` trong
  `phase14_decomposition.py` thành `shapleyK()` tổng quát; giữ bản 3 thành phần để đối chiếu.
  *Ghi chú:* bản 3 thành phần đã có (bias / bị cắt / bịa thật) — SUMMARY_ALL mục 7; kết quả dự kiến chỉ tách
  "bịa thật" thành lỗi định dạng và không lựa chọn.
- [x] **1d.** *Xong 05:36 — `paper/figures/invalid_types_base.{png,pdf}` và `invalid_types_base_c.{png,pdf}`, vẽ bằng
  `paper/scripts/phase14_render_tables.py`.* Hình cột chồng: trục x = 16 ô (nhóm theo mô hình), mỗi cột = tỷ lệ 7 loại của judge `base`; hình
  thứ hai cho `base_c` (phụ lục). Màu cố định theo loại; loại 1 ở đáy. Script `paper/scripts/fig_invalid_types.py`
  → `paper/figures/invalid_types_base.png` (+ `.pdf`). Ứng viên hình trung tâm (Figure 2).

**Đầu ra mục 1:** bảng 7 loại, bảng phân rã 4 yếu tố, hình cột chồng, kết quả kiểm tra `rescued`.

---

## 2. Độ nhạy của phân rã

- [x] **2a.** *Xong — `phase14_decomp4.md` mục 2a–2c (gộp vào đây thay cho `phase14_decomp_sensitivity.md`). Shapley
  ≥ cận dưới ở 21/21 ô; vượt cận trên ghi trước π·acc_AB ở 6/21 ô nhưng chỉ tối đa 0,10 điểm (phần tương tác với
  bias bị chia đôi) → coi như bằng cận trên: gần như mọi lần không hợp lệ lẽ ra đã được chấm đúng như lần hợp lệ.*
  **Cận cho phần do lựa chọn không hợp lệ.** Gọi π = tỷ lệ lần chấm không hợp lệ (loại 2–7) của ô,
  acc_AB = accuracy trên các lần chấm hợp lệ. Hai kịch bản phản thực:
  - *Cận dưới* — giả định các lần không hợp lệ, nếu buộc chọn, chỉ đúng 50% (đoán mò):
    mất do không hợp lệ ≥ π × 0,5.
  - *Cận trên* — giả định chúng đúng như các lần hợp lệ: mất ≤ π × acc_AB.
  Báo cáo cận dưới, cận trên và giá trị Shapley của mục 1c; Shapley phải nằm giữa hai cận (kiểm tra trong code).
- [x] **2b.** *Xong — bias với probit lớn nhất 1,9 điểm (logit 2,5) → tiêu chí ≤ 3 điểm ở mọi ô **đạt**. Dòng
  không cần mô hình: **bằng 0 theo định nghĩa**, nên không tính riêng — chỉ trên lần chấm loại 1, cặp không nhất
  quán chọn cùng một vị trí ở hai thứ tự → một đúng một sai → trung bình hai thứ tự = trung bình một thứ tự = 0,5
  (chính là đẳng thức tie-break − một thứ tự = phần không hợp lệ, phụ lục E). Muốn đo bias mà không cần mô hình thì
  phải giả định một phản thực (judge không bias), tức là quay lại mô hình (δ, β).* **Hàm link khác.** Thay σ (logistic) bằng Φ (probit) trong mô hình (δ, β, π), tính lại phần do bias.
  Tiêu chí ghi trước: "position bias chỉ tốn 0–2,5 điểm" đứng nếu với probit phần do bias vẫn ≤ 3 điểm ở mọi ô.
  *Ghi chú:* thêm một dòng không cần mô hình — phần do bias = acc(hai thứ tự lấy trung bình trên cặp nhất quán)
  − acc một thứ tự, chỉ trên lần chấm loại 1 — để có bản không phụ thuộc dạng hàm.
- [x] **2c.** *Xong — đẳng thức đúng tới 1e-16 ở mọi ô. π_A − π_B: KTC chứa 0 ở 15/21 ô; **không** chứa 0 ở 5/5 ô
  Qwen (−7 đến −18 điểm: Qwen bỏ lựa chọn nhiều hơn khi lời giải đúng ở B) và Llama MATH (−6,9). → Với Qwen phép
  tách bias / không hợp lệ không hoàn toàn sạch: một phần nghiêng vị trí đi qua kênh "không chọn". Ghi vào giới hạn.*
  **Phân rã không cần mô hình.** Một thứ tự = (1 − π) × acc_AB (đẳng thức, kiểm tra trong code). Kiểm tra
  π không phụ thuộc vị trí lời giải đúng: π_A (lời giải đúng ở vị trí A) so với π_B, KTC của hiệu; nếu chứa 0 ở
  mọi ô thì phép tách "bias" và "không hợp lệ" là sạch. *Đã có:* π_A, π_B trong `phase14_decomposition.json`
  (khóa `pi_A`, `pi_B`) — chỉ cần thêm KTC của hiệu.

**Đầu ra:** `paper/results/phase14_decomp_sensitivity.md` (bảng phụ lục) và một câu cho mục 5 của paper.

---

## 3. Bảng giá trị thực tế

**Định nghĩa chiến lược** (mỗi bài, mỗi ô; accuracy = tỷ lệ bài mà đáp số được chọn đúng):

| Chiến lược | Quy tắc | Lượt gọi (định nghĩa chính xác) | Đã có trong `report.json` |
|---|---|---|---|
| Giữ s1 | s1 | 1 | `accuracy.keep` |
| Ngẫu nhiên s1/s2 | kỳ vọng ½(đúng s1 + đúng s2) khi bất đồng | 2 | `random_s1_s2` |
| Judge, không nhất quán thì ngẫu nhiên | s1 = s2 → s1; nếu không: lựa chọn nhất quán của judge, không thì ngẫu nhiên | 2 + 2·P(bất đồng) | `judge_or_random_base` (parser chuẩn); **`rescued`: cần tính** |
| Self-check | như trên, không nhất quán → vote@3 (thêm s3) | 2 + 2·P(bất đồng) + P(bất đồng và không nhất quán) | `self_check_base` |
| Agreement-gated vote@3 | s1 = s2 → s1; không thì vote@3 | 2 + P(bất đồng) | `agree_gated` |
| vote@3 | đa số s1–s3 | 3 | `maj3` |
| vote@5 | đa số s1–s5 | 5 | `maj5` |
| Trần | đúng nếu s1 hoặc s2 đúng | — | **cần tính** |

Ghi chú: kế hoạch cũ ghi "~4 lượt" cho judge; con số đúng là 2 + 2·P(bất đồng). Đã đo P(s1 ≠ s2) ở 16 ô:
22–87% (thấp nhất Qwen ASDiv, cao nhất Mistral MATH) → judge 2,4–3,7 lượt; self-check thêm tối đa ~0,5 lượt. Token: cộng `generated_tokens` của các lời giải và lần chấm thực sự dùng.

- [x] **3a.** *Xong — `paper/results/phase14_practical.md` (tính trong `phase14_paper_tables.py`, hàm `practical`).*
  Script `paper/scripts/phase14_practical.py` → `paper/results/phase14_practical.md`: accuracy, lượt gọi
  trung bình, token trung bình cho 8 chiến lược × 16 ô (+ phiên bản judge đọc bằng `rescued`).
- [x] **3b.** *Xong. So với vote@3: judge→ngẫu nhiên −1,7 đến −9,6 điểm (KTC < 0 ở 15/16 ô; bản `rescued` 13/16);
  self-check −2,4 đến +1,1 (KTC > 0 ở 0/16, < 0 ở 3/16); agreement-gated = vote@3 đúng tuyệt đối (cùng quy tắc,
  rẻ hơn: 2,2–2,9 lượt). So với vote@5: mọi chiến lược hai ứng viên thua ở 15–16/16 ô.* Hiệu số so với vote@3 và vote@5, KTC bootstrap ghép cặp theo bài (cùng bài, khác chiến lược).
- [x] **3c.** *Xong — `paper/figures/practical_calls.{png,pdf}`.* Hình accuracy theo lượt gọi trung bình: mỗi mô hình một màu, mỗi chiến lược một điểm, nối các điểm
  theo thứ tự lượt gọi; một hình nhỏ cho mỗi dataset (2 × 2) → `paper/figures/practical_calls.png`.
- [x] **3d.** *Xong — trong tiêu đề từng ô của `phase14_practical.md`: cặp bất đồng sai cả hai 19,7–75,8%. Trần
  (s1 hoặc s2 đúng) chỉ hơn vote@5 tối đa 1,6 điểm, thua ở 7/16 ô → kể cả judge hoàn hảo cũng không hơn vote@5 đáng
  kể.* Tỷ lệ cặp bất đồng sai cả hai mỗi ô (s1 ≠ s2 và cả hai sai) — giải thích vì sao trần của phân xử
  hai ứng viên gần vote@5.
  *Đã có:* kết luận định tính ở SUMMARY_ALL mục 10 (vote@5 tốt nhất ở mọi ô; self-check ±2,5 điểm quanh vote@3).

---

## 4. (tùy chọn) Kiểm tra F2 ngoài mẫu

- [x] *Đã ghi trong `RUN_STATUS.md` (2026-10-15, iv) trước khi chạy.* **Ghi trước:** probe theo từng lời giải (vị trí `answer`, s1–s3) train trên dataset X, test trên dataset Y;
  layer chọn bằng CV 5-fold theo bài trên X; StandardScaler + logistic L2, C = 0,01; trọng số mỗi cụm đáp số = tổng
  P(đúng) của các lời giải trong cụm, chọn cụm lớn nhất; so với vote@3 trên cùng bài. **Tiêu chí:** hiệu số có KTC
  ghép cặp nằm trên 0 ở ≥ 2/3 cặp train/test của một mô hình thì ghi "xác nhận ngoài mẫu cho mô hình đó".
- [x] *Xong 06:12 — `paper/results/phase14_f2_oos.{json,md}`. Xác nhận ngoài mẫu: **Llama có** (2/3 cặp, +1,7 đến
  +3,4 điểm so với vote@3), **Mistral có** (3/3, +3,2 đến +4,4); Self_Correction không (1/3; MATH→ASDiv cận dưới
  đúng 0), Qwen không (0/3, vote@3 đã cao). AUC test 0,77–0,90. Đã đưa vào mục Hidden states của `main.tex`.*
  Chạy cho 4 mô hình × (ASDiv→MATH, MATH→ASDiv, SVAMP→GSM8K).
  *Đầu vào:* `P.hidden(model, ds, "answer")` (khóa `tasks` "pid|k", `hidden [N, L, D]`). *Đầu ra:*
  `paper/results/phase14_f2_oos.md`.
  *Ghi chú:* probe pointwise đã có trong `probe_report.json` (AUC 0,69–0,97) nhưng chưa có bước bỏ phiếu có trọng
  số. Không có s4–s5 nên không kiểm tra được bản vote@5 (ghi vào giới hạn).

---

## 5. Bảng số liệu sau lọc

- [x] *Xong — `paper/results/phase14_counts.md` (21 ô).* Mỗi ô (16 + bridge + hồi quy): số bài; số bài có s1 ≠ s2; số cặp được chấm; số cặp đúng-một; số cặp sai cả
  hai; tỷ lệ lời giải chạm trần 768 (`hit_primary_cap`); tỷ lệ lần chấm chạm trần (`base`, `base_c`). Thêm cột GSM8K
  sau audit (519). → `paper/results/phase14_counts.md`, đưa vào phụ lục A.
  *Đã có rải rác:* `one_right_pairs`, `judged_pairs`, `cap_hit_rate` trong `report.json`; chỉ cần gom lại.

---

## 6. Related work và trích dẫn

- [x] *Xong (trừ R7, R8): ID ghi vào `paper/manuscript/refs.bib` (23 mục).* **Kiểm tra từng trích dẫn** (đọc abstract + bảng/số được dẫn; ghi arXiv ID vào `paper/references.bib`):

| Trích dẫn | Trạng thái | Dùng cho |
|---|---|---|
| Ren et al. 2023, *Self-Evaluation Improves Selective Generation in LLMs* (arXiv 2312.09300) | ✅ (Bảng 4: +0,61 / +0,50) | Bảng 3, so sánh khử bias |
| Huang et al., *LLMs Cannot Self-Correct Reasoning Yet* (arXiv 2310.01798; ICLR 2024 — xác nhận lại hội nghị) | ✅ | bối cảnh tự sửa: không có phản hồi ngoài thì tự sửa không cải thiện |
| Tyen et al., *LLMs cannot find reasoning errors, but can correct them given the error location* (arXiv 2311.08516; Findings of ACL 2024) | ✅ | khó ở khâu phát hiện, không ở khâu sửa |
| Kadavath et al. 2022, *Language Models (Mostly) Know What They Know* (arXiv 2207.05221) | ✅ | tự đánh giá, P(True) |
| Kumar et al., SCoRe (arXiv 2409.12917; ICLR 2025) | ✅ | huấn luyện tự sửa bằng RL |
| Qu et al., RISE (arXiv 2407.18219; NeurIPS 2024) | ✅ | tự sửa nhiều lượt, mô hình 7B |
| Zheng et al. 2023, MT-Bench / Chatbot Arena (arXiv 2306.05685; NeurIPS 2023 D&B) | ✅ | position, verbosity, self-enhancement bias của judge |
| Wang et al., *LLMs are not Fair Evaluators* (arXiv 2305.17926; ACL 2024) | ✅ **thêm** | position bias; giải thích trước rồi chấm |
| Shi et al., *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge* (arXiv 2406.07791; có nhiều phiên bản — ghi rõ v) | ✅ | bias phụ thuộc khoảng cách chất lượng giữa hai lời giải |
| Tan et al., JudgeBench (arXiv 2410.12784; ICLR 2025) | ✅ | judge mạnh chỉ hơn đoán mò chút trên cặp có nhãn đúng/sai khách quan |
| Whitehouse et al., J1 (arXiv 2505.10320; ICLR 2026) | ✅ | huấn luyện judge bằng GRPO, thưởng nhất quán thứ tự |
| *Mitigating Selection Bias in LLMs via Permutation-Aware GRPO* (PA-GRPO, arXiv 2603.21016) | ✅ (mới đọc phần phân rã; đọc toàn văn trước khi dẫn) | tách bias nhãn và bias vị trí |
| Panickssery et al., *LLM Evaluators Recognize and Favor Their Own Generations* (arXiv 2404.13076; NeurIPS 2024) | ✅ | self-preference: giả thuyết cạnh tranh |
| Chen et al., Universal Self-Consistency (arXiv 2311.17311) | ✅ | mô hình tự chọn câu trả lời nhất quán nhất |
| *Try Again, Don't Look Back*, V1, CAPS | ⚠️ **không tìm được nguồn chắc chắn** → bỏ khỏi bản thảo; xem R7 | — |

- [x] **Bài probe "model biết nhiều hơn nó thể hiện"** (đã kiểm tra abstract):
  Burns et al., CCS (arXiv 2212.03827); Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying*
  (arXiv 2304.13734, Findings EMNLP 2023); Marks & Tegmark, *The Geometry of Truth* (arXiv 2310.06824); Orgad et
  al., *LLMs Know More Than They Show* (arXiv 2410.02707, ICLR 2025 — đáng chú ý: probe chuyển giữa dataset kém,
  giống quan sát của chúng ta là probe không vượt bề mặt ở mọi mô hình).
- [x] *Xong — đoạn "Positioning" cuối mục Related work của `main.tex`.* **Đoạn định vị** (một đoạn): đã biết — position bias ở LLM-as-judge, chấm hai thứ tự, tự sửa khó với mô hình
  nhỏ, probe đọc được tính đúng; đóng góp — (i) phân rã một thứ tự thành kỹ năng / bias / không lựa chọn hợp lệ trên
  lời giải tự sinh có nhãn tất định, (ii) cho thấy bias tốn ít, không lựa chọn hợp lệ tốn nhiều, (iii) kiểm định
  đăng ký trước trên 2 họ mô hình + 2 dataset mới, (iv) khử bias kiểu Ren et al. thêm ít.

---

## 7. Quyết định viết (không chạy)

- [x] *Đã áp dụng trong bản thảo (endpoint = "general/conditional"; mọi phân tích mới ghi "exploratory"). Bạn duyệt lại cùng R2–R3.* **Gắn mỗi kết quả vào một tầng** (đề xuất, đã khớp quy tắc khóa trong `RUN_STATUS.md`):

| Tầng | Kết quả | Cách viết |
|---|---|---|
| **Xác nhận** (đăng ký trước, mở một lần) | E1 4/4: skill index > 0 (Llama, Mistral × ASDiv, MATH); E2 1/4: khử bias log-prob ≥ 3 điểm chỉ ở Llama MATH | "tổng quát" / "có điều kiện" |
| **Tái lập** | hồi quy Phase 13/12 khớp tuyệt đối; Self_Correction mẫu mới; D1 bridge 0,774 | "tái lập" |
| **Mô tả** | 16 ô lưới: skill index > 0 ở 16/16; hướng bias ổn định theo mô hình; Bảng 3 mở rộng | "phù hợp với E1", "không đồng nhất" |
| **Khám phá** | phân rã Shapley, 7 loại, `rescued`, độ nhạy 1536, probe, mẹ–con, F2 | luôn kèm "khám phá", KTC, không claim nhân quả |

- [ ] **(R2)** **Chốt tiêu đề.** *1c và 2a–2b đã xác nhận "invalid choices, not position bias" (bias ≤ 2,5 điểm, probit ≤ 1,9) → phụ đề đó dùng được, trừ lưu ý Qwen ở 2c.* Đề xuất ở đầu file phù hợp với kết quả (SUMMARY_ALL mục 0 khuyên không đặt "position bias"
  làm nguyên nhân trong tiêu đề). Phương án ngắn: *Why Self-Arbitration Looks Like Guessing*; phụ đề nêu "invalid
  choices, not position bias" chỉ sau khi 1c và 2a–2b xác nhận.
- [ ] **(R3)** **Chốt cách viết về SFT** *(R5 đã xong: 513/513 dòng train đều là "lần đầu bị kiểm tra SAI → bản sửa đã verify")*: *bản thảo đang dùng đúng câu đề xuất dưới đây (phụ lục B, C; thảo luận).* "Mô hình đã SFT tự sửa nghiêng vị trí sau và có skill index thấp nhất; dữ liệu SFT
  có tiên nghiệm 'câu sau là đúng' — **phù hợp với** quan sát này." Câu hỏi mẹ–con (skill Qwen gấp 2–3 lần) để ở
  thảo luận như câu hỏi mở, nêu lý do không kết luận được (mỗi mô hình chấm tập cặp khác nhau).

---

## 8. Hoàn thiện bản thảo

- [x] *Xong — `paper/manuscript/main.tex` không còn `\pending{}`; bảng 3–5 sinh tự động vào `paper/manuscript/tables/*.tex`.* Sửa bản nháp v2 theo kết quả mục 1–5; mọi số lấy từ `paper/results/*` (không gõ tay).
- [x] *Xong — tóm tắt/kết luận theo số mới (51–90% hợp lệ; judge thua vote@3 ở 15/16 ô).* Viết lại tóm tắt và kết luận theo tầng claim của mục 7.
- [x] *Xong — A (bảng số liệu + kiểm tra `rescued`), B (Self_Correction), C (bridge), D (cắt trần), E (đẳng thức), F (trỏ tới catalog).* Phụ lục (nguồn đã có cho từng phần):

| Phụ lục | Nội dung | Nguồn |
|---|---|---|
| A | Toàn vẹn dữ liệu: audit GSM8K 42/561, nguồn ASDiv/MATH, khóa, bảng số liệu mục 5 | `CHECKLIST_RESULTS.md` A1–A4, `phase14/DATA_CATALOG.md` |
| B | Phân tích sâu Self_Correction (B1–B5, C1–C5, D, F) | `CHECKLIST_RESULTS.md`, SUMMARY_ALL mục 3 |
| C | Bridge và T1–T6 | SUMMARY_ALL mục 2, 5; `CHECKLIST_RESULTS.md` T1–T6 |
| D | Phân loại chạm trần, độ nhạy 1536, độ nhạy phân rã (mục 2) | `results/phase14_truncation.md`, `phase14_decomposition.md` |
| E | Đẳng thức tie-break − một thứ tự = phần bịa; bốn ứng viên; so sánh Ren et al. | `CHECKLIST_RESULTS.md` E1–E2 |
| F | Prompt, siêu tham số, revision mô hình, mã, hash | `phase14/DATA_CATALOG.md` mục 3, `DATA_MANIFEST.json` |

- [x] *Xong — `paper/README.md`.* Tái lập: `paper/README.md` ngắn — một lệnh cho mỗi bảng/hình, chỉ dùng output đã lưu (mọi script đã dùng
  `phase14_data.py`).
- [x] *Xong (biên dịch PDF chưa làm: máy không có LaTeX — xem R9).* Chuyển sang tiếng Anh; làm tròn: tỷ lệ 1 chữ số thập phân, skill index và tie-break 3 chữ số, KTC cùng độ
  chính xác; bảng LaTeX `booktabs`.
- [ ] **(R4)** Nhờ một người đọc và endorse; nộp arXiv (cs.CL, cross-list cs.LG).

---

## Đã xong (chỉ cần dẫn nguồn trong paper)

- [x] Tính lại 20 ô tại local, khớp báo cáo trên host (`results/phase14_preliminary.md`).
- [x] Kiểm tra hồi quy: tái tạo đúng tuyệt đối số Phase 13 (`report.json` → `report_reproduction`).
- [x] KTC bootstrap theo bài cho mọi ô.
- [x] Mô hình (δ, β, π) và phân rã Shapley ba thành phần (`results/phase14_decomposition.md`).
- [x] Độ nhạy trần 1536 token cho Llama (khoảng cách không đổi; `RUN_STATUS.md`).
- [x] Đẳng thức tie-break − một thứ tự = phần bịa (kiểm tra trong code).
- [x] Endpoint E1/E2 mở một lần; Spearman liều–đáp ứng (ρ = −0,20 trên 4 ô, −0,27 trên 16 ô).
- [x] Probe, baseline bề mặt, baseline likelihood cho 4 mô hình (`probe_report.json`); DPO (Phase 12–13).
- [x] Audit nhiễm GSM8K; nguồn ASDiv/MATH không trùng phase trước.
- [x] Trần của phân xử hai ứng viên (F3).
- [x] Độ nhạy `rescued` cho một thứ tự / skill / tie-break (`results/phase14_rescued_picks.md`).
- [x] Gán nhãn 4 loại cho lần bị cắt, cả 4 mô hình (`results/phase14_truncation.md`).
- [x] Danh mục dữ liệu, manifest SHA-256, module nạp (`phase14/DATA_CATALOG.md`, `paper/scripts/phase14_data.py`).
- [x] Tổng kết một file (`paper/summary/SUMMARY_ALL.md`).

## Không làm (ghi vào giới hạn / hướng tiếp theo)

- Holdout mới để xác nhận phần khám phá.
- Cặp trung lập cho câu hỏi mẹ–con (cả hai mô hình chấm cùng một tập cặp).
- Lưới temperature và repetition penalty (judge greedy, không phạt lặp — ảnh hưởng tỷ lệ lặp vòng).
- Probe-weighted vote@5 (không có hidden state s4–s5).
- Model family thứ năm.

## Thứ tự và ước lượng

0 (amendment, 15 phút) → 1 (1a–1c ~3 giờ CPU + 1b gán tay ~1 giờ; 1d ~1 giờ) → 2 (~2 giờ) → 3 (~2 giờ) →
5 (~30 phút) → (4) (~2 giờ, probe trên F:) → 6 (~1–2 ngày đọc) → 7 (~1 giờ quyết định) → 8 (~1 tuần viết).
Giới hạn thời gian: bản nháp hoàn chỉnh trong ~2 tuần.
