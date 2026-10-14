# Phase 14 — danh mục dữ liệu (tra cứu và nạp lại, không chạy lại)

Mọi output của Phase 14 được giữ lại để **mọi con số đều tính lại được từ file, không cần GPU**.
File này định nghĩa: dữ liệu nằm ở đâu (mục 1), mỗi file chứa gì (mục 2), các khái niệm và thông số
(mục 3), phần nào là endpoint (mục 4), và lệnh để lấy lại từng kết quả (mục 5). Mục 6 là kiểm kê
sinh tự động kèm SHA-256 (`python paper/scripts/phase14_catalog.py`); bản máy đọc là
[`DATA_MANIFEST.json`](DATA_MANIFEST.json). Nhật ký chạy và mọi sửa đổi: [`RUN_STATUS.md`](RUN_STATUS.md).

## 1. Vị trí

| Nội dung | Vị trí | Ghi chú |
|---|---|---|
| Đề bài + đáp án đúng (đã freeze) | `phase14/data/sources_v1/` (ASDiv, MATH 1–3: pilot + main pool); GSM8K = `phase13/data/sources_v1/holdout.jsonl`; SVAMP = `phase12/data/sources_v1/holdout.jsonl` | băm SHA-256 trong `report.json`; mọi loader kiểm tra hash |
| Cỡ main (từ pilot) | `phase14/data/main_size.json` | ASDiv 822, MATH 785 bài đầu của main pool |
| Lời giải, lời chấm, log-prob, settings | `outputs/phase14_remote/outputs/phase14_v1/<split>/<model>/<dataset>/<stage>/` | bản sao đồng bộ từ GPU host, kiểm tra SHA-256 mỗi lần |
| Phân tích đã mở (endpoint + ô thêm sau) | `outputs/phase14_remote/outputs/phase14_v1/main/analysis/{report.json, probe_report.json, report_tables.md}` | |
| Smoke test (4 mô hình) | `outputs/phase14_remote/outputs/phase14_v1/smoke/` | `<model>.json` (vLLM), `<model>_hidden.json` (Transformers), `<model>_toy.json` |
| Hidden state (mảng số) | `F:\AGI_data\phase14_hidden\main\<model>\<dataset>\{answer,judge,direct}.npz` | `D:\AGI\outputs\phase14_hidden` là junction tới đây; `HIDDEN_STATE_REPORT.md` ở cùng thư mục |
| Tóm tắt hidden state (shape, finite, prefix) | `outputs/phase14_remote/outputs/phase14_v1/hidden/main/<model>/<dataset>/<site>.json` | |
| Log GPU | `outputs/phase14_remote/logs/` (`phase14_main.log`, `phase14_complete.log`, `phase14_long_judge.log`, smoke, pilot, bootstrap) | |
| Trạng thái đồng bộ | `outputs/phase14_remote/SYNC_STATUS.md` | một dòng mỗi 10 phút |
| Kết quả khám phá cho báo cáo | `paper/results/phase14_decomposition.{json,md}`, `paper/results/phase14_truncation.{json,md}` | |
| Code đã khóa | `phase14/scripts/`, khóa `phase14/data/execution_lock_{pilot,main,completion}.json` | |

## 2. Lược đồ file

**`samples/answers.jsonl`** — một dòng mỗi lời giải. `task = [pid, k]` (k = 1: prompt Phase 1; k = 2..5:
prompt giải mù); `output` = góc nhìn chính 768 token (dùng cho mọi phép chấm); `output_full` = toàn bộ đến
1536 token; `generated_tokens`; `hit_primary_cap` (chạm 768); `hit_token_cap` (chạm 1536); `request_seed`;
`index`.

**`judge_<judge>/answers.jsonl`** — một dòng mỗi lần chấm. `task = [pid, k, order]`, `order = "ab"` (s1 hiện
là Solution A) hoặc `"ba"` (s1 là B); `output`; `generated_tokens`; `hit_token_cap`; với `base_c`:
`verdict_logprobs = {"position", "A", "B"}` (log-prob của "A", "B" tại token ngay sau dòng "Verdict:
Solution" cuối; `null` nếu không có dòng đó). Stage `judge_<judge>_1536` (kiểm tra độ nhạy) cùng lược đồ,
trần 1536, kèm `skipped_1536.json`.

**`summary.json`** — `status`, `rows`, `answers_sha256` (SHA-256 sau khi đổi CRLF→LF), `audit_sha256`,
`settings` (`model`, `revision`, `temperature`, `top_p`, `top_k`, `max_new_tokens`, `primary_cap`,
`max_model_len`, `batch_size`, `logprobs`, `gpu`). Stage nhập từ Phase 13/12 có `imported_from`.
**`audit.jsonl`** — nhật ký ghi từng dòng khi sinh (cho phép chạy tiếp sau khi máy sập).

**Hidden state `.npz`** (Transformers, FP16, mọi layer = embedding + các lớp decoder; Qwen 29 × 3584,
Llama/Mistral 33 × 4096):

| Vị trí | Khóa | Ý nghĩa |
|---|---|---|
| `answer` | `tasks` ("pid\|k", k = 1..3), `hidden [N, L, D]`, `mean_logprob`, `answer_tokens`, `prefix_ok` | token cuối của lời giải sau prompt giải; log-prob trung bình của lời giải |
| `judge` | `base_tasks`, `base_hidden`, `base_c_tasks`, `base_c_hidden` ("pid\|k\|order") | token cuối của prompt judge, trước khi judge viết |
| `direct` | `tasks`, `final_logprobs [N, 2]` (log P(A), log P(B)), `lens_margin [N, L]` | prompt `base_c` + "Verdict: Solution" ép ngay; logit lens mọi layer |

**`analysis/report.json`** — `cells["<model>/<dataset>"]["lenient"|"strict"|"rescued"|"lenient_audited"]`
(mọi thước đo của ô, cả hai judge), `E1`, `E2`, `summary` (phán quyết), `opened_at`, `comparison_additions`
(các ô thêm sau khi mở, kèm giờ), `report_reproduction` (kiểm tra hồi quy). **`probe_report.json`** —
probe theo mô hình: verdict ép, bộ chọn train/test giữa dataset, probe lời giải đầu.

## 3. Định nghĩa và thông số

- **Mô hình** (`phase14/scripts/config.py`, revision cố định): `llama31_8b` = trọng số
  `NousResearch/Meta-Llama-3.1-8B-Instruct` @ d10aef79 (SHA-256 4 shard trùng repo Meta @ 0e9e39f2) +
  tokenizer/chat template `RedHatAI/...-FP8-dynamic` @ 442e7f52 (trùng blob Meta); `mistral_7b` =
  `mistralai/Mistral-7B-Instruct-v0.3` @ c170c708; `original_solver` = `Kxck/Self_Correction_v1` @ 6437f947;
  `qwen25_instruct` = `Qwen/Qwen2.5-7B-Instruct` @ a09a3545.
- **Dataset:** `asdiv`, `math13` (MATH level 1–3) — endpoint; `gsm8k13` (561 bài Phase 13; 519 sau audit),
  `svamp12` (960 bài Phase 12) — so sánh; `bridge13` — các cặp Phase 13 (lời giải Self_Correction cũ) do mô
  hình khác chấm lại; `gsm8k13_report`, `svamp12_report` — output gốc của báo cáo, chỉ cho kiểm tra hồi quy.
- **Split:** `pilot` (100 bài mỗi dataset, đã dùng để chốt cỡ main, không dùng lại) và `main`.
- **Sinh lời giải:** 5 lời giải mỗi bài, temperature 0,7, top_p 1, không phạt lặp, seed theo từng request,
  sinh đến 1536 token, **dùng 768 token đầu** (= một lần chạy trần 768 với cùng seed, đã kiểm tra trong smoke).
- **Judge:** greedy, không phạt lặp, trần 768. `base` = prompt Phase 9 (viết phần kiểm tra, đáp số tự do);
  `base_c` = prompt Phase 13 ép dòng "Verdict: Solution A/B", có log-prob top-20 ở token verdict.
- **Cặp được chấm:** (s1, s2) và (s1, s3) có đáp số cuối khác nhau (parser chuẩn, góc nhìn 768), cả hai thứ
  tự. **Cặp đúng-một:** đúng một trong hai lời giải đúng.
- **Đọc đáp số:** `lenient` = parser chuẩn của dự án (chính); `strict` = chỉ `\boxed{}` hoặc dòng có nhãn;
  `rescued` = như `lenient`, nhưng nếu không ra số thì lấy số đầu tiên trên dòng "Đáp số" cuối.
- **Lựa chọn của một lần chấm** (`phase13.scripts.constrained.pick`): `base_c` lấy dòng Verdict cuối nếu có;
  nếu không (và với `base`), so đáp số cuối của judge với hai lời giải → "A", "B", hoặc kết cục thứ ba
  (`base`: **bịa đáp số**; `base_c`: **không có lựa chọn hợp lệ**). Tách thêm theo `hit_token_cap`: **bị cắt
  ở trần** vs **bịa thật / viết xong mà không lựa chọn**.
- **Thước đo** (skill index, tỷ lệ chọn vị trí 2, tie-break, coverage, mức tăng khử bias log-prob, …):
  định nghĩa đầy đủ ở [`paper/REPORT_SPEC.md`](../paper/REPORT_SPEC.md) mục 3; code ở `phase14/scripts/metrics.py`.

## 4. Endpoint so với mô tả

- **Endpoint (đăng ký trước):** Llama và Mistral × ASDiv và MATH, chấm một lần lúc 2026-10-13 03:55 UTC
  (khóa main 5b782972…): E1 4/4 → tổng quát; E2 1/4 → có điều kiện. **Không tính lại**; các ô thêm sau
  không làm thay đổi.
- **Mô tả (sinh sau khi mở, khóa `completion` bbff8d73…):** Self_Correction và Qwen-Instruct trên cả 4
  dataset; Llama và Mistral trên GSM8K, SVAMP; bridge; chạy lại `base_c` cho hồi quy. Quy tắc diễn giải ghi
  trước trong `RUN_STATUS.md`.
- **Khám phá (post-hoc):** phân rã Shapley, gán nhãn lần bị cắt, kiểm tra trần 1536, probe.
- Batch vLLM: 64 cho các ô endpoint của Llama, 256 cho mọi ô còn lại (chỉ đổi thứ tự xử lý; ghi trong
  `settings.batch_size` của từng stage).

## 5. Lấy lại từng kết quả (CPU, không chạy lại mô hình)

| Kết quả | Lệnh / cách đọc |
|---|---|
| Mọi dữ liệu từng ô | `from paper.scripts import phase14_data as P` → `P.samples`, `P.judgments`, `P.correct`, `P.one_right_pairs`, `P.hidden`, `P.settings` (xem docstring) |
| E1, E2, bảng mọi ô | `P.report()` hoặc đọc `main/analysis/report.json`; bảng: `main/analysis/report_tables.md` |
| Probe, verdict ép, logit lens | `P.probe_report()`; tính lại: `phase14/scripts/probe_analysis.py` (cần hidden state trên F:) |
| Phân rã bias / bị cắt / bịa (Shapley, KTC) | `python paper/scripts/phase14_decomposition.py` → `paper/results/phase14_decomposition.md` |
| Lần bị cắt đang làm gì (4 nhãn, 4 mô hình) | `python paper/scripts/phase14_truncation.py` → `paper/results/phase14_truncation.md` |
| Kiểm kê + SHA-256 | `python paper/scripts/phase14_catalog.py` → mục 6 dưới đây + `DATA_MANIFEST.json` |
| Kiểm tra toàn vẹn một stage | `generate.completed()` (dùng trong mọi loader) báo lỗi nếu `answers.jsonl` không khớp `answers_sha256` |
| Kiểm tra khóa code | `python phase14/scripts/verify_lock.py --stage main` (hoặc `pilot`, `completion`) |

Không lưu: prompt đã render (tái tạo đúng từ code + chat template của tokenizer đã pin), và trạng thái
KV/logit đầy đủ của vLLM (chỉ log-prob top-20 ở token verdict của `base_c`).

<!-- INVENTORY: generated by paper/scripts/phase14_catalog.py; do not edit below -->

## 6. Kiểm kê (sinh tự động 2026-10-14 05:38 (local))

- **text_outputs** (`D:\AGI\outputs\phase14_remote\outputs\phase14_v1`): 256 file, 0.44 GB
- **analysis** (`D:\AGI\outputs\phase14_remote\outputs\phase14_v1\main\analysis`): 3 file, 0.00 GB
- **smoke** (`D:\AGI\outputs\phase14_remote\outputs\phase14_v1\smoke`): 12 file, 0.00 GB
- **hidden_summaries** (`D:\AGI\outputs\phase14_remote\outputs\phase14_v1\hidden`): 57 file, 0.00 GB
- **logs** (`D:\AGI\outputs\phase14_remote\logs`): 22 file, 0.00 GB
- **paper_results** (`D:\AGI\paper\results`): 6 file, 0.00 GB
- **frozen_sources** (`D:\AGI\phase14\data`): 11 file, 0.00 GB
- **hidden_arrays** (`F:\AGI_data\phase14_hidden`): 57 file, 19.43 GB

### Mỗi stage sinh dữ liệu (split / mô hình / dataset)

| Split | Mô hình | Dataset | Stage | Dòng | Revision | temperature | max tokens | batch | GPU | sha256 answers (LF) |
|---|---|---|---|---:|---|---:|---:|---:|---|---|
| main | llama31_8b | asdiv | judge_base | 860 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `23baac41e433…` |
| main | llama31_8b | asdiv | judge_base_1536 | 860 | d10aef79 | 0.0 | 1536 | 256 | Tesla V100-SXM2-32GB | `6b66b35f85cb…` |
| main | llama31_8b | asdiv | judge_base_c | 860 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `149517c6f303…` |
| main | llama31_8b | asdiv | judge_base_c_1536 | 860 | d10aef79 | 0.0 | 1536 | 256 | Tesla V100-SXM2-32GB | `0379929faffc…` |
| main | llama31_8b | asdiv | samples | 4110 | d10aef79 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `b6cd5c90115e…` |
| main | llama31_8b | bridge13 | judge_base | 972 | d10aef79 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `53da295663eb…` |
| main | llama31_8b | bridge13 | judge_base_c | 972 | d10aef79 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `3a0db2f5495f…` |
| main | llama31_8b | gsm8k13 | judge_base | 804 | d10aef79 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `6b630afbbc3c…` |
| main | llama31_8b | gsm8k13 | judge_base_c | 804 | d10aef79 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `5543b0609793…` |
| main | llama31_8b | gsm8k13 | samples | 2805 | d10aef79 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `ca644718340b…` |
| main | llama31_8b | math13 | judge_base | 2042 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `ef11bac76162…` |
| main | llama31_8b | math13 | judge_base_1536 | 2042 | d10aef79 | 0.0 | 1536 | 256 | Tesla V100-SXM2-32GB | `25497cc7f31e…` |
| main | llama31_8b | math13 | judge_base_c | 2042 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `c83f2feaaf6a…` |
| main | llama31_8b | math13 | judge_base_c_1536 | 2042 | d10aef79 | 0.0 | 1536 | 256 | Tesla V100-SXM2-32GB | `9c0deeac9d22…` |
| main | llama31_8b | math13 | samples | 3925 | d10aef79 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `96311483a5d5…` |
| main | llama31_8b | svamp12 | judge_base | 1172 | d10aef79 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `d5754ae6fef3…` |
| main | llama31_8b | svamp12 | judge_base_c | 1172 | d10aef79 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `6f2df85a935e…` |
| main | llama31_8b | svamp12 | samples | 4800 | d10aef79 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `231e89e037cd…` |
| main | mistral_7b | asdiv | judge_base | 1866 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `05f9754b07d6…` |
| main | mistral_7b | asdiv | judge_base_c | 1866 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `090841007583…` |
| main | mistral_7b | asdiv | samples | 4110 | c170c708 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `74a1b6421da7…` |
| main | mistral_7b | bridge13 | judge_base | 972 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `7052a2cbc568…` |
| main | mistral_7b | bridge13 | judge_base_c | 972 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `98fc42df3d15…` |
| main | mistral_7b | gsm8k13 | judge_base | 1560 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `09c8062a5913…` |
| main | mistral_7b | gsm8k13 | judge_base_c | 1560 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `63b59b0fd5bf…` |
| main | mistral_7b | gsm8k13 | samples | 2805 | c170c708 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `b5697aded30c…` |
| main | mistral_7b | math13 | judge_base | 2754 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `c749053c575b…` |
| main | mistral_7b | math13 | judge_base_c | 2754 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `84020e68d73d…` |
| main | mistral_7b | math13 | samples | 3925 | c170c708 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `d990fb27662d…` |
| main | mistral_7b | svamp12 | judge_base | 2318 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `24273481b7c3…` |
| main | mistral_7b | svamp12 | judge_base_c | 2318 | c170c708 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `a5bb13078a88…` |
| main | mistral_7b | svamp12 | samples | 4800 | c170c708 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `4e1ae544b850…` |
| main | original_solver | asdiv | judge_base | 892 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `1c90a514cb69…` |
| main | original_solver | asdiv | judge_base_c | 892 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `a3d4c72346c8…` |
| main | original_solver | asdiv | samples | 4110 | 6437f947 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `daf2d70b5ec4…` |
| main | original_solver | gsm8k13 | judge_base | 986 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `44f7adfa958b…` |
| main | original_solver | gsm8k13 | judge_base_c | 986 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `baedcad914c2…` |
| main | original_solver | gsm8k13 | samples | 2805 | 6437f947 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `224f05a5f614…` |
| main | original_solver | gsm8k13_report | judge_base | 972 | nhập từ `outputs/phase13_v1/judge_base` | None | None | None | — | `fc26c14605c9…` |
| main | original_solver | gsm8k13_report | judge_base_c | 972 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `233df3343543…` |
| main | original_solver | math13 | judge_base | 1940 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `5c7a4845468f…` |
| main | original_solver | math13 | judge_base_c | 1940 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `ec1becf0cebe…` |
| main | original_solver | math13 | samples | 3925 | 6437f947 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `ff8b87767d10…` |
| main | original_solver | svamp12 | judge_base | 1000 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `042872800b53…` |
| main | original_solver | svamp12 | judge_base_c | 1000 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `d684fe7ff510…` |
| main | original_solver | svamp12 | samples | 4800 | 6437f947 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `66470e4c0719…` |
| main | original_solver | svamp12_report | judge_base | 1002 | nhập từ `outputs/phase12_v1/judge_base` | None | None | None | — | `8b699e20dbd9…` |
| main | original_solver | svamp12_report | judge_base_c | 1002 | 6437f947 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `1fbac11bfb77…` |
| main | qwen25_instruct | asdiv | judge_base | 722 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `2601a77dac5d…` |
| main | qwen25_instruct | asdiv | judge_base_c | 722 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `65408d79f247…` |
| main | qwen25_instruct | asdiv | samples | 4110 | a09a3545 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `361ce8e8b419…` |
| main | qwen25_instruct | bridge13 | judge_base | 972 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `36d1ed468cdd…` |
| main | qwen25_instruct | bridge13 | judge_base_c | 972 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `bdb0f33c19f9…` |
| main | qwen25_instruct | gsm8k13 | judge_base | 676 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `cec21097cf33…` |
| main | qwen25_instruct | gsm8k13 | judge_base_c | 676 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `137a0ef85e62…` |
| main | qwen25_instruct | gsm8k13 | samples | 2805 | a09a3545 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `7e5458a425e7…` |
| main | qwen25_instruct | math13 | judge_base | 1730 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `77e774a00422…` |
| main | qwen25_instruct | math13 | judge_base_c | 1730 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `12d2661d21ca…` |
| main | qwen25_instruct | math13 | samples | 3925 | a09a3545 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `6dc1d4216dc2…` |
| main | qwen25_instruct | svamp12 | judge_base | 910 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `57854330660c…` |
| main | qwen25_instruct | svamp12 | judge_base_c | 910 | a09a3545 | 0.0 | 768 | 256 | Tesla V100-SXM2-32GB | `de96895542c4…` |
| main | qwen25_instruct | svamp12 | samples | 4800 | a09a3545 | 0.7 | 1536 | 256 | Tesla V100-SXM2-32GB | `de43065fb33a…` |
| pilot | llama31_8b | asdiv | judge_base | 118 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `c8825f2b95e2…` |
| pilot | llama31_8b | asdiv | judge_base_c | 118 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `485c0167aa49…` |
| pilot | llama31_8b | asdiv | samples | 500 | d10aef79 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `5054c9d11860…` |
| pilot | llama31_8b | math13 | judge_base | 244 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `101afa8d9300…` |
| pilot | llama31_8b | math13 | judge_base_c | 244 | d10aef79 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `8ca0541acdbf…` |
| pilot | llama31_8b | math13 | samples | 500 | d10aef79 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `5e474ec77f42…` |
| pilot | mistral_7b | asdiv | judge_base | 228 | c170c708 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `67951d55d78f…` |
| pilot | mistral_7b | asdiv | judge_base_c | 228 | c170c708 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `d4ead4324de7…` |
| pilot | mistral_7b | asdiv | samples | 500 | c170c708 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `a61377d1fa38…` |
| pilot | mistral_7b | math13 | judge_base | 350 | c170c708 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `5039722aeda8…` |
| pilot | mistral_7b | math13 | judge_base_c | 350 | c170c708 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `70f2624c3550…` |
| pilot | mistral_7b | math13 | samples | 500 | c170c708 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `a185182e0993…` |
| pilot | original_solver | asdiv | judge_base | 98 | 6437f947 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `ceb46c21e89f…` |
| pilot | original_solver | asdiv | judge_base_c | 98 | 6437f947 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `bd15f46d850e…` |
| pilot | original_solver | asdiv | samples | 500 | 6437f947 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `88600bcca15c…` |
| pilot | original_solver | math13 | judge_base | 238 | 6437f947 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `f1c670ac1bbf…` |
| pilot | original_solver | math13 | judge_base_c | 238 | 6437f947 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `03e5da296ac5…` |
| pilot | original_solver | math13 | samples | 500 | 6437f947 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `8c98b110c488…` |
| pilot | qwen25_instruct | asdiv | judge_base | 72 | a09a3545 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `5df6e36b187a…` |
| pilot | qwen25_instruct | asdiv | judge_base_c | 72 | a09a3545 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `630ec4918522…` |
| pilot | qwen25_instruct | asdiv | samples | 500 | a09a3545 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `c3539d3ee324…` |
| pilot | qwen25_instruct | math13 | judge_base | 212 | a09a3545 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `48829181bdac…` |
| pilot | qwen25_instruct | math13 | judge_base_c | 212 | a09a3545 | 0.0 | 768 | 64 | Tesla V100-SXM2-32GB | `ed0a1881c45d…` |
| pilot | qwen25_instruct | math13 | samples | 500 | a09a3545 | 0.7 | 1536 | 64 | Tesla V100-SXM2-32GB | `d5d3f98492aa…` |

### Hidden state (F:)

| File | GB | sha256 |
|---|---:|---|
| `main/llama31_8b/asdiv/answer.npz` | 0.57 | `f8318155aab532b2…` |
| `main/llama31_8b/asdiv/direct.npz` | 0.00 | `ab5dab1b1de9127e…` |
| `main/llama31_8b/asdiv/judge.npz` | 0.40 | `c678d4a6684035df…` |
| `main/llama31_8b/bridge13/answer.npz` | 0.39 | `0d0ac7a9287382fb…` |
| `main/llama31_8b/bridge13/direct.npz` | 0.00 | `5609791fed42d7b4…` |
| `main/llama31_8b/bridge13/judge.npz` | 0.45 | `8ee3551a33c3147d…` |
| `main/llama31_8b/gsm8k13/answer.npz` | 0.39 | `62cee15c7ed8d4fb…` |
| `main/llama31_8b/gsm8k13/direct.npz` | 0.00 | `f3c9beb1866dfa78…` |
| `main/llama31_8b/gsm8k13/judge.npz` | 0.37 | `8e992814f2e5fa0c…` |
| `main/llama31_8b/math13/answer.npz` | 0.55 | `05161d11e00cd6ad…` |
| `main/llama31_8b/math13/direct.npz` | 0.00 | `6e605a609b55910b…` |
| `main/llama31_8b/math13/judge.npz` | 0.95 | `00725b25a3f019a4…` |
| `main/llama31_8b/svamp12/answer.npz` | 0.67 | `c3867e7b714e5282…` |
| `main/llama31_8b/svamp12/direct.npz` | 0.00 | `fd74324c680f0999…` |
| `main/llama31_8b/svamp12/judge.npz` | 0.54 | `0cbb3ce2309bfa60…` |
| `main/mistral_7b/asdiv/answer.npz` | 0.57 | `3ba118c905cddc5c…` |
| `main/mistral_7b/asdiv/direct.npz` | 0.00 | `104d826714cf22dc…` |
| `main/mistral_7b/asdiv/judge.npz` | 0.87 | `8c78d5831a7d0fbe…` |
| `main/mistral_7b/bridge13/answer.npz` | 0.39 | `78f3a3d8ebd53acf…` |
| `main/mistral_7b/bridge13/direct.npz` | 0.00 | `cd312686d7419cf9…` |
| `main/mistral_7b/bridge13/judge.npz` | 0.45 | `0486a357b24031a7…` |
| `main/mistral_7b/gsm8k13/answer.npz` | 0.39 | `f9a0d317f8e677ae…` |
| `main/mistral_7b/gsm8k13/direct.npz` | 0.00 | `07000c36b210f560…` |
| `main/mistral_7b/gsm8k13/judge.npz` | 0.73 | `6af6a64a9250210d…` |
| `main/mistral_7b/math13/answer.npz` | 0.55 | `a9817af91fdd4972…` |
| `main/mistral_7b/math13/direct.npz` | 0.00 | `512aface63299a54…` |
| `main/mistral_7b/math13/judge.npz` | 1.28 | `afb37eb51fef5189…` |
| `main/mistral_7b/svamp12/answer.npz` | 0.67 | `1aa0b0a5020bcf6c…` |
| `main/mistral_7b/svamp12/direct.npz` | 0.00 | `94469c4a1e5b8caf…` |
| `main/mistral_7b/svamp12/judge.npz` | 1.08 | `bb6f34c116714959…` |
| `main/original_solver/asdiv/answer.npz` | 0.44 | `9594acb1592f62ee…` |
| `main/original_solver/asdiv/direct.npz` | 0.00 | `24d48e25d7c03429…` |
| `main/original_solver/asdiv/judge.npz` | 0.32 | `206de0b829bb2586…` |
| `main/original_solver/gsm8k13/answer.npz` | 0.30 | `56ea62603683a25e…` |
| `main/original_solver/gsm8k13/direct.npz` | 0.00 | `939ea324f7f2dc17…` |
| `main/original_solver/gsm8k13/judge.npz` | 0.35 | `5b8d58b69faf17c1…` |
| `main/original_solver/math13/answer.npz` | 0.42 | `d0ea85a049624538…` |
| `main/original_solver/math13/direct.npz` | 0.00 | `bd0c0155894b2883…` |
| `main/original_solver/math13/judge.npz` | 0.69 | `a095efe2e65e0e64…` |
| `main/original_solver/svamp12/answer.npz` | 0.52 | `4fe03798b35be524…` |
| `main/original_solver/svamp12/direct.npz` | 0.00 | `3dcd818c42b92a0f…` |
| `main/original_solver/svamp12/judge.npz` | 0.36 | `26a846116dbb9183…` |
| `main/qwen25_instruct/asdiv/answer.npz` | 0.44 | `b53725b462e0c01d…` |
| `main/qwen25_instruct/asdiv/direct.npz` | 0.00 | `71dcd511acac95d1…` |
| `main/qwen25_instruct/asdiv/judge.npz` | 0.26 | `7620ddf13928994d…` |
| `main/qwen25_instruct/bridge13/answer.npz` | 0.30 | `e9a62cd1628dfff1…` |
| `main/qwen25_instruct/bridge13/direct.npz` | 0.00 | `5e40a274eda7e67e…` |
| `main/qwen25_instruct/bridge13/judge.npz` | 0.35 | `d2c9c6a6ccf80152…` |
| `main/qwen25_instruct/gsm8k13/answer.npz` | 0.30 | `b0b55fa8e870d00e…` |
| `main/qwen25_instruct/gsm8k13/direct.npz` | 0.00 | `abeba8fd808a3a54…` |
| `main/qwen25_instruct/gsm8k13/judge.npz` | 0.24 | `87d297a81eb0ebeb…` |
| `main/qwen25_instruct/math13/answer.npz` | 0.42 | `34118007dba39a03…` |
| `main/qwen25_instruct/math13/direct.npz` | 0.00 | `db47a655c990c272…` |
| `main/qwen25_instruct/math13/judge.npz` | 0.62 | `e5519ee0f5faf622…` |
| `main/qwen25_instruct/svamp12/answer.npz` | 0.52 | `b5acbf5c3e781466…` |
| `main/qwen25_instruct/svamp12/direct.npz` | 0.00 | `a20930cf1387743e…` |
| `main/qwen25_instruct/svamp12/judge.npz` | 0.33 | `9dfd511020ce5913…` |
