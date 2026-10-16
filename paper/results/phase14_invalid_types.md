# Phase 14 — phân loại 7 kết cục của mỗi lần chấm một thứ tự (khám phá)

Cặp đúng-một, cả hai thứ tự; tỷ lệ theo lần chấm (%). Quy tắc: `phase14/RUN_STATUS.md` (2026-10-15, i). Hai ô báo cáo không có cờ chạm trần nên loại 3–4 luôn bằng 0 ở đó.

## Judge `base`

| Mô hình | Dataset | Lần chấm | 1 hợp lệ | 2 lỗi định dạng | 3 cắt-lặp vòng | 4 cắt-khác | 5 tự giải lại | 6 đáp số thứ ba | 7 không có số |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | ASDiv | 548 | 80.7 | 1.8 | 6.4 | 2.4 | 2.9 | 5.8 | 0.0 |
| Llama-3.1-8B | MATH 1–3 | 868 | 73.3 | 2.9 | 9.7 | 4.6 | 2.5 | 6.9 | 0.1 |
| Llama-3.1-8B | GSM8K | 528 | 70.8 | 6.6 | 6.6 | 2.3 | 7.2 | 6.2 | 0.2 |
| Llama-3.1-8B | SVAMP | 778 | 77.9 | 3.0 | 8.5 | 0.9 | 2.8 | 6.9 | 0.0 |
| Llama-3.1-8B | bridge | 576 | 77.4 | 0.3 | 5.9 | 0.9 | 9.7 | 5.7 | 0.0 |
| Mistral-7B | ASDiv | 1116 | 70.8 | 10.0 | 0.6 | 0.5 | 7.4 | 8.0 | 2.6 |
| Mistral-7B | MATH 1–3 | 662 | 68.6 | 5.0 | 2.0 | 2.3 | 6.9 | 14.4 | 0.9 |
| Mistral-7B | GSM8K | 822 | 71.3 | 8.3 | 0.6 | 0.1 | 9.0 | 9.4 | 1.3 |
| Mistral-7B | SVAMP | 1362 | 70.8 | 9.5 | 1.1 | 0.9 | 9.3 | 6.3 | 2.1 |
| Mistral-7B | bridge | 576 | 62.5 | 11.5 | 1.7 | 5.0 | 9.4 | 9.9 | 0.0 |
| Self_Correction | ASDiv | 622 | 89.1 | 1.3 | 1.8 | 0.5 | 3.2 | 4.0 | 0.2 |
| Self_Correction | MATH 1–3 | 904 | 80.4 | 1.5 | 5.0 | 0.9 | 5.5 | 6.6 | 0.0 |
| Self_Correction | GSM8K | 554 | 86.3 | 0.4 | 0.9 | 0.2 | 4.5 | 7.8 | 0.0 |
| Self_Correction | SVAMP | 742 | 90.0 | 0.7 | 0.8 | 0.0 | 5.0 | 3.5 | 0.0 |
| Self_Correction | GSM8K (báo cáo) | 576 | 83.3 | 0.2 | 0.0 | 0.0 | 6.2 | 10.1 | 0.2 |
| Self_Correction | SVAMP (báo cáo) | 734 | 86.8 | 0.5 | 0.0 | 0.0 | 3.5 | 8.9 | 0.3 |
| Qwen2.5-7B-Instruct | ASDiv | 512 | 74.4 | 18.2 | 1.0 | 2.5 | 1.2 | 2.3 | 0.4 |
| Qwen2.5-7B-Instruct | MATH 1–3 | 760 | 50.9 | 25.9 | 7.9 | 7.9 | 3.7 | 3.4 | 0.3 |
| Qwen2.5-7B-Instruct | GSM8K | 482 | 68.9 | 15.6 | 5.2 | 6.6 | 1.2 | 2.3 | 0.2 |
| Qwen2.5-7B-Instruct | SVAMP | 716 | 77.8 | 16.3 | 1.5 | 1.1 | 0.6 | 2.1 | 0.6 |
| Qwen2.5-7B-Instruct | bridge | 576 | 82.6 | 3.1 | 5.6 | 3.8 | 0.7 | 4.2 | 0.0 |

## Judge `base_c`

| Mô hình | Dataset | Lần chấm | 1 hợp lệ | 2 lỗi định dạng | 3 cắt-lặp vòng | 4 cắt-khác | 5 tự giải lại | 6 đáp số thứ ba | 7 không có số |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | ASDiv | 548 | 94.0 | 0.0 | 4.9 | 0.9 | 0.0 | 0.2 | 0.0 |
| Llama-3.1-8B | MATH 1–3 | 868 | 78.9 | 0.2 | 11.4 | 8.3 | 0.0 | 0.9 | 0.2 |
| Llama-3.1-8B | GSM8K | 528 | 90.3 | 0.0 | 5.1 | 4.2 | 0.2 | 0.0 | 0.2 |
| Llama-3.1-8B | SVAMP | 778 | 93.4 | 0.0 | 5.4 | 1.0 | 0.0 | 0.1 | 0.0 |
| Llama-3.1-8B | bridge | 576 | 93.2 | 0.0 | 5.6 | 0.7 | 0.0 | 0.5 | 0.0 |
| Mistral-7B | ASDiv | 1116 | 96.5 | 2.2 | 0.0 | 0.0 | 0.0 | 1.3 | 0.0 |
| Mistral-7B | MATH 1–3 | 662 | 97.7 | 1.1 | 0.0 | 0.0 | 0.0 | 1.1 | 0.2 |
| Mistral-7B | GSM8K | 822 | 97.6 | 1.2 | 0.0 | 0.0 | 0.0 | 1.2 | 0.0 |
| Mistral-7B | SVAMP | 1362 | 98.0 | 1.3 | 0.0 | 0.0 | 0.0 | 0.6 | 0.1 |
| Mistral-7B | bridge | 576 | 98.8 | 0.7 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 |
| Self_Correction | ASDiv | 622 | 94.2 | 0.0 | 1.6 | 0.5 | 1.0 | 2.7 | 0.0 |
| Self_Correction | MATH 1–3 | 904 | 91.0 | 0.1 | 5.2 | 1.2 | 0.7 | 1.8 | 0.0 |
| Self_Correction | GSM8K | 554 | 94.9 | 0.0 | 0.9 | 0.0 | 2.2 | 2.0 | 0.0 |
| Self_Correction | SVAMP | 742 | 96.5 | 0.3 | 0.8 | 0.0 | 1.2 | 1.1 | 0.1 |
| Self_Correction | GSM8K (báo cáo) | 576 | 95.7 | 0.0 | 0.9 | 0.3 | 0.9 | 2.3 | 0.0 |
| Self_Correction | SVAMP (báo cáo) | 734 | 95.4 | 0.3 | 1.5 | 0.0 | 1.5 | 1.4 | 0.0 |
| Qwen2.5-7B-Instruct | ASDiv | 512 | 98.2 | 0.8 | 0.4 | 0.4 | 0.0 | 0.2 | 0.0 |
| Qwen2.5-7B-Instruct | MATH 1–3 | 760 | 95.7 | 2.0 | 0.4 | 1.3 | 0.0 | 0.5 | 0.1 |
| Qwen2.5-7B-Instruct | GSM8K | 482 | 97.3 | 0.6 | 0.8 | 1.2 | 0.0 | 0.0 | 0.0 |
| Qwen2.5-7B-Instruct | SVAMP | 716 | 99.4 | 0.4 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen2.5-7B-Instruct | bridge | 576 | 98.1 | 0.2 | 0.9 | 0.7 | 0.0 | 0.2 | 0.0 |

KTC 95% từng tỷ lệ có trong `phase14_paper_tables.json` (khóa `classes.<loại>.ci`).
