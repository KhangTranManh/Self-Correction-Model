# Phase 14 — số liệu sau lọc mỗi ô

| Mô hình | Dataset | Bài | s1 ≠ s2 | Cặp được chấm | Cặp đúng-một | Cặp sai cả hai | Lời giải chạm 768 | Judge `base` chạm trần | Judge `base_c` chạm trần |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | ASDiv | 822 | 218 | 430 | 274 | 156 | 1.6% | 18.1% | 14.2% |
| Llama-3.1-8B | MATH 1–3 | 785 | 509 | 1021 | 434 | 587 | 15.7% | 31.9% | 36.5% |
| Llama-3.1-8B | GSM8K | 561 | 201 | 402 | 264 | 138 | 1.5% | 14.7% | 13.8% |
| Llama-3.1-8B | SVAMP | 960 | 304 | 586 | 389 | 197 | 2.4% | 17.9% | 12.8% |
| Llama-3.1-8B | bridge | 561 | 240 | 486 | 288 | 198 | — | 11.2% | 14.4% |
| Mistral-7B | ASDiv | 822 | 469 | 933 | 558 | 375 | 1.8% | 3.5% | 0.3% |
| Mistral-7B | MATH 1–3 | 785 | 681 | 1377 | 331 | 1046 | 15.8% | 16.9% | 1.6% |
| Mistral-7B | GSM8K | 561 | 398 | 780 | 411 | 369 | 2.1% | 3.7% | 0.1% |
| Mistral-7B | SVAMP | 960 | 582 | 1159 | 681 | 478 | 1.4% | 3.0% | 0.1% |
| Mistral-7B | bridge | 561 | 240 | 486 | 288 | 198 | — | 11.9% | 1.3% |
| Self_Correction | ASDiv | 822 | 221 | 446 | 311 | 135 | 1.5% | 7.4% | 5.9% |
| Self_Correction | MATH 1–3 | 785 | 483 | 970 | 452 | 518 | 9.0% | 15.0% | 14.4% |
| Self_Correction | GSM8K | 561 | 255 | 493 | 277 | 216 | 1.7% | 3.9% | 3.3% |
| Self_Correction | SVAMP | 960 | 246 | 500 | 371 | 129 | 1.2% | 3.2% | 2.5% |
| Self_Correction | GSM8K (báo cáo) | 561 | 240 | 486 | 288 | 198 | — | — | 3.7% |
| Self_Correction | SVAMP (báo cáo) | 960 | 246 | 501 | 367 | 134 | — | — | 3.2% |
| Qwen2.5-7B-Instruct | ASDiv | 822 | 178 | 361 | 256 | 105 | 0.6% | 9.6% | 1.8% |
| Qwen2.5-7B-Instruct | MATH 1–3 | 785 | 442 | 865 | 380 | 485 | 10.1% | 27.1% | 7.2% |
| Qwen2.5-7B-Instruct | GSM8K | 561 | 174 | 338 | 241 | 97 | 0.1% | 23.1% | 5.8% |
| Qwen2.5-7B-Instruct | SVAMP | 960 | 238 | 455 | 358 | 97 | 0.2% | 5.4% | 0.9% |
| Qwen2.5-7B-Instruct | bridge | 561 | 240 | 486 | 288 | 198 | — | 20.1% | 4.9% |

Ô bridge dùng lời giải Phase 13 của Self_Correction (không có cờ chạm trần của lời giải); hai ô báo cáo không có cờ chạm trần của judge `base`. GSM8K sau audit (519 bài): xem `report.json` → `lenient_audited`.
