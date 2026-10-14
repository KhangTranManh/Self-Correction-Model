# Phase 14 — đánh giá sơ bộ mọi ô (mô tả)

Parser chuẩn (`lenient`), cặp đúng-một. Ô endpoint lấy nguyên từ báo cáo đã mở; ô khác tính tại local bằng cùng `metrics.cell`. Số đứng trước là ước lượng điểm, [a; b] là KTC 95% bootstrap theo bài.

## Judge `base`

| Mô hình | Dataset | Nguồn | Cặp | s1 đúng | Một thứ tự | Skill index [KTC] | Chọn vị trí 2 | Bịa | Tie-break [KTC] | Đúng khi nhất quán / coverage |
|---|---|---|---:|---:|---:|---|---:|---:|---|---|
| llama31_8b | asdiv | endpoint | 274 | 79.6% | 57.3% | 0.321 [0.230; 0.413] | 44.3% | 19.3% | 0.646 [0.605; 0.687] | 79.4% / 49.6% |
| llama31_8b | math13 | endpoint | 434 | 44.2% | 53.0% | 0.362 [0.291; 0.428] | 43.9% | 26.7% | 0.628 [0.597; 0.659] | 80.0% / 42.6% |
| llama31_8b | gsm8k13 | mô tả | 264 | 74.3% | 50.2% | 0.307 [0.210; 0.404] | 40.9% | 29.2% | 0.604 [0.565; 0.644] | 78.4% / 36.7% |
| llama31_8b | svamp12 | mô tả | 389 | 75.7% | 56.4% | 0.357 [0.282; 0.433] | 41.7% | 22.1% | 0.647 [0.613; 0.680] | 80.6% / 47.8% |
| llama31_8b | bridge13 | mô tả | 288 | 66.1% | 65.3% | 0.545 [0.468; 0.619] | 48.7% | 22.6% | 0.729 [0.693; 0.764] | 91.8% / 54.9% |
| mistral_7b | asdiv | endpoint | 558 | 54.6% | 51.3% | 0.335 [0.269; 0.400] | 45.4% | 29.2% | 0.616 [0.586; 0.645] | 76.8% / 43.2% |
| mistral_7b | math13 | endpoint | 331 | 15.8% | 50.5% | 0.344 [0.261; 0.431] | 41.9% | 31.4% | 0.609 [0.574; 0.644] | 78.1% / 38.7% |
| mistral_7b | gsm8k13 | mô tả | 411 | 43.1% | 47.4% | 0.255 [0.178; 0.332] | 41.5% | 28.7% | 0.575 [0.544; 0.607] | 69.1% / 39.4% |
| mistral_7b | svamp12 | mô tả | 681 | 51.8% | 47.4% | 0.254 [0.194; 0.314] | 44.3% | 29.2% | 0.584 [0.558; 0.611] | 69.6% / 43.0% |
| mistral_7b | bridge13 | mô tả | 288 | 66.1% | 45.7% | 0.302 [0.219; 0.384] | 38.1% | 37.5% | 0.595 [0.568; 0.625] | 84.8% / 27.4% |
| original_solver | asdiv | mô tả | 311 | 79.6% | 55.9% | 0.222 [0.127; 0.312] | 52.2% | 10.9% | 0.603 [0.557; 0.648] | 67.8% / 57.9% |
| original_solver | math13 | mô tả | 452 | 50.2% | 49.0% | 0.155 [0.080; 0.229] | 55.7% | 19.6% | 0.573 [0.538; 0.607] | 64.0% / 52.2% |
| original_solver | gsm8k13 | mô tả | 277 | 65.8% | 52.2% | 0.191 [0.095; 0.286] | 55.4% | 13.7% | 0.590 [0.548; 0.632] | 67.6% / 51.3% |
| original_solver | svamp12 | mô tả | 371 | 82.7% | 56.3% | 0.237 [0.151; 0.323] | 54.8% | 10.0% | 0.598 [0.556; 0.639] | 65.7% / 62.8% |
| original_solver | gsm8k13_report | mô tả | 288 | 66.1% | 52.3% | 0.215 [0.123; 0.305] | 53.8% | 16.7% | 0.601 [0.561; 0.642] | 71.0% / 47.9% |
| original_solver | svamp12_report | mô tả | 367 | 82.1% | 56.3% | 0.248 [0.166; 0.331] | 54.3% | 13.2% | 0.609 [0.567; 0.649] | 68.7% / 58.3% |
| qwen25_instruct | asdiv | mô tả | 256 | 85.6% | 63.5% | 0.609 [0.532; 0.684] | 36.0% | 25.6% | 0.701 [0.665; 0.736] | 94.8% / 44.9% |
| qwen25_instruct | math13 | mô tả | 380 | 55.9% | 45.1% | 0.466 [0.406; 0.527] | 41.3% | 49.1% | 0.609 [0.584; 0.635] | 91.9% / 26.1% |
| qwen25_instruct | gsm8k13 | mô tả | 241 | 81.1% | 56.4% | 0.485 [0.402; 0.568] | 41.9% | 31.1% | 0.651 [0.613; 0.689] | 87.6% / 40.2% |
| qwen25_instruct | svamp12 | mô tả | 358 | 85.0% | 63.1% | 0.575 [0.513; 0.637] | 33.9% | 22.2% | 0.679 [0.648; 0.709] | 90.0% / 44.7% |

## Judge `base_c`: log-prob ở token verdict

| Mô hình | Dataset | Skill index `base_c` | Coverage log-prob | Bias log-odds (> 0 = A) | Một thứ tự → trung bình | Mức tăng [KTC] | Phần bỏ bias / hai lần đọc |
|---|---|---|---:|---:|---|---|---|
| llama31_8b | asdiv | 0.230 | 79.6% | 1.651 | 61.8% → 62.4% | 0.006 [-0.037; 0.048] | -0.008 / 0.014 |
| llama31_8b | math13 | 0.159 | 50.2% | 1.347 | 62.4% → 67.4% | 0.050 [0.009; 0.093] | 0.014 / 0.037 |
| llama31_8b | gsm8k13 | 0.117 | 76.1% | 2.608 | 54.2% → 65.2% | 0.109 [0.058; 0.162] | 0.027 / 0.082 |
| llama31_8b | svamp12 | 0.316 | 77.4% | 1.522 | 66.7% → 69.4% | 0.027 [-0.008; 0.063] | -0.004 / 0.032 |
| llama31_8b | bridge13 | 0.573 | 79.5% | 0.092 | 81.9% → 85.6% | 0.037 [0.008; 0.067] | 0.002 / 0.035 |
| mistral_7b | asdiv | 0.511 | 68.8% | 1.891 | 71.7% → 72.4% | 0.007 [-0.023; 0.036] | -0.003 / 0.009 |
| mistral_7b | math13 | 0.281 | 87.0% | 2.396 | 62.7% → 66.0% | 0.033 [-0.007; 0.073] | 0.012 / 0.021 |
| mistral_7b | gsm8k13 | 0.445 | 82.5% | 2.229 | 70.1% → 70.2% | 0.001 [-0.035; 0.039] | -0.019 / 0.021 |
| mistral_7b | svamp12 | 0.433 | 73.0% | 1.411 | 68.1% → 70.8% | 0.028 [0.002; 0.053] | 0.031 / -0.003 |
| mistral_7b | bridge13 | 0.194 | 93.8% | 1.736 | 59.6% → 61.9% | 0.022 [-0.024; 0.069] | 0.006 / 0.017 |
| original_solver | asdiv | 0.119 | 68.5% | 1.540 | 57.6% → 63.4% | 0.058 [0.010; 0.105] | 0.046 / 0.012 |
| original_solver | math13 | 0.024 | 65.5% | 1.456 | 52.4% → 54.7% | 0.023 [-0.021; 0.067] | 0.011 / 0.012 |
| original_solver | gsm8k13 | 0.094 | 66.1% | 1.281 | 56.6% → 59.0% | 0.025 [-0.025; 0.073] | 0.003 / 0.022 |
| original_solver | svamp12 | 0.116 | 73.6% | 1.486 | 53.2% → 60.1% | 0.069 [0.027; 0.109] | 0.039 / 0.029 |
| original_solver | gsm8k13_report | 0.139 | 70.1% | 1.420 | 56.7% → 64.9% | 0.082 [0.036; 0.128] | 0.022 / 0.059 |
| original_solver | svamp12_report | 0.174 | 71.4% | 1.515 | 57.1% → 62.8% | 0.057 [0.013; 0.100] | 0.063 / -0.006 |
| qwen25_instruct | asdiv | 0.129 | 87.9% | 5.086 | 55.3% → 61.3% | 0.060 [0.009; 0.112] | 0.027 / 0.033 |
| qwen25_instruct | math13 | 0.142 | 83.7% | 4.478 | 55.3% → 56.8% | 0.014 [-0.031; 0.061] | -0.008 / 0.022 |
| qwen25_instruct | gsm8k13 | 0.183 | 89.2% | 5.322 | 58.8% → 58.6% | -0.002 [-0.056; 0.052] | -0.002 / 0.000 |
| qwen25_instruct | svamp12 | 0.190 | 91.3% | 3.091 | 58.3% → 62.7% | 0.044 [0.002; 0.085] | -0.002 / 0.046 |

## Chiến lược (accuracy theo bài)

| Mô hình | Dataset | Giữ s1 | Ngẫu nhiên s1/s2 | Judge hoặc ngẫu nhiên (base) | Self-check (base) | vote@3 | vote@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| llama31_8b | asdiv | 79.6% | 79.3% | 81.8% | 83.9% | 84.3% | 86.3% |
| llama31_8b | math13 | 44.2% | 43.9% | 47.4% | 52.1% | 51.2% | 57.3% |
| llama31_8b | gsm8k13 | 74.3% | 71.9% | 74.2% | 77.0% | 77.4% | 82.5% |
| llama31_8b | svamp12 | 75.7% | 75.2% | 78.6% | 81.9% | 82.0% | 85.4% |
| llama31_8b | bridge13 | 66.1% | 65.7% | 71.3% | 74.3% | 72.0% | 77.5% |
| mistral_7b | asdiv | 54.6% | 57.7% | 61.6% | 69.3% | 70.8% | 77.0% |
| mistral_7b | math13 | 15.8% | 19.6% | 22.1% | 26.1% | 26.9% | 30.6% |
| mistral_7b | gsm8k13 | 43.1% | 45.5% | 48.3% | 55.6% | 57.4% | 65.6% |
| mistral_7b | svamp12 | 51.8% | 55.6% | 58.6% | 67.0% | 68.5% | 75.7% |
| mistral_7b | bridge13 | 66.1% | 65.7% | 67.6% | 71.8% | 72.0% | 77.5% |
| original_solver | asdiv | 79.6% | 79.5% | 81.1% | 83.3% | 85.2% | 88.3% |
| original_solver | math13 | 50.2% | 48.5% | 50.8% | 54.0% | 55.2% | 61.0% |
| original_solver | gsm8k13 | 65.8% | 64.2% | 66.5% | 69.2% | 70.6% | 75.8% |
| original_solver | svamp12 | 82.7% | 81.7% | 83.5% | 85.2% | 86.6% | 91.1% |
| original_solver | gsm8k13_report | 66.1% | 65.7% | 67.8% | 70.6% | 72.0% | 77.5% |
| original_solver | svamp12_report | 82.1% | 81.6% | 84.3% | 86.5% | 87.6% | 90.1% |
| qwen25_instruct | asdiv | 85.6% | 84.7% | 87.5% | 91.1% | 91.0% | 92.5% |
| qwen25_instruct | math13 | 55.9% | 54.1% | 57.0% | 66.2% | 66.4% | 71.6% |
| qwen25_instruct | gsm8k13 | 81.1% | 77.5% | 81.3% | 85.9% | 84.3% | 88.2% |
| qwen25_instruct | svamp12 | 85.0% | 82.7% | 86.3% | 90.2% | 89.7% | 92.2% |

## Kiểm tra hồi quy với số báo cáo (Phase 13/12)

| Ô | Judge | Thước đo | Báo cáo | Tính lại | Chênh |
|---|---|---|---:|---:|---:|
| original_solver/gsm8k13_report | base | one_right_pairs | 288 | 288.000 | +0.000 |
| original_solver/gsm8k13_report | base | single_accuracy | 0.523 | 0.523 | -0.000 |
| original_solver/gsm8k13_report | base | skill_index | 0.215 | 0.215 | +0.000 |
| original_solver/gsm8k13_report | base | position_2_rate | 0.537 | 0.537 | +0.000 |
| original_solver/gsm8k13_report | base | invented_rate | 0.167 | 0.167 | -0.000 |
| original_solver/gsm8k13_report | base | tie_break | 0.601 | 0.601 | -0.000 |
| original_solver/gsm8k13_report | base | consistent_accuracy | 0.71 | 0.710 | +0.000 |
| original_solver/gsm8k13_report | base | coverage | 0.479 | 0.479 | +0.000 |
| original_solver/gsm8k13_report | base | all_pairs_accuracy | 0.34 | 0.340 | +0.000 |
| original_solver/gsm8k13_report | base_c | single_accuracy | 0.549 | 0.550 | +0.001 |
| original_solver/gsm8k13_report | base_c | position_2_rate | 0.37 | 0.361 | -0.009 |
| original_solver/gsm8k13_report | base_c | invented_rate | 0.049 | 0.043 | -0.006 |
| original_solver/gsm8k13_report | base_c | tie_break | 0.568 | 0.568 | -0.000 |
| original_solver/gsm8k13_report | base_c | consistent_accuracy | 0.638 | 0.636 | -0.002 |
| original_solver/gsm8k13_report | base_c | coverage | 0.49 | 0.497 | +0.007 |
| original_solver/svamp12_report | base | skill_index | 0.248 | 0.248 | -0.000 |
| original_solver/svamp12_report | base | position_2_rate | 0.543 | 0.543 | +0.000 |
| original_solver/svamp12_report | base | invented_rate | 0.132 | 0.132 | +0.000 |
| original_solver/svamp12_report | base | tie_break | 0.609 | 0.609 | -0.000 |
| original_solver/svamp12_report | base | consistent_accuracy | 0.687 | 0.687 | -0.000 |
| original_solver/svamp12_report | base | coverage | 0.583 | 0.583 | +0.000 |
| original_solver/svamp12_report | base | all_pairs_accuracy | 0.401 | 0.401 | -0.000 |
