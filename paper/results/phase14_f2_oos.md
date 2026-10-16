# F2 ngoài mẫu: vote có trọng số theo probe (s1–s3) so với vote@3 (khám phá)

Quy tắc ghi trước: `phase14/RUN_STATUS.md` (2026-10-15, iv). Số đứng trước là ước lượng điểm, [a; b] là KTC 95% bootstrap ghép cặp theo bài.

| Mô hình | Train → test | Layer | AUC CV (train) | AUC test | Vote có trọng số | vote@3 | Hiệu [KTC] |
|---|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | asdiv->math13 | 13 | 0.895 | 0.801 | 54.3% | 50.8% | +3.4 [+0.9; +5.9] |
| Llama-3.1-8B | math13->asdiv | 32 | 0.871 | 0.811 | 85.4% | 83.7% | +1.7 [+0.2; +3.3] |
| Llama-3.1-8B | svamp12->gsm8k13 | 14 | 0.918 | 0.784 | 80.6% | 78.3% | +2.3 [+0.0; +4.6] |
| Mistral-7B | asdiv->math13 | 19 | 0.876 | 0.773 | 27.0% | 23.8% | +3.2 [+1.0; +5.4] |
| Mistral-7B | math13->asdiv | 21 | 0.829 | 0.813 | 72.9% | 68.5% | +4.4 [+2.1; +6.7] |
| Mistral-7B | svamp12->gsm8k13 | 20 | 0.865 | 0.775 | 59.7% | 55.6% | +4.1 [+1.2; +7.3] |
| Self_Correction | asdiv->math13 | 20 | 0.883 | 0.791 | 57.1% | 56.4% | +0.6 [-1.9; +3.1] |
| Self_Correction | math13->asdiv | 19 | 0.837 | 0.833 | 87.0% | 85.6% | +1.3 [+0.0; +2.7] |
| Self_Correction | svamp12->gsm8k13 | 19 | 0.891 | 0.778 | 74.5% | 71.1% | +3.4 [+1.1; +5.9] |
| Qwen2.5-7B-Instruct | asdiv->math13 | 24 | 0.922 | 0.875 | 65.5% | 66.6% | -1.1 [-3.1; +0.6] |
| Qwen2.5-7B-Instruct | math13->asdiv | 19 | 0.928 | 0.903 | 91.6% | 91.0% | +0.6 [-0.4; +1.6] |
| Qwen2.5-7B-Instruct | svamp12->gsm8k13 | 17 | 0.940 | 0.835 | 86.3% | 84.7% | +1.6 [-0.2; +3.4] |

Xác nhận ngoài mẫu (KTC > 0 ở ≥ 2/3 cặp): Llama-3.1-8B có, Mistral-7B có, Self_Correction không, Qwen2.5-7B-Instruct không
