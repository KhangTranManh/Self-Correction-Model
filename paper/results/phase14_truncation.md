# Phase 14 — output bị cắt ở trần token đang làm gì (4 mô hình)

**Khám phá, mô tả, gán nhãn tự động.** Mỗi output chạm trần (lời giải s1–s5: góc nhìn 768 token; lời chấm: trần 768) nhận một nhãn, xét theo thứ tự c → b → d → a:

- **(c) lạc đề / rác / lẫn ngôn ngữ:** chuỗi ≥ 30 ký tự giống nhau, ≥ 2% ký tự chữ khác Latin/tiếng Việt (Hán, Kirin…), hoặc đổi ngôn ngữ giữa 1/3 đầu và 1/3 cuối (Việt ↔ Anh). Viết hoàn toàn bằng tiếng Anh **không** tính là (c); đếm riêng ở cột cuối.
- **(b) lặp vòng:** một chuỗi 10 từ hoặc một câu ≥ 25 ký tự lặp nguyên văn ≥ 3 lần, hoặc liệt kê theo khuôn (thay mọi số bằng 0 thì một dòng lặp ≥ 5 lần).
- **(d) bỏ hai lời giải, tự giải lại lan man** (chỉ cho lời chấm): 60% cuối bài không còn nhắc tới lời giải A/B nào và đang viết lời giải riêng từng bước.
- **(a) dài nhưng đúng hướng, chưa xong:** còn lại. Trong ngoặc: phần của (a) mà 400 ký tự cuối đã có câu kết luận (bị cắt đúng lúc kết luận).

Ngưỡng cố định trước khi đọc số đếm; ngưỡng ≥ 5 cho liệt kê và nhãn (d) đặt ra sau khi kiểm tra tay 12 ca của một lần chạy trước (khoảng một nửa "lặp" khi đó thực ra là chép lại từng bước hai lời giải). Bản 2, sau khi kiểm tra tay ví dụ của bản 1: nhận diện ngôn ngữ bằng từ chức năng và bỏ qua công thức LaTeX (trước đó đoạn toàn công thức bị coi là tiếng Anh nên vòng lặp phương trình rơi vào (c)); nhận cả cách gọi "Phương pháp 1/2", "Method 1/2" cho hai lời giải (trước đó bị gán nhầm (d)).

| Split | Mô hình | Dataset | Loại output | Số output | Chạm trần | (a) đúng hướng, chưa xong (đang kết luận) | (b) lặp vòng | (c) lạc đề / rác / lẫn ngôn ngữ | (d) tự giải lại lan man | Toàn tiếng Anh |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| pilot | Llama-3.1-8B | ASDiv | lời chấm base | 118 | 17 (14.4%) | 5.9% (0.0%) | 88.2% | 5.9% | 0.0% | 0.0% |
| pilot | Llama-3.1-8B | ASDiv | lời chấm base_c | 118 | 11 (9.3%) | 27.3% (27.3%) | 63.6% | 0.0% | 9.1% | 0.0% |
| pilot | Llama-3.1-8B | ASDiv | lời giải s1–s5 | 500 | 8 (1.6%) | 12.5% (0.0%) | 87.5% | 0.0% | — | 0.0% |
| pilot | Llama-3.1-8B | MATH 1–3 | lời chấm base | 244 | 88 (36.1%) | 34.1% (14.8%) | 55.7% | 0.0% | 10.2% | 0.0% |
| pilot | Llama-3.1-8B | MATH 1–3 | lời chấm base_c | 244 | 88 (36.1%) | 29.5% (10.2%) | 62.5% | 0.0% | 8.0% | 0.0% |
| pilot | Llama-3.1-8B | MATH 1–3 | lời giải s1–s5 | 500 | 79 (15.8%) | 34.2% (7.6%) | 65.8% | 0.0% | — | 0.0% |
| pilot | Mistral-7B | ASDiv | lời chấm base | 228 | 11 (4.8%) | 27.3% (0.0%) | 27.3% | 9.1% | 36.4% | 90.9% |
| pilot | Mistral-7B | ASDiv | lời chấm base_c | 228 | 0 (0.0%) | 0.0% (0.0%) | 0.0% | 0.0% | 0.0% | 0.0% |
| pilot | Mistral-7B | ASDiv | lời giải s1–s5 | 500 | 11 (2.2%) | 54.5% (9.1%) | 45.5% | 0.0% | — | 90.9% |
| pilot | Mistral-7B | MATH 1–3 | lời chấm base | 350 | 62 (17.7%) | 25.8% (9.7%) | 33.9% | 1.6% | 38.7% | 96.8% |
| pilot | Mistral-7B | MATH 1–3 | lời chấm base_c | 350 | 4 (1.1%) | 25.0% (25.0%) | 50.0% | 25.0% | 0.0% | 75.0% |
| pilot | Mistral-7B | MATH 1–3 | lời giải s1–s5 | 500 | 85 (17.0%) | 77.6% (11.8%) | 22.4% | 0.0% | — | 87.1% |
| pilot | Self_Correction | ASDiv | lời chấm base | 98 | 7 (7.1%) | 0.0% (0.0%) | 100.0% | 0.0% | 0.0% | 0.0% |
| pilot | Self_Correction | ASDiv | lời chấm base_c | 98 | 14 (14.3%) | 0.0% (0.0%) | 92.9% | 7.1% | 0.0% | 0.0% |
| pilot | Self_Correction | ASDiv | lời giải s1–s5 | 500 | 8 (1.6%) | 0.0% (0.0%) | 62.5% | 37.5% | — | 25.0% |
| pilot | Self_Correction | MATH 1–3 | lời chấm base | 238 | 35 (14.7%) | 0.0% (0.0%) | 77.1% | 5.7% | 17.1% | 5.7% |
| pilot | Self_Correction | MATH 1–3 | lời chấm base_c | 238 | 34 (14.3%) | 0.0% (0.0%) | 73.5% | 8.8% | 17.6% | 5.9% |
| pilot | Self_Correction | MATH 1–3 | lời giải s1–s5 | 500 | 47 (9.4%) | 19.1% (6.4%) | 76.6% | 4.3% | — | 10.6% |
| pilot | Qwen2.5-7B-Instruct | ASDiv | lời chấm base | 72 | 13 (18.1%) | 53.8% (30.8%) | 30.8% | 0.0% | 15.4% | 7.7% |
| pilot | Qwen2.5-7B-Instruct | ASDiv | lời chấm base_c | 72 | 3 (4.2%) | 66.7% (33.3%) | 0.0% | 0.0% | 33.3% | 0.0% |
| pilot | Qwen2.5-7B-Instruct | ASDiv | lời giải s1–s5 | 500 | 6 (1.2%) | 66.7% (16.7%) | 33.3% | 0.0% | — | 0.0% |
| pilot | Qwen2.5-7B-Instruct | MATH 1–3 | lời chấm base | 212 | 62 (29.2%) | 40.3% (11.3%) | 35.5% | 3.2% | 21.0% | 1.6% |
| pilot | Qwen2.5-7B-Instruct | MATH 1–3 | lời chấm base_c | 212 | 21 (9.9%) | 33.3% (4.8%) | 38.1% | 4.8% | 23.8% | 14.3% |
| pilot | Qwen2.5-7B-Instruct | MATH 1–3 | lời giải s1–s5 | 500 | 68 (13.6%) | 61.8% (8.8%) | 36.8% | 1.5% | — | 0.0% |
| main | Llama-3.1-8B | ASDiv | lời chấm base | 860 | 156 (18.1%) | 20.5% (4.5%) | 70.5% | 0.0% | 9.0% | 0.0% |
| main | Llama-3.1-8B | ASDiv | base_1536 | 860 | 110 (12.8%) | 15.5% (2.7%) | 76.4% | 0.0% | 8.2% | 0.0% |
| main | Llama-3.1-8B | ASDiv | lời chấm base_c | 860 | 122 (14.2%) | 18.9% (1.6%) | 73.0% | 0.0% | 8.2% | 0.0% |
| main | Llama-3.1-8B | ASDiv | base_c_1536 | 860 | 73 (8.5%) | 12.3% (0.0%) | 78.1% | 0.0% | 9.6% | 0.0% |
| main | Llama-3.1-8B | ASDiv | lời giải s1–s5 | 4110 | 67 (1.6%) | 26.9% (3.0%) | 73.1% | 0.0% | — | 0.0% |
| main | Llama-3.1-8B | bridge | lời chấm base | 972 | 109 (11.2%) | 10.1% (3.7%) | 81.7% | 1.8% | 6.4% | 0.0% |
| main | Llama-3.1-8B | bridge | lời chấm base_c | 972 | 140 (14.4%) | 16.4% (2.1%) | 80.0% | 1.4% | 2.1% | 0.0% |
| main | Llama-3.1-8B | GSM8K | lời chấm base | 804 | 118 (14.7%) | 21.2% (11.9%) | 75.4% | 0.0% | 3.4% | 0.0% |
| main | Llama-3.1-8B | GSM8K | lời chấm base_c | 804 | 111 (13.8%) | 36.0% (9.9%) | 58.6% | 0.0% | 5.4% | 0.0% |
| main | Llama-3.1-8B | GSM8K | lời giải s1–s5 | 2805 | 41 (1.5%) | 26.8% (4.9%) | 73.2% | 0.0% | — | 0.0% |
| main | Llama-3.1-8B | MATH 1–3 | lời chấm base | 2042 | 651 (31.9%) | 24.9% (6.3%) | 65.0% | 0.3% | 9.8% | 0.0% |
| main | Llama-3.1-8B | MATH 1–3 | base_1536 | 2042 | 322 (15.8%) | 15.2% (3.1%) | 73.9% | 0.6% | 10.2% | 0.0% |
| main | Llama-3.1-8B | MATH 1–3 | lời chấm base_c | 2042 | 745 (36.5%) | 24.2% (5.6%) | 63.4% | 0.0% | 12.5% | 0.0% |
| main | Llama-3.1-8B | MATH 1–3 | base_c_1536 | 2042 | 300 (14.7%) | 15.7% (2.3%) | 72.0% | 0.0% | 12.3% | 0.0% |
| main | Llama-3.1-8B | MATH 1–3 | lời giải s1–s5 | 3925 | 618 (15.7%) | 34.6% (6.3%) | 64.9% | 0.5% | — | 0.0% |
| main | Llama-3.1-8B | SVAMP | lời chấm base | 1172 | 210 (17.9%) | 6.2% (1.9%) | 91.0% | 0.0% | 2.9% | 0.0% |
| main | Llama-3.1-8B | SVAMP | lời chấm base_c | 1172 | 150 (12.8%) | 10.0% (2.7%) | 84.0% | 0.0% | 6.0% | 0.0% |
| main | Llama-3.1-8B | SVAMP | lời giải s1–s5 | 4800 | 115 (2.4%) | 15.7% (2.6%) | 84.3% | 0.0% | — | 0.0% |
| main | Mistral-7B | ASDiv | lời chấm base | 1866 | 65 (3.5%) | 21.5% (6.2%) | 43.1% | 6.2% | 29.2% | 86.2% |
| main | Mistral-7B | ASDiv | lời chấm base_c | 1866 | 6 (0.3%) | 16.7% (0.0%) | 66.7% | 0.0% | 16.7% | 100.0% |
| main | Mistral-7B | ASDiv | lời giải s1–s5 | 4110 | 73 (1.8%) | 76.7% (17.8%) | 23.3% | 0.0% | — | 75.3% |
| main | Mistral-7B | bridge | lời chấm base | 972 | 116 (11.9%) | 19.8% (9.5%) | 30.2% | 17.2% | 32.8% | 8.6% |
| main | Mistral-7B | bridge | lời chấm base_c | 972 | 13 (1.3%) | 7.7% (7.7%) | 46.2% | 0.0% | 46.2% | 15.4% |
| main | Mistral-7B | GSM8K | lời chấm base | 1560 | 57 (3.7%) | 15.8% (1.8%) | 31.6% | 1.8% | 50.9% | 77.2% |
| main | Mistral-7B | GSM8K | lời chấm base_c | 1560 | 1 (0.1%) | 0.0% (0.0%) | 0.0% | 0.0% | 100.0% | 100.0% |
| main | Mistral-7B | GSM8K | lời giải s1–s5 | 2805 | 59 (2.1%) | 72.9% (16.9%) | 25.4% | 1.7% | — | 78.0% |
| main | Mistral-7B | MATH 1–3 | lời chấm base | 2754 | 465 (16.9%) | 25.6% (5.2%) | 33.5% | 2.8% | 38.1% | 85.6% |
| main | Mistral-7B | MATH 1–3 | lời chấm base_c | 2754 | 44 (1.6%) | 22.7% (0.0%) | 47.7% | 0.0% | 29.5% | 90.9% |
| main | Mistral-7B | MATH 1–3 | lời giải s1–s5 | 3925 | 619 (15.8%) | 67.4% (9.9%) | 32.0% | 0.6% | — | 84.3% |
| main | Mistral-7B | SVAMP | lời chấm base | 2318 | 69 (3.0%) | 10.1% (1.4%) | 58.0% | 4.3% | 27.5% | 75.4% |
| main | Mistral-7B | SVAMP | lời chấm base_c | 2318 | 2 (0.1%) | 0.0% (0.0%) | 100.0% | 0.0% | 0.0% | 50.0% |
| main | Mistral-7B | SVAMP | lời giải s1–s5 | 4800 | 65 (1.4%) | 63.1% (15.4%) | 35.4% | 1.5% | — | 66.2% |
| main | Self_Correction | ASDiv | lời chấm base | 892 | 66 (7.4%) | 0.0% (0.0%) | 80.3% | 12.1% | 7.6% | 6.1% |
| main | Self_Correction | ASDiv | lời chấm base_c | 892 | 53 (5.9%) | 1.9% (0.0%) | 86.8% | 7.5% | 3.8% | 7.5% |
| main | Self_Correction | ASDiv | lời giải s1–s5 | 4110 | 62 (1.5%) | 6.5% (0.0%) | 83.9% | 9.7% | — | 4.8% |
| main | Self_Correction | GSM8K | lời chấm base | 986 | 38 (3.9%) | 0.0% (0.0%) | 86.8% | 2.6% | 10.5% | 0.0% |
| main | Self_Correction | GSM8K | lời chấm base_c | 986 | 33 (3.3%) | 0.0% (0.0%) | 90.9% | 0.0% | 9.1% | 0.0% |
| main | Self_Correction | GSM8K | lời giải s1–s5 | 2805 | 49 (1.7%) | 10.2% (0.0%) | 87.8% | 2.0% | — | 4.1% |
| main | Self_Correction | GSM8K (báo cáo) | lời chấm base | 972 | 0 (0.0%) | 0.0% (0.0%) | 0.0% | 0.0% | 0.0% | 0.0% |
| main | Self_Correction | GSM8K (báo cáo) | lời chấm base_c | 972 | 36 (3.7%) | 2.8% (0.0%) | 83.3% | 8.3% | 5.6% | 0.0% |
| main | Self_Correction | MATH 1–3 | lời chấm base | 1940 | 291 (15.0%) | 2.4% (0.3%) | 82.1% | 2.7% | 12.7% | 1.4% |
| main | Self_Correction | MATH 1–3 | lời chấm base_c | 1940 | 279 (14.4%) | 2.2% (0.0%) | 83.2% | 2.9% | 11.8% | 2.9% |
| main | Self_Correction | MATH 1–3 | lời giải s1–s5 | 3925 | 355 (9.0%) | 14.9% (4.8%) | 81.1% | 3.9% | — | 6.8% |
| main | Self_Correction | SVAMP | lời chấm base | 1000 | 32 (3.2%) | 0.0% (0.0%) | 96.9% | 0.0% | 3.1% | 0.0% |
| main | Self_Correction | SVAMP | lời chấm base_c | 1000 | 25 (2.5%) | 0.0% (0.0%) | 92.0% | 4.0% | 4.0% | 0.0% |
| main | Self_Correction | SVAMP | lời giải s1–s5 | 4800 | 60 (1.2%) | 0.0% (0.0%) | 98.3% | 1.7% | — | 6.7% |
| main | Self_Correction | SVAMP (báo cáo) | lời chấm base | 1002 | 0 (0.0%) | 0.0% (0.0%) | 0.0% | 0.0% | 0.0% | 0.0% |
| main | Self_Correction | SVAMP (báo cáo) | lời chấm base_c | 1002 | 32 (3.2%) | 0.0% (0.0%) | 96.9% | 3.1% | 0.0% | 0.0% |
| main | Qwen2.5-7B-Instruct | ASDiv | lời chấm base | 722 | 69 (9.6%) | 66.7% (26.1%) | 23.2% | 1.4% | 8.7% | 8.7% |
| main | Qwen2.5-7B-Instruct | ASDiv | lời chấm base_c | 722 | 13 (1.8%) | 61.5% (30.8%) | 15.4% | 0.0% | 23.1% | 7.7% |
| main | Qwen2.5-7B-Instruct | ASDiv | lời giải s1–s5 | 4110 | 23 (0.6%) | 60.9% (13.0%) | 39.1% | 0.0% | — | 0.0% |
| main | Qwen2.5-7B-Instruct | bridge | lời chấm base | 972 | 195 (20.1%) | 37.4% (10.8%) | 53.8% | 2.1% | 6.7% | 0.0% |
| main | Qwen2.5-7B-Instruct | bridge | lời chấm base_c | 972 | 48 (4.9%) | 43.8% (20.8%) | 50.0% | 2.1% | 4.2% | 2.1% |
| main | Qwen2.5-7B-Instruct | GSM8K | lời chấm base | 676 | 156 (23.1%) | 53.2% (22.4%) | 41.7% | 1.9% | 3.2% | 0.6% |
| main | Qwen2.5-7B-Instruct | GSM8K | lời chấm base_c | 676 | 39 (5.8%) | 59.0% (25.6%) | 35.9% | 0.0% | 5.1% | 0.0% |
| main | Qwen2.5-7B-Instruct | GSM8K | lời giải s1–s5 | 2805 | 4 (0.1%) | 50.0% (0.0%) | 25.0% | 25.0% | — | 0.0% |
| main | Qwen2.5-7B-Instruct | MATH 1–3 | lời chấm base | 1730 | 468 (27.1%) | 38.7% (11.1%) | 40.6% | 1.7% | 19.0% | 0.9% |
| main | Qwen2.5-7B-Instruct | MATH 1–3 | lời chấm base_c | 1730 | 124 (7.2%) | 37.1% (16.1%) | 37.1% | 3.2% | 22.6% | 9.7% |
| main | Qwen2.5-7B-Instruct | MATH 1–3 | lời giải s1–s5 | 3925 | 397 (10.1%) | 61.5% (14.9%) | 35.0% | 3.5% | — | 0.5% |
| main | Qwen2.5-7B-Instruct | SVAMP | lời chấm base | 910 | 49 (5.4%) | 24.5% (6.1%) | 65.3% | 0.0% | 10.2% | 0.0% |
| main | Qwen2.5-7B-Instruct | SVAMP | lời chấm base_c | 910 | 8 (0.9%) | 37.5% (37.5%) | 62.5% | 0.0% | 0.0% | 0.0% |
| main | Qwen2.5-7B-Instruct | SVAMP | lời giải s1–s5 | 4800 | 12 (0.2%) | 16.7% (8.3%) | 75.0% | 8.3% | — | 0.0% |

Tỷ lệ các nhãn tính trên số output chạm trần của dòng đó. Ví dụ từng nhãn (300 ký tự cuối) trong file .json để kiểm tra tay.
