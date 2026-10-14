# Phase 14 — độ nhạy: đọc lựa chọn của judge bằng `rescued` (khám phá)

Judge `base`, cặp đúng-một cố định theo nhãn đăng ký trước (parser chuẩn). Chỉ đổi cách đọc đáp số cuối của **judge** khi so với hai lời giải: parser chuẩn → `rescued` (nếu parser chuẩn không ra số thì lấy số đầu tiên trên dòng "Đáp số" cuối). Mỗi ô: parser chuẩn → rescued.

| Mô hình | Dataset | Một thứ tự (%) | Skill index | Bịa (%) | Tie-break |
|---|---|---|---|---|---|
| Llama-3.1-8B | ASDiv | 57,3 → 59,7 | 0,321 → 0,343 | 19,3 → 17,7 | 0,646 → 0,666 |
| Llama-3.1-8B | MATH 1–3 | 53,0 → 55,6 | 0,362 → 0,380 | 26,7 → 23,8 | 0,628 → 0,645 |
| Llama-3.1-8B | GSM8K | 50,2 → 56,6 | 0,307 → 0,371 | 29,2 → 23,3 | 0,604 → 0,657 |
| Llama-3.1-8B | SVAMP | 56,4 → 59,3 | 0,357 → 0,383 | 22,1 → 19,4 | 0,647 → 0,670 |
| Mistral-7B | ASDiv | 51,3 → 65,2 | 0,335 → 0,520 | 29,2 → 20,0 | 0,616 → 0,706 |
| Mistral-7B | MATH 1–3 | 50,5 → 55,6 | 0,344 → 0,381 | 31,4 → 26,7 | 0,609 → 0,647 |
| Mistral-7B | GSM8K | 47,4 → 58,3 | 0,255 → 0,387 | 28,7 → 21,9 | 0,575 → 0,647 |
| Mistral-7B | SVAMP | 47,4 → 62,5 | 0,254 → 0,461 | 29,2 → 20,6 | 0,584 → 0,692 |
| Self_Correction | ASDiv | 55,9 → 60,5 | 0,222 → 0,305 | 10,9 → 10,3 | 0,603 → 0,641 |
| Self_Correction | MATH 1–3 | 49,0 → 50,6 | 0,155 → 0,173 | 19,6 → 18,5 | 0,573 → 0,583 |
| Self_Correction | GSM8K | 52,2 → 53,4 | 0,191 → 0,227 | 13,7 → 15,5 | 0,590 → 0,603 |
| Self_Correction | SVAMP | 56,3 → 59,0 | 0,237 → 0,291 | 10,0 → 10,0 | 0,598 → 0,625 |
| Qwen2.5-7B-Instruct | ASDiv | 63,5 → 85,9 | 0,609 → 0,812 | 25,6 → 7,4 | 0,701 → 0,881 |
| Qwen2.5-7B-Instruct | MATH 1–3 | 45,1 → 71,6 | 0,466 → 0,682 | 49,1 → 23,2 | 0,609 → 0,787 |
| Qwen2.5-7B-Instruct | GSM8K | 56,4 → 73,9 | 0,485 → 0,631 | 31,1 → 16,2 | 0,651 → 0,788 |
| Qwen2.5-7B-Instruct | SVAMP | 63,1 → 87,8 | 0,575 → 0,827 | 22,2 → 6,0 | 0,679 → 0,894 |
