# Phase 15 recovered results

This directory preserves the recovered P15-A2 and P15-A3 development
artifacts. The original recovery copies remain under
`C:\Users\khang\Documents\Phase15-recovered-2026-11-03`.

- `p15_a2_kaggle_audit.jsonl` and `p15_a2_kaggle_summary.json`: complete
  40-episode P15-A2 Kaggle run.
- `p15_a2_colab_partial_audit.jsonl`: separate partial P15-A2 Colab run;
  never pool it with the Kaggle run.
- `P15_A2_RESCORE.json`: independently reconstructed P15-A2 scores. Its arm
  comparison is decoding-seed-confounded.
- `p15_a3_audit.jsonl` and `p15_a3_summary.json`: complete 40-episode P15-A3
  matched-seed Colab diagnostic.
- `P15_A3_RESCORE.json`: independent P15-A3 integrity and paired scores.

No protected evaluation split was opened in either run.
