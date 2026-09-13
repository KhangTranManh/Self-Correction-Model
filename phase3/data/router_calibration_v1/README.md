# Router calibration v1

This dataset calibrates the frozen Decision-Only V1 hidden-state score. It is
not an LLM training dataset and must never update Decision-Only or probe weights.

- `calibration_balanced.jsonl`: 100 verified rows used to fit a scalar
  probability calibrator or threshold.
- `calibration_natural.jsonl`: 40 disjoint rows used only to audit behavior
  under naturally sampled label skew.
- `source_manifest.jsonl`: compact provenance index.
- `overlap_audit.json`: protected-source and historical-training overlap.
- `calibration_summary.json`: counts, limitations, and usage contract.

Both views are source-disjoint from probe-256 and frozen-200. Frozen-200 remains
test-only. Calibration v1 covers GSM8K/APPS in the balanced fit and adds MBPP in
the natural audit; it does not claim SVAMP or HumanEval calibration.

If existing rows fail to supply fresh GSM8K REVISE cases, `generation/` contains
a source-disjoint stochastic sampling manifest. Run it against
`Kxck/Self_Correction_v1`, fresh-verify every raw output, and retain at most one
naturally wrong answer per source. Do not manufacture wrong answers.

## Execution order

The manifest is already CPU-built. Once a GPU is available, serve
`Kxck/Self_Correction_v1` as `self-correction-v1`, then run:

```bash
python phase3/scripts/data/collect_same_origin_candidates.py \
  --manifest phase3/data/router_calibration_v1/generation/generation_manifest.jsonl \
  --output phase3/data/router_calibration_v1/generation/raw_candidates.jsonl \
  --origin self_correction_v1 --model self-correction-v1 \
  --temperature 0.9 --top-p 0.95 --max-samples-per-task 4

python phase3/scripts/data/verify_calibration_candidates.py \
  --candidates phase3/data/router_calibration_v1/generation/raw_candidates.jsonl

python phase3/scripts/data/build_router_calibration_v1.py
```

Stop if verification produces fewer than 21 unique GSM8K wrong sources. When
the dataset passes, extract layer 28 for `calibration_all.jsonl`, release the
GPU, and fit calibration on CPU with
`phase3/scripts/evaluation/fit_router_calibration.py`. The frozen-200 result is
exploratory because its outcomes have already informed this protocol.
