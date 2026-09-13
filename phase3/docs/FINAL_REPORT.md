# Phase 3 final report

**Status:** closed on 2026-09-13.

## Research question

Can a 7B self-correction model reliably distinguish a correct previous answer
that should be kept from a plausible wrong answer that should be revised?

## Final answer

Partially, but not reliably enough for deployment. Decision-Only V1 learned a
repeatable correctness-ranking signal, yet native REVISE recall remained 49%
on the frozen 200-row benchmark. Every attempted follow-up either failed to
improve wrong-answer detection or shifted the policy toward KEEP.

## Final model decision

- `Kxck/Self_Correction_v1`: retained as canonical solver/repair checkpoint.
- Decision-Only V1: retained as the final Phase 3 router baseline.
- Router V3, both DPO variants, Decision-Token V2, Same-Origin Router V1, and
  verifier-reward GRPO: not promoted.
- Frozen classifier and calibration: diagnostic only; thresholds did not
  transfer reliably.
- Activation Steering V2: rejected as a causal intervention; it changed bias,
  not semantic error detection.

## Key evidence

| Method | Balanced accuracy | KEEP recall | REVISE recall | Decision |
|---|---:|---:|---:|---|
| Decision-Only V1 | 65.0% | 81.0% | 49.0% | final baseline |
| DPO Pilot V1 | 66.5% | 85.0% | 48.0% | rejected |
| DPO Semantic V2 | 65.5% | 85.0% | 46.0% | rejected |
| Decision-Token V2 | 64.0% | 94.0% | 34.0% | rejected |
| Same-Origin Router V1 | 61.0% | 98.0% | 24.0% | rejected |
| Verifier-reward pilot | 62.0% | 98.0% | 26.0% | rejected |

The frozen layer-28 representation retained ROC-AUC 71.7% on the external
frozen benchmark, but the precommitted threshold collapsed. Rank balancing was
diagnostic, not deployable calibration. Probe/CAA activation steering at layers
14, 21, and 28 produced no valid behavioral improvement.

## Data disposition

- Canonical Decision-Only corpus: 240 rows, split 168/36/36.
- Protected frozen behavior benchmark: 200 rows, balanced by domain and state.
- Representation probe: 256 rows with source-disjoint train/dev/test.
- Router Recovery V2: 108 verified same-origin pairs / 216 behavior rows,
  preserved but closed below its 200-pair minimum; no training was run.
- Unexecuted full APPS/MBPP generation queue: compacted because it was
  reproducible planning data, not observed evidence.

See `../data/catalog.json` for paths, row counts, hashes, and lifecycle status.

## Scientific conclusion

The model contains partially decodable correctness information, especially for
math, but the signal is weaker for executable code and is not converted into a
stable KEEP/REVISE policy. Weight updates repeatedly increased KEEP bias;
simple calibration and residual steering did not solve the boundary problem.

Any continuation must be a new phase with a new preregistered hypothesis. It
should not silently extend Phase 3 or tune against the frozen Phase 3 benchmark.
