# Router Recovery V2 dataset status

As of Phase 3 closure on 2026-09-13, the canonical partial pool contains **108 verified pairs**
and **216 balanced behavior rows**. It is **92 pairs below** the 200-pair
minimum, so no final train/calibration/internal-test split was created and no
training was started.

## Composition

| Dimension | Count |
|---|---:|
| KEEP rows | 108 |
| REVISE rows | 108 |
| Math pairs | 84 |
| Code pairs | 24 |
| GSM8K pairs | 84 |
| APPS pairs | 23 |
| MBPP pairs | 1 |
| Self_Correction_v1-origin pairs | 78 |
| Base-origin pairs | 30 |

## Validation

- Correct answers freshly passed: 108/108.
- Wrong answers freshly failed: 108/108.
- Same-model-origin pairs: 108/108.
- Synthetic wrong answers: 0.
- Duplicate source IDs or pair IDs: 0.
- Protected-source overlap: 0 against the 736-source union of probe-256,
  frozen-200, Decision-Only, and Router Calibration V1.
- Each source has exactly one KEEP and one REVISE row.
- KEEP and REVISE use the identical `canonical_review_v1` distribution.

## GPU mining outcome

The original CPU pool contributed 65 pairs. Sequential GPU mining admitted 43
new pairs from 2,804 valid-prompt candidates. A separate 480-candidate batch
used an incorrect raw-problem prompt and remains explicitly excluded. The last
adaptive batch yielded only 2 new pairs from 184 candidates, so repeating the
same strategy is not cost-effective.

No next step is authorized inside Phase 3. A future phase may import these
immutable pairs, but must define new holdouts and a new protocol before adding
the missing 92 source-disjoint pairs.

A full source scan is now ready: 4,087 new APPS-train sources and 339 new
non-test MBPP sources. They are structurally eligible candidates, not verified
pairs. They are divided into 32-source GPU batches. Running all 35,408 queued
generations is explicitly prohibited; run one APPS and one MBPP batch, verify
yield, and repeat only while useful or until the 200-pair gate is reached.

GPU generation logs and environment provenance are backed up under
`outputs/phase3_router_recovery_v2/gpu_generation_20260908/`.
