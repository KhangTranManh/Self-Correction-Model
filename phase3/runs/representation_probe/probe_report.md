# Phase 3 Representation Probe

## Conclusion

`policy_shift_without_representation_gain`

The language-model weights were frozen. Only regularized logistic-regression
probes were fitted; layer and regularization selection used probe-dev only.

## Data

- Sources: 256 (maximum clean balanced size under current constraints).
- Train/dev/test: 152/52/52.
- Every split is balanced KEEP/REVISE and math/code; no CW-as-wrong, HumanEval, frozen leakage, or source overlap.

## Layer-wise test curve

| Layer | V1 balanced accuracy | V3 balanced accuracy |
|---:|---:|---:|
| 7 | 61.5% | 59.6% |
| 14 | 61.5% | 61.5% |
| 21 | 65.4% | 67.3% |
| 28 | 67.3% | 69.2% |

## Dev-selected probes

| Metric | Decision-Only V1 | Router V3 |
|---|---:|---:|
| Selected layer | 28 | 28 |
| Test accuracy | 67.3% | 69.2% |
| Test balanced accuracy | 67.3% | 69.2% |
| KEEP recall | 80.8% | 76.9% |
| REVISE recall | 53.8% | 61.5% |
| ROC-AUC | 0.734 | 0.714 |

## Controls

- Random labels (20 repeats), V1: 50.1% ± 6.1 pp.
- Random labels (20 repeats), V3: 50.8% ± 5.8 pp.
- Answer length only: 57.7% balanced accuracy.
- Dataset/domain only: 50.0% balanced accuracy.

## Selected-layer subgroup results

| Group | V1 balanced accuracy | V3 balanced accuracy |
|---|---:|---:|
| code | 57.7% | 57.7% |
| math | 76.9% | 80.8% |
| apps | 68.8% | 68.8% |
| gsm8k | 76.9% | 80.8% |
| mbpp | 40.0% | 40.0% |

## Interpretation

Router generation changed sharply (KEEP recall 81.0% -> 62.0%),
while the dev-selected representation probes differ by +1.9 pp on test.
That difference is one example in the 52-row test split, while ROC-AUC falls
from 0.734 to 0.714.
Both probes show partial rather than strong linear separation: math is useful,
but code remains weak and MBPP is below chance in this small test cell.
The answer-length control reaching above chance also means part of the signal
may be superficial. Overall, V3 primarily changed the generated decision policy
rather than demonstrating a material improvement in correctness representation.
