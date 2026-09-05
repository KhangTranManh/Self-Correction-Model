# Phase 3 initial-attempt report

Run date: 2026-09-03 (Asia/Saigon)

## Scope

- Hardware: NVIDIA GeForce RTX 3090 Ti, 24,564 MiB VRAM
- Runtime: vLLM 0.28.0, PyTorch 2.13.0+cu130, BF16
- Base: `Qwen/Qwen2.5-7B-Instruct`
- Tuned: `Kxck/Self_Correction_v1`
- Important: `Self_Correction_v1` is a standalone merged BF16 checkpoint on
  Hugging Face, not a LoRA adapter repository. It was therefore served as its
  own checkpoint and compared with the original base model.
- Identical generation settings: `temperature=0.7`, `top_p=0.9`,
  `max_tokens=1024`, seed `314159` plus the source-row index.
- Prompt: the Phase 1 initial-solve prompt, unchanged between models.
- Correctness: deterministic SymPy answer checking for GSM8K and real MBPP unit
  tests in isolated subprocesses.

## Source pool

| Dataset | Split | Rows |
|---|---|---:|
| GSM8K | `train` | 1,000 |
| MBPP | `train` + `validation` + `prompt` | 474 |
| **Total** | | **1,474** |

The frozen P0 sources (`GSM8K/test` and `MBPP/full/test`) were excluded. The
source manifest SHA-256 is
`0d6d04e28db674728636b4d3ce8e01a0a09cd1077567a930db26865c6c4ae8ea`.

## Results

| Model | Dataset | Correct | Wrong | Total | Accuracy |
|---|---|---:|---:|---:|---:|
| Base | GSM8K | 734 | 266 | 1,000 | 73.40% |
| Base | MBPP | 35 | 439 | 474 | 7.38% |
| Base | **Overall** | **769** | **705** | **1,474** | **52.17%** |
| Self_Correction_v1 | GSM8K | 873 | 127 | 1,000 | 87.30% |
| Self_Correction_v1 | MBPP | 29 | 445 | 474 | 6.12% |
| Self_Correction_v1 | **Overall** | **902** | **572** | **1,474** | **61.19%** |

Compared with base, `Self_Correction_v1` gains 9.02 percentage points overall.
That aggregate gain is entirely mathematical: GSM8K gains 13.90 points, while
MBPP loses 1.27 points (six fewer correct programs).

## Paired outcome transitions

Every model saw the same source row with the same per-row seed.

| Dataset | Both correct | Base only correct | Tuned only correct | Both wrong |
|---|---:|---:|---:|---:|
| GSM8K | 692 | 42 | 181 | 85 |
| MBPP | 24 | 11 | 5 | 434 |
| **Overall** | **716** | **53** | **186** | **519** |

The tuned checkpoint repairs 186 cases that base misses but regresses on 53
cases that base solves, a net gain of 133. On MBPP specifically, only five base
failures become correct while eleven base successes become wrong.

## Format observation

- Both models produced an extractable answer for all 1,474 rows.
- Base followed the requested fenced-code format on 474/474 MBPP rows.
- `Self_Correction_v1` used a code fence on 116/474 MBPP rows (24.47%). The
  remaining outputs were mostly bare Python code, so this is instruction-format
  loss rather than proof of correction-report scaffolding.
- No literal `<thinking>` or `### Sửa lại` correction-report markers appeared in
  either model's initial attempts.
- HumanEval was not run here; this result does not replace the frozen P0
  HumanEval measurement.

## Artifact validation

- Both raw files contain exactly 1,474 unique IDs and the same source IDs.
- Problem text, dataset identity, and ground truth match pairwise across models.
- Every row has exactly the requested nine fields.
- Correct and wrong pool files are disjoint, their union equals the raw file,
  and every pool row matches its raw source row.
- Base raw SHA-256:
  `004f304734af42090fde5f164643faa1f7100d05fe22996263020667df22a2ac0`
- Self_Correction_v1 raw SHA-256:
  `7a2d7e0622c2f39d0951f32e5a6f5ef991b723019efe242e9068dff060a5da6c3`

## Initial Phase 3 data implication

The tuned policy yields 902 correct and 572 wrong attempts, which is sufficient
in aggregate to construct KEEP and REVISE variants. The domain mix is highly
skewed, however: the correct pool is dominated by GSM8K (873 math, 29 code),
while the wrong pool is dominated by MBPP (127 math, 445 code). Any later
counterfactual SFT builder should therefore balance by both behavior condition
and domain rather than sampling only from the aggregate pools.
