# Phase 3 APPS introductory pilot

## Decision

**Scale is recommended for APPS introductory as an additional Phase 3 source of
unique `code + correct` problems.** Self_Correction_v1 solved 141/300 (47.0%),
well above the predeclared minimum of 60/300 (20%). This is a source-suitability
decision, not evidence that V1 is a better general code solver: Base solved
147/300 (49.0%), six more problems than V1.

## Frozen candidate construction

- Dataset: `codeparrot/apps`, revision
  `21e74ddf8de1a21436da12e3e653065c5213e9d1`
- Split/difficulty: `train` / `introductory`
- Selection seed: `314159`
- Selected before either model was run: yes
- Raw train rows inspected: 5,000
- Introductory rows: 2,639
- Metadata/test-eligible rows: 2,398
- Final reference-validated candidates: 300 unique problems
- Existing Phase 3 normalized-prompt overlap: 0
- Duplicate IDs/content in the candidate pool: 0/0
- Executable tests in the frozen pool: 1,635
- Candidate SHA-256:
  `eb4a3347309e386c1314f50f01c60711a604809894ee9a909607dbb30b656f29`

The selector excluded 241 introductory rows with missing/mismatched tests. It
also encountered 32 rows for which none of the supplied reference solutions
passed the local executable verifier; these were deterministically backfilled
within the same frozen diversity stratum. No Base/V1 output or correctness was
used during selection.

### Diversity

| Dimension | Count |
|---|---:|
| Call-based tests | 225 |
| Standard-input tests | 75 |
| Codewars | 166 |
| LeetCode | 55 |
| HackerRank | 48 |
| Codeforces | 30 |
| AtCoder | 1 |

Prompt-length bins contain 96 short, 94 medium, 75 long, and 35 very-long
problems. The largest problem has 13,955 characters. Serving context was set to
16,384 tokens for both models so the fixed 2,048-token response budget fits the
longest prompt.

## Matched inference protocol

Both models used vLLM, the same ordered candidate manifest, prompt template,
per-row seeds, and generation parameters:

- `max_tokens=2048`
- `temperature=0.7`
- `top_p=0.9`
- base seed `314159`
- strict correctness: all supplied tests must pass

Models:

- Base: `Qwen/Qwen2.5-7B-Instruct`
- V1: `Kxck/Self_Correction_v1` (merged BF16 checkpoint)

## Results

| Bucket | Definition | Count |
|---|---|---:|
| CC | Base correct, V1 correct | 120 |
| WW | Base wrong, V1 wrong | 132 |
| WC | Base wrong, V1 correct | 21 |
| CW | Base correct, V1 wrong | 27 |
| **Total** | | **300** |

- Base correct: `CC + CW = 147/300 = 49.0%`
- V1 correct: `CC + WC = 141/300 = 47.0%`
- V1 net change relative to Base: `WC - CW = -6`
- V1-correct pool available for later behavior construction: 141 unique problems

By test mode:

| Mode | CC | WW | WC | CW | V1 correct |
|---|---:|---:|---:|---:|---:|
| Call-based | 88 | 96 | 19 | 22 | 107/225 (47.6%) |
| Standard input | 32 | 36 | 2 | 5 | 34/75 (45.3%) |

The predeclared gate passed: at least 60 V1-correct rows, at least 20% V1
accuracy, both test modes, at least three source hosts, and no source exceeding
70% of the pilot.

## Format observation

Base returned a fenced code block on 300/300 rows. V1 returned a fence on only
19/300 rows, but 135 of V1's 141 passing solutions were valid directly-executable
raw Python without a fence. This is a formatting regression worth retaining in
the raw artifacts; it does not invalidate executable correctness, but behavior
data construction should not treat formatting compliance as already repaired.

## Serving state

At handoff, vLLM is serving `Qwen/Qwen2.5-7B-Instruct` as model name `base` on
GPU port 8999 with `max_model_len=16384`. The RTX 3090 uses approximately
22.2/24 GB VRAM while the server is loaded.
