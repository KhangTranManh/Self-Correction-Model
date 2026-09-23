# Phase 5 CPU candidate selection — 2026-09-23

**Selection-time status:** 1,200 arithmetic candidates were ready locally;
no Phase 5 initial answer, review, or activation had been generated. A later
GPU pass collected 815 initial answers (575 correct, 240 wrong), but the
240/80/160 balanced source split, reviews, and probes are still pending. This
report records the CPU candidate selection; see `../docs/EXECUTION_PLAN.md`
for current work.

The raw input is the cached official GSM8K **train** split, exported locally as
`raw/gsm8k_train.jsonl`. It contains 7,473 questions. The official
[dataset card](https://huggingface.co/datasets/openai/gsm8k/blob/main/README.md)
describes its train split and calculation annotations; the
[source repository](https://github.com/openai/grade-school-math/blob/master/README.md)
documents the JSONL `question`/`answer` format. The raw local file has SHA-256
`9556bf9d7dba73d7fcaaeff6ca814c71684db1d8e8d485d83ec62f1c428b888d`.

The source-selection script scanned JSON and JSONL records under `phase1/`,
`phase2/`, `phase3/`, and `phase4/`, including prior train, development,
probing and unopened confirmation manifests. It excluded a GSM8K source if
its normalized question matched or its dataset index was identified in a
prior-phase record. The exclusion inventory records every input file hash.
The complete inventory hash is recorded in `candidate_report.json` and the
Phase 5 config. This checks **project source overlap**; it cannot establish
whether the original pretrained checkpoint saw GSM8K outside this project.

| CPU selection stage | Count |
|---|---:|
| Raw GSM8K train | 7,473 |
| Excluded for prior-phase source overlap | 4,000 |
| Excluded for question over 120 words | 7 |
| Excluded for fewer than 2 or more than 6 arithmetic annotations | 291 |
| Excluded for an unverifiable reference annotation | 1 |
| Eligible after CPU checks | 3,174 |
| Selected by fixed SHA-256 rank | 1,200 |

All 1,200 selected reference answers passed fresh verification with the
project MathVerifier under CPython 3.10.21. A second exclusion audit found
zero selected source overlaps. Step-count distribution is 392 / 352 / 259 /
140 / 57 for two / three / four / five / six annotations, respectively.

Evidence:

- `candidates_v1/candidate_problems.jsonl`: selected problem manifest, SHA-256
  `3dc103e168ea149f51d16e5d22fad66e97b1af9e956fba1d8914b714120e954a`.
- `candidates_v1/reference_steps.jsonl`: source reference steps stored apart
  from model-visible prompts, SHA-256
  `a3ed483127f14ed07d6d78824791cbb6f8c5e3d12983ce1095bfa1f7b94a2e64`.
- `candidates_v1/candidate_report.json`: raw and exclusion input hashes,
  selection counts and rejection reasons.
- `candidates_v1/verifier_audit.json`: 1,200 reference passes and zero overlap.
- `scripts/prepare_candidates.py`, `scripts/extract_reference_steps.py`,
  `scripts/validate_candidates.py`: local preparation and verification logic.

The next GPU stage needs only one natural original-solver answer per candidate,
in frozen rank order, until the balanced 480-source target is feasible or the
1,200-candidate budget is exhausted. Labels must be freshly verified; no
guided correction or multi-sample review is needed to establish this pool.
Error-location/type hints additionally require a reliable annotation of the
model's own erroneous step. Reference steps alone do not identify that step.
