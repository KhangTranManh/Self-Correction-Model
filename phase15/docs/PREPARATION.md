# Phase 15 local preparation

**Status: recorded 2026-10-27. This is not a pilot, not evidence, and not a
frozen protocol.** No split was drawn and no protected text was written. The
seven open decisions in `PREREGISTRATION.md` stay open.

Counts are in `../data/preparation_v1/`. Recompute them with:

```powershell
python phase15/scripts/test_local_preparation.py
python phase15/scripts/audit_sources.py
```

The second command reads public MATH train and MultiArith files into a
temporary directory. It does not keep those files.

## Self-edit gate

`scripts/self_edit_gate.py` reviews one lesson before a LoRA update.

- A line-leading marker (`1.`, `2)`, `Step 3:`) is removed before numbers are
  read. A marker is not a copied teaching value.
- A trailing `Verdict: Solution A` or `Verdict: Solution B` is removed when the
  other side is not named. The lesson may name both sides as the format.
- After that, any remaining number equal to a teaching, probe, or control
  number is rejected. An 8-word phrase shared with a probe or control source is
  rejected. The teaching text itself is the model's input and is not a
  protected source.
- The lesson must contain a rule and a template of at least eight words each.
- `accept` keeps the original text. `normalize` keeps the cleaned text.
  `retry` means one new sample may be reviewed. This function does not call a
  model.

On the eight saved Kaggle lessons, all eight were `normalize` and the cleaned
text then passed. Four lessons had list markers (`smoke_01` through
`smoke_04`, with 4, 4, 3, and 3 markers). All eight named the teaching
verdict side, and that line was removed. None still contained a forbidden
number or a protected phrase, so none asked for a model retry. Fixtures cover
`accept`, marker normalization against a forbidden `3`, a copied number, and a
protected 8-word phrase.

## Detectors

`scripts/detectors.py` wraps the Phase 14 judge classes without changing them.

| Type | Rule |
|---|---|
| J-format | Phase 14 class 2 |
| J-loop | Phase 14 class 3 |
| J-resolve | Phase 14 class 5 |
| J-third | Phase 14 class 6 |
| S-format | Final line is not `Đáp số: <number>`, lenient parsing is not a number, rescued parsing is |
| S-loop | The 768-token cap was hit and the Phase 14 solution labeller says looping (`b`) |
| S-integer | Proposed, not frozen. The question matches `how many`, `number of`, or `bao nhiêu`, contains none of the listed unit words, and the parsed answer is a non-integer |

Classes 1, 4, and 7 are not candidate types. Fixtures for all seven types
pass. The Phase 14 answer directory `outputs/phase14_remote/outputs/phase14_v1`
is not on this machine, so no base rate was counted. The 64 saved smoke judge
completions and 32 smoke solutions produced none of these types. That checks
that the wrapper runs on saved text. It does not estimate how often a real
model makes the errors.

## Source audit

`scripts/audit_sources.py` hashed `question`, `problem`, and `Body` plus
`Question` across 377 JSON/JSONL files under Phases 1–14 and `paper/`. That is
18,324 keys (11,305 of them outside Phase 14). The file inventory SHA-256 is
`b876a6b1e7db8f0776c13b03875c66bee2e97a5487bc95e11eccc99d1adaab50`. The raw
MATH test file is included through its `problem` field, so levels 4 and 5 are
in the key set even though they were not copied into the Phase 14 pool.

ASDiv eligibility, when Phase 14 files are excluded, reproduces the frozen
pool: 2,019 eligible, with the same rejection counts (132 near SVAMP, 146
non-numeric, 7 formula mismatches, 1 duplicate). Of that pool, 100 pilot rows
and the first 822 main-pool rows were generated. The other 1,097 rows are
still in `phase14/data/sources_v1/`. Eligible ASDiv text absent from every
scanned Phase 1–14 file: 0. ASDiv is not a fresh Phase 15 pool.

MATH train (`EleutherAI/hendrycks_math`, 7,500 rows) has 0 exact key matches.
Levels 1–3 that survive the Phase 14 constant-answer rule and the Phase 1
verifier: 2,900 (Level 1: 495, Level 2: 1,121, Level 3: 1,284). The other
4,000 training rows are level 4 or 5, and 604 level 1–3 answers are not a
single constant. Phase 1 has a loader that can append MATH train problems, and
the current `phase1/data/problems/math.jsonl` is GSM8K only. This scan cannot
see a file that was deleted. No split of the 2,900 was written.

MultiArith (`ChilleD/MultiArith`, 420 rows) was the reserved-set candidate and
was not reserved. Twelve rows repeat an earlier row. Of the 408 unique rows,
40 are in the Phase 14 ASDiv pool and none are in the MATH test pool. Of the
368 that are not exact copies, 361 have token Jaccard at least 0.5 with some
ASDiv problem, the same cap Phase 14 used against SVAMP. Seven rows are
neither an exact nor a near copy. That is not a reserved evaluation set.

## Still open

The learner, the reserved evaluation dataset, the error-type list, the
meta-train versus held-out split, the LoRA settings, the ReST-EM schedule, the
episode counts, and the GPU budget are unchanged. The next scientific step is
a pilot specification, after a clean reserved dataset is chosen. The Kaggle T4
NF4 smoke remains a different precision from the FP16 pilot named in the
preregistration.
