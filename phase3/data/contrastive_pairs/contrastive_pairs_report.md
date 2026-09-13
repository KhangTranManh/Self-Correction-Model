# Phase 3 Same-Problem Contrastive Dataset Report

> Historical dataset note (2026-09-07): this dataset was used for DPO Pilot V1.
> It remains reproducibility evidence, but `data/semantic_model_dpo/` supersedes
> it for reference-free model-vs-model pair construction. Neither resulting DPO
> adapter passed the Phase 3 promotion gate.

## Outcome

The construction pass found **120 clean same-problem pairs**, producing **240 behavioral rows**: one verified-correct `KEEP` row and one plausible verified-wrong `REVISE` row per problem. DPO Pilot V1 was subsequently trained from these rows; this sentence supersedes the original pre-training status.

## Composition

| Measure | Count |
|---|---:|
| Total pairs | 120 |
| Behavioral rows | 240 |
| Code pairs | 90 |
| Math pairs | 30 |
| APPS pairs | 70 |
| MBPP pairs | 20 |
| GSM8K pairs | 30 |
| Hard/plausible wrong members | 120 |

The wrong-answer types are 34 partial-test-pass code answers, 56 semantic-test-failure code answers, and 30 near-miss-final math answers. All 90 code wrong members preserve the expected interface and fail semantically; none is included merely because of a trivial syntax or formatting failure.

## Fresh verification

- Correct members passing fresh verification: **120/120**
- Wrong members failing fresh verification: **120/120**
- Selected rows with verifier polarity failures: **0**
- Synthetic wrong answers: **0**

## Shortcut controls

Each problem contributes both labels, so task, dataset, and domain identity cannot directly identify the decision. The paired members use the same neutral-review template. Each of the five templates appears exactly 24 times with `KEEP` and 24 times with `REVISE`.

Tokenizer-length means are 151.34 tokens for correct answers and 150.14 for wrong answers. The standardized mean difference (wrong minus correct) is **-0.009**, indicating negligible aggregate token-length imbalance. Character-length SMD is **0.219**; correct and wrong medians are 350.5 and 421.5 characters respectively.

Two residual shortcut risks remain and should be monitored in downstream splits and evaluation:

- **Formatting risk:** only 27/90 code pairs have matching code-fence usage. Model-token lengths are nevertheless closely balanced, and answers were not rewritten to manufacture similarity.
- **Origin risk:** 77 correct members use verified references, while every wrong member is a natural Base/V1 attempt. The other 43 correct members are natural Self-Correction V1 outputs. This makes reference-vs-model style a possible medium-strength cue.

## Duplicate and overlap checks

- Duplicate pair IDs: **0**
- Duplicate source IDs: **0**
- Duplicate normalized source/problem pairs: **0**
- Frozen-evaluation overlap: **0**
- HumanEval overlap: **0**
- CW answers used as ordinary wrong examples: **0**

The builder excluded 478 candidates rather than weakening verification: 444 lacked a natural plausible wrong answer, 32 candidate wrong answers passed fresh verification, one candidate correct answer failed fresh verification, and one verifier failure was not a semantic wrong-answer result.

## Validation conclusion

All requested structural and verifier validations pass. The 120-pair set is large and clean enough for a **contrastive-training pilot**, with the formatting and answer-origin risks above treated as explicit evaluation variables rather than hidden assumptions. Router V4 training, class reweighting, and synthetic answer generation were not performed.
