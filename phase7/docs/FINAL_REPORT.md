# Phase 7 final report — blind versus answer-visible re-solving

**Status: completed diagnostic, no model or router promoted.** All three
frozen checkpoints finished both paired arms on the 40-source development
set and the 160-source protected set. The protected analysis was opened once
after all 960 protected outputs were complete. The remote pipeline ended with
`PHASE7_PIPELINE_COMPLETE`; the local copy of every paired audit and output
matches its remote-generated summary hash. Re-running the protected analysis
locally produced the same report SHA-256,
`95f9da44b66fbc359afc1f5f24c4db51cdaa6e64a9fd94bd82a07e06dd0cb2c6`.

## Question and design

Phase 6's repair attempts rarely fixed an initially wrong answer, even when
an oracle identified it. Phase 7 tested whether exposing the earlier answer
contributes to that weakness. The **blind** arm gave a fresh single-turn solve
prompt with only the original problem. The **answer-visible** arm gave the
same solve prompt plus the complete earlier answer labeled as an unverified
candidate. Neither arm received a correctness label, verifier trace, probe
score or reference answer. The same source and checkpoint received both arms.

The source audit found 1,974 eligible fresh problems after Phase 1–6
exclusions and froze 600 in hash order. The original solver generated natural
first answers until the first quota prefix at 334 problems: 234 verifier-
correct and 100 verifier-wrong. A predetermined hash rule selected 20/20
development and 80/80 protected sources. Initial answers were held fixed
across all three checkpoint comparisons. Generation used vLLM 0.7.0, FP16,
4096-token context, greedy paired decoding, and a 768-token output cap.

## Protected result

Each checkpoint was evaluated on the same 80 initially wrong and 80 initially
correct protected sources. The primary difference is blind minus visible
wrong-to-correct rate; confidence intervals are source-level paired bootstrap
95% intervals. The listed p-values are exact paired tests with Holm correction
across the three checkpoints.

| Checkpoint | Blind fixes / 80 wrong | Visible fixes / 80 wrong | Difference (95% CI) | Holm p | Blind harms / 80 correct | Visible harms / 80 correct |
|---|---:|---:|---:|---:|---:|---:|
| Original solver | 25 (31.25%) | 4 (5.00%) | +26.25 pp [16.25, 36.25] | 0.00000572 | 8 (10.00%) | 19 (23.75%) |
| Warm-start V2 | 28 (35.00%) | 5 (6.25%) | +28.75 pp [18.75, 40.00] | 0.00000310 | 5 (6.25%) | 15 (18.75%) |
| Correction SFT V3 | 32 (40.00%) | 8 (10.00%) | +30.00 pp [20.00, 41.25] | 0.00000241 | 6 (7.50%) | 18 (22.50%) |

The paired wrong-row tables were, respectively, blind-only/visible-only
22/1 for original, 24/1 for V2, and 25/1 for V3. This is a large,
consistent answer-visibility effect under the frozen prompts. All three
primary intervals exclude zero and all Holm-adjusted p-values are below
0.05, meeting the preregistered evidence gate for H1. The low second-pass
solving-ceiling hypothesis H2 is **not** supported by these blind-arm rates.

Blind re-solving also produced fewer correct-to-wrong changes than the
answer-visible arm. This does not make an unconditional rerun safe: even the
blind arm harms 5–8 of 80 initially correct sources for V2/V3 and 8 of 80
for original. On the deliberately balanced set, always taking the blind
second answer would yield 60.625%, 64.375%, and 66.25% accuracy for original,
V2 and V3, respectively, versus the constructed 50% initial baseline. These
figures are descriptive for this selected set and are **not** deployment
accuracy. An oracle that routes only known-wrong rows would yield 65.625%,
67.50%, and 70.00%; that condition is not autonomous detection and was not
shown to the model.

## Limits and disposition

The frozen Phase 5 probe artifact was absent from its documented local
location and could not be recovered. Therefore the preregistered non-oracle
routed result, including detector recall, false positives and final routed
accuracy, is **unavailable**. No replacement probe or threshold was fitted
on protected data. The Phase 7 result supports a causal effect of making the
old answer visible under these paired prompts; it does not uniquely prove an
internal psychological anchoring mechanism. Prompt framing and information
content are part of the intervention.

The original-solver initial answers and most development outputs were
generated on a V100 32 GB. That host became unavailable during V2
development. A checked resume bundle moved the existing audits to an RTX
3090 24 GB. The original-solver FP16 smoke passed at 19,430 MiB on the new
GPU, and all 20 paired-protocol lock inputs—including the rebuilt V2 FP16
shards—matched the original lock. One V2 **development** pair straddled the
host change because its first arm was already saved; all protected pairs for
all checkpoints ran entirely on the RTX 3090. vLLM's attention backend
therefore differs between part of development and protected, but the
protected paired comparison does not mix GPUs within a source or arm.

No model was trained in Phase 7. No checkpoint, probe, or autonomous
self-correction system is promoted. The next step, if pursued, should freeze
a fresh confirmation set and test an actual pre-answer router plus blind
re-solving end to end. The current protected set must not be used for
retuning prompts, seeds, thresholds or models.

## Artifacts and integrity

- Initial audit: `outputs/phase7_initials_v1/`; audit SHA-256
  `f396a37e76da9281fb0f1120dcf8cb979b4ecac06d3827dd241a5a738b14812b`.
- Source manifest: `phase7/data/candidates_v1/candidate_problems.jsonl`;
  SHA-256 `ac1ebd0612e20c599905c547554d0c95d07f0932e99c58aef6e1dd478fb11799`.
- Balanced split: `phase7/data/split_v1/`; paired lock SHA-256
  `f468394d73726595d913252a1ce897ae2f2bf7dcdfc7719c8ad726f271e4fa46`.
- Six paired output sets and both analyses: `outputs/phase7_paired_v1/`.
  Every audit/output pair matches its `summary.json` hash on the local
  workstation. The remote GPU also retains the code and results under
  `/root/AGI_phase7/`.
