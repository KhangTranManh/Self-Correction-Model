# Phase 16 engineering results

These files are engineering artifacts. They are excluded from the future
40-episode pilot and protected confirmation set.

- `colab_definition_smoke.json`: CPU-only scorer mechanics on three trusted
  task variants, with no model output.
- `colab_runner_smoke.json`: CPU-only isolated runner mechanics on nine
  trusted tasks, with no model output.
- `p16_e1_audit.jsonl`: append-only Colab T4 A0 engineering run, including
  metadata, stages, all nine raw model completions and their scores; SHA-256
  `11b5f13417fe5af11b353512f1df227c1ff30d7d725417a2c32979c29433e420`.
- `p16_e1_summary.json`: nine completed, nine correct, no boundary errors.
- `p16_e1_rescore.json`: repeat execution of all saved completions in the
  isolated verifier; zero disagreements.

The live partial snapshot used during monitoring was removed after the full
audit was downloaded and verified. See `../docs/ENGINEERING_A0.md` for the
frozen specification and interpretation.

## CPU source-review artifacts

These are contract-review records, not model results or certified study data:

- `kodcode_family_contract_review_batch1.json` and
  `kodcode_family_contract_review_batch2.json`: original three-label reviews
  for frozen family indices 0--39.
- `kodcode_family_lane_review_40_v1.json`: taxonomy-v1 relabel of indices
  0--39 while preserving the original labels.
- `kodcode_family_lane_review_batch3.json`: taxonomy-v1 review of indices
  40--59.
- `kodcode_family_lane_review_60_99.json`: taxonomy-v1 review of indices
  60--99.

Across the first 100 hash-frozen families, 17 are direct candidates, 9 need a
permitted interface adapter, 25 are control candidates and 49 are hard
rejects. None is fully certified. See
`../docs/CPU_FAMILY_REVIEW_LOG_V1.md` for the capacity decision.

- `kodcode_recall_sample_v1.json`: hash-frozen 100-family sample from the
  3,469 broad components omitted by the semantic shortlist; it retains only
  sampled representatives and summary counts.
- `kodcode_recall_review_v1.json`: taxonomy review of that sample: 4 false
  negatives and 96 true negatives, plus the two-stratum precision/recall
  estimates. It is not a certification or model result.

See `../docs/KODCODE_RECALL_AUDIT_V1.md` for the sampling definition,
limitations and interpretation.

## Recovery-first CPU reports

`../scripts/validate_recovery_first_cpu.py` may write validation reports here
only from frozen, reviewable input artifacts. No real recovery-first CPU
release report exists yet: the current source branch does not supply 64
certified residual corrections plus the 80/20/40 held-out roles. Synthetic
coverage lives in `tests/test_validate_recovery_first_cpu.py` and is not a
study result.

## A0 residual-discovery preparation

`a0_candidate_bank_summary_v1.json` is the non-text public index for the
gitignored private TACO train candidate bank. It records 1,323 structural
call-based candidates in frozen hash order, their family/prompt/test/reference
hashes, and the strong 1,322 CodeWars / 1 LeetCode source concentration. It is
not model output, independent-oracle certification, or GPU authorization. The
remote preflight in `../docs/A0_DISCOVERY_GPU_V1.md` must freeze exactly 400
passing families before inference, and training remains prohibited.

## A0 residual-discovery execution

`a0_discovery_run_v2/` is the completed inference-only P16-G0 run. Its fresh
preflight froze 400 distinct families under lock SHA-256
`f655a2f3de813bc1d420d37d105dff7d881156e37b05b34a44cd155e04ae3b4c`,
then four immutable 100-item shards completed. The aggregate is 156 correct,
169 wrong answer, 57 runtime error and 18 syntax/interface error. The 244
dataset-side failures are CPU-screening inputs only.

`a0_discovery_run_v2_manifest.json` records every downloaded file SHA-256,
preflight counts and aggregate shard counts. The run has no independent-oracle
validation and authorizes neither training nor protected data opening.

`a0_discovery_run_v2/a0_rescore_v1.json` is a model-free re-execution of all
400 saved recovered outputs in the frozen sandbox. It reproduced the four
label totals, but one item (`taco_train_05701`, position 171) changed its
verification payload while remaining `wrong_answer`. Its use of unordered set
iteration produced match counts of 6, 8 or 9 across repeated fresh processes.
`a0_rescore_nondeterminism_v1.json` preserves that evidence; the item fails the
deterministic-rerun gate and is ineligible for R0 admission.

## A0 CPU admission inventory

`a0_admission_inventory_v1.json` is the hash-linked, fail-closed worklist for
all 244 dataset-side failures. It rejects 75 frozen execution/scaffold records
and the one nondeterministic wrong-answer record, queues 168 deterministic
`substantive_content` candidates for independent-oracle review, and admits
zero corrections. It contains no prompt text, reference solution, published
test payload, expected value or model code. See
`../docs/A0_CPU_ADMISSION_V1.md` for the integrity and review boundary.
