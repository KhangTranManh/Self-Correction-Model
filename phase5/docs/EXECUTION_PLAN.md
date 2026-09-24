# Phase 5 execution plan after initial collection

**Status (2026-09-24): complete.** Guided reviews, representation probes,
controls, and the single protected opening finished without post-lock changes.
No model was trained or promoted. Results are in `phase5/docs/FINAL_REPORT.md`.

## Fixed inputs and current limit

The candidate manifest contains 1,200 new GSM8K train sources, with no recorded
Phase 1-4 source overlap. The original `Kxck/Self_Correction_v1` checkpoint
produced one natural initial answer for each of the first 815 candidates. Fresh
verification found 575 correct and 240 wrong answers; the run stopped at the
first ordered prefix meeting both 240-per-class quotas. The final local backup
is `outputs/phase5_remote_v100/` and its audit and rollout SHA-256 hashes are
recorded in `phase5/configs/experiments.yaml`.

The run recorded BF16 weights on a Tesla V100 without native BF16. PyTorch
selected software emulation despite the intended FP16 setup. Preserve the
as-run archive and setting as provenance. Choose and record one inference dtype
for all three reviewers before comparing checkpoints; do not relabel the
initial answers as FP16 or silently regenerate them.

## Next steps

1. **Completed - freeze a source-disjoint split locally.** The deterministic
   manifest contains 240 train, 80 development, and 160 protected rows, each
   balanced by initial correctness. Exact IDs, rules, seed, and hashes are in
   `phase5/data/splits/v1/manifest.json`. The protected set is not a source for
   prompt changes, model choice, or training.
2. **Completed - audit hint feasibility locally.** Strict parser V2 uses only
   literal single-operation equations from the model's own answer; it does not
   use a reference answer, correction, or verifier trace. Neutral/status covers
   all 480 rows. Location/type wrong-case coverage is one train row and zero
   development/protected rows, so those conditions are disabled rather than
   weakened or refilled. Parser V1 is retained as rejected audit evidence.
3. **Completed - lock the GPU protocol before model calls.** Private Hub
   revisions, exact parents, and SHA-256 values are pinned. The protocol fixes
   FP16, greedy decoding, strict XML, prompts, seed, metrics, bootstrap, probe
   layers/C-grid, and the V2-merge/V3-attach order. The config and controlling
   input hashes are in `phase5/configs/review_protocol_v1.yaml` and
   `phase5/data/protocol/review_protocol_v1_lock.json`.
4. **Completed - run paired guided reviews.** Begin with the frozen eight-source
   train GPU smoke across all three checkpoints and both active conditions.
   Require successful model loading, no OOM, and at least 75% strict contract
   validity. Then collect neutral/status train and development reviews. Store
   every exact prompt, raw output, parsed action, final verification, and
   failure. Status discloses correctness, so any gain is assisted repair rather
   than autonomous detection. Do not run location/type.
5. **Completed - probe before the hint.** Extract frozen hidden states at the registered
   problem-plus-initial-answer position for each checkpoint. Fit regularized
   linear probes and surface-feature/shuffled-label controls on train; choose
   layer and regularization on development only. Read the protected test once
   after prompts, eligibility, metrics, and selection rules are locked.
6. **Completed - report the result, including failures.** Publish paired accuracy changes,
   source-level uncertainty, fixes, harms, probe balanced accuracy and both
   class recalls, hint coverage, and all deviations. A pilot result alone does
   not promote a checkpoint or establish a solution to Phase 3's error-detection
   problem.

All steps are complete. The protected set was opened once only after
development choices and controls were frozen. Raw evidence is backed up under
`outputs/phase5_gpu_vllm/`; reopening it for model or threshold selection is
not allowed.
