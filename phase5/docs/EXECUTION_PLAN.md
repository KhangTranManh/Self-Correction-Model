# Phase 5 execution plan after initial collection

**Status (2026-09-23):** CPU source selection and the first GPU collection are
complete. Guided reviews, representation probes, and protected evaluation have
not run. This is a diagnostic experiment; no new model training or promotion is
planned from the pilot alone.

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

1. **Freeze a source-disjoint split locally.** Verify the final hashes and
   ordered audit. Choose 240 natural correct and all 240 natural wrong sources
   by a written deterministic rule and seed, before viewing review or probe
   outcomes. Assign balanced train (120 + 120), development (40 + 40), and
   protected test (80 + 80) sources. Save the exact IDs, source hashes,
   selection rule, seed, and manifest hash. The protected set is not a source
   for prompt changes, model choice, or training.
2. **Audit hint feasibility locally.** Freeze a rule for locating the model's
   first erroneous calculation and for naming an error type. Verify each hint
   independently of the reviewing model, and ensure no reference answer,
   correction, or verifier trace enters its prompt. The earlier
   [selection rules](../data/SELECTION_RULES.md) called for this alignment
   rule *before the first GPU pass*, which did not happen. Record that
   deviation. Neutral/status review and the pre-hint probe can use the frozen
   split; location/type
   analyses under the current run must be labeled exploratory unless a new
   untouched source pool is registered with the alignment rule frozen before
   its initial generation. Report eligible coverage;
   never refill failed cases from the protected set.
3. **Lock the GPU protocol before model calls.** Verify the exact parent and
   SHA-256 of the Phase 4 warm-start V2 and correction SFT V3 adapters. Freeze
   the same initial answer, prompt, output contract, decoding settings, seed
   schedule, precision, and verification method across the original solver and
   both adapters. Check adapter metadata and prompt/output parsing locally;
   keep the protected set sealed.
4. **Run paired guided reviews.** Start with a small train/development GPU
   smoke test for loading and output validity. Then, for each eligible source,
   checkpoint, and condition, collect neutral, truthful status, location, and
   type reviews as permitted by step 2. Store every exact hint, raw output,
   parsed KEEP/REVISE
   decision, final verified answer, and failure. Compare wrong-to-correct fixes
   with correct-to-wrong harms and contract validity. Status hints disclose
   correctness; improvement there is assisted repair, not autonomous error
   detection.
5. **Probe before the hint.** Extract frozen hidden states at the registered
   problem-plus-initial-answer position for each checkpoint. Fit regularized
   linear probes and surface-feature/shuffled-label controls on train; choose
   layer and regularization on development only. Read the protected test once
   after prompts, eligibility, metrics, and selection rules are locked.
6. **Report the result, including failures.** Publish paired accuracy changes,
   source-level uncertainty, fixes, harms, probe balanced accuracy and both
   class recalls, hint coverage, and all deviations. A pilot result alone does
   not promote a checkpoint or establish a solution to Phase 3's error-detection
   problem.

Steps 1-3 are local CPU work. The GPU is needed again for steps 4-5 after those
gates are complete. Keep raw evidence and checked copies on both machines.
