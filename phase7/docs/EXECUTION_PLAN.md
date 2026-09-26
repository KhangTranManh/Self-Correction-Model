# Phase 7 execution plan

**Status: execution completed.** The [final report](FINAL_REPORT.md) records
the protected result; this file preserves the planned execution sequence.

The run began on a V100 32 GB, then resumed on an RTX 3090 24 GB after the
V100 host disappeared. Both hosts used pinned vLLM 0.7.0 with FP16 weights,
one checkpoint at a time. The original-solver smoke passed on both hosts;
no quantized result entered the paired analysis. The initial collector was
`scripts/collect_initials_vllm.py`.

For a future 16 GB GPU, use the archived 4-bit scripts after freezing a
separate quantized protocol. Before any protected work, verify GPU compute
capability, available VRAM, CUDA/Python support, host RAM and disk, V2/V3
merge lineage, and one checkpoint's actual peak allocation. Materialize V2 at
FP16 on CPU before 4-bit loading. Run V2 from the quantized merged V2 model and
V3 by attaching its adapter to the same quantized merged V2 parent. Run one
model at a time with batch size one. If that cannot fit, a 24-32 GB GPU or
more host RAM is needed; do not substitute a different model.

1. **Completed - CPU source audit.** Inventory raw arithmetic sources and all Phase 1-6
   source IDs/normalized questions. Exclude the full Phase 5 selected pool
   and Phase 6 160-source pool. Produce a candidate manifest, exclusion
   counts, source hashes, verifier/version record, and deterministic order.
   The audit verified 286 prior inventory files, found 1,974 eligible sources,
   and froze 600 candidates under `data/candidates_v1/`. The manifest SHA-256
   is `ac1ebd0612e20c599905c547554d0c95d07f0932e99c58aef6e1dd478fb11799`.
2. **Completed - freeze initial collection.** The resumable FP16 vLLM collector ran
   from `scripts/collect_initials_vllm.py`. The lock at
   `data/protocol/initial_collection_v1_lock.json` pins the original-solver checkpoint,
   original Phase 1 math prompt, generation settings, seed schedule, maximum
   600 candidates, and first-prefix 100/100 stop rule. Package only code,
   manifests, and public configuration for the future GPU; keep `.env` and
   credentials out of archives. Verify the three checkpoint revisions and
   adapter-parent order before loading models. Search old backups for the
   exact frozen Phase 5 probe; its documented local directory is currently
   absent. Hash-check recovered artifacts before any probe route is scored.
3. **Completed - GPU initial answers and local backup.** Generate and verify one natural
   first answer per ordered candidate. Append raw outputs and verifier
   verdicts durably under `outputs/phase7_initials_v1/`; sync the audit to this
   workstation during the run. Once
   the 100/100 quota is met, run `scripts/freeze_balanced_split.py` to select
   and hash-lock the 40 development and 160 protected sources before any
   re-solve generation. Within each initial-correctness class, sort by
   SHA-256 of `phase7-split-v1|20260924|problem_id`; take the first 20 for
   development and the next 80 for protected. This rule was fixed before
   the initial-answer collection ended.
4. **Completed - lock the paired protocol.** Finalize the two exact prompt templates,
   independent chat construction, FP16/decoding budget, source/arm ordering,
   failure handling, frozen probe artifact/threshold, output parser, metrics,
   and protected opening procedure. Check automatically that blind prompts
   contain no prior answer or prior-answer metadata. Save a machine-readable
   protocol lock and SHA-256 for every controlling input.
5. **Completed - technical smoke and development.** Run both arms on a fixed
   development subset across all three checkpoints. Confirm loading, memory,
   valid output, and verifier operation. No prompt search against outcomes.
   Finish development collection and freeze the analysis code before opening
   protected results.
6. **Completed - one protected generation and evaluation.** Run paired arms on the locked
   160 sources, save all outputs, then perform one protected analysis. Apply
   frozen-probe routing only after generation and only if its artifact was
   recovered and verified; report oracle-known-wrong and non-oracle results
   separately. Compute paired intervals, harms, detector failures, answer
   agreement, and final system accuracy. If the probe is unavailable, do not
   create a replacement from protected outcomes.
7. **Completed diagnostic - further work needs a new lock.** Write a final report with every
   deviation and uncertainty. Do not train on or retune against the protected
   sources. If repair remains weak, preserve the negative result. If blind
   re-solving helps, require a separate fresh confirmation before promotion.

The run completed all 240 development and 960 protected paired outputs.
Protected analysis was opened once after all generations finished, and its
local rerun matched the remote report hash. The frozen Phase 5 probe artifact
was unavailable, so non-oracle routed analysis was not performed. No model
training or promotion occurred.
