# Continuation snapshot

Workspace: `D:\AGI`. Remote: `https://github.com/KhangTranManh/Self-Correction-Model.git`.
Read this, then `phase16/README.md` and `phase16/RUN_STATUS.md`, before acting.

## Git

Raw Phase 16 source dumps stay on disk for Drive. They are gitignored:
parquet shards, `taco_raw/`, `codecontests_candidate_raw/`, `four_source_raw/`,
`extended_source_raw/`, `mathematics_dataset_source/`, `python_deps/`, TACO
`train_metadata.jsonl`, and the large four-source/KodCode text-audit JSON.
Git tracks protocol, scripts, tests, small review JSON, and inventory
summaries.

Local `main` can sit behind `origin/main`. Pull `--ff-only` before push.
`gh` may be logged out.

## What is closed

Phases 1–14 are closed. Phase 14 produced the paper
(doi:10.5281/zenodo.23541462). Phase 15 development diagnostics are closed;
the reserved final evaluation was never opened. Reports live under each
`phaseN/docs/FINAL_REPORT.md`.

## Active work: Phase 16 CPU data

Question: can one verified correction reduce a named content error on unseen
tasks without harming unrelated tasks? Learner: `Qwen/Qwen2.5-7B-Instruct`.
GPU gate is **closed**. No certified study pool, no A0 screen, no LoRA.

Named error **C-upper-inclusive**: one explicit Python function; inclusive
`[L,R]`; output is count, sum, or select of elements with `L <= key <= R`.
The observable signature is two distinct upper-bound metamorphic pairs
(exclusive-upper mutant). Other failures are `other_failure`. Scorer:
`phase16/scripts/score_boundary.py`. Fixture: `phase16/docs/C_UPPER_INCLUSIVE.md`.

A text/shortlist hit is not eligibility. Family review is the contract gate.

KodCode-V1 pin `97d81f0695ddd7c5c02bfe16436db9a17a4a21e5`. Review unit is a
connected `seed_ids` family in
`phase16/results/kodcode_family_review_manifest.json` (627 families from 896
shortlisted rows). Frozen-order Batch 1 (20 families): 6
`applicable_candidate`, 1 `near_negative_candidate`, 13 `reject`, 0 fully
certified. Artifact:
`phase16/results/kodcode_family_contract_review_batch1.json`. Log:
`phase16/docs/CPU_FAMILY_REVIEW_LOG_V1.md`.

Rejects stay in the inventory. They are out of the **positive** C pool
because the statement is not the locked contract (incidental range language,
stdio/multi-op, missing signature, first-scalar instead of selected set,
malformed scaffold). `Algorithm_38927_I` is near-negative: exclusive upper
endpoint. Planned 300-task S0 split (160/40/20/20/60) is defined in
`phase16/docs/SYNTHETIC_DEVELOPMENT_AMENDMENT_V1.md` and is not frozen.

Batch 2 recorder `phase16/scripts/record_kodcode_family_review_batch2.py` may
exist; Batch 2 JSON was missing when last checked. Chat proposed four lanes
(`direct_candidate`, `execution_adapter_candidate`, `control_candidate`,
`hard_reject` plus `deferred_multiop`). Those names are **not** in protocol
files yet.

P16-E1 engineering smoke: Qwen 9/9 on synthetic tasks. Those tasks are
excluded from any study pool.

TACO train pin `d593ed0a2becbbc952230bb89be09189bf1056dc`. TACO test is
unopened. CodeContests, EvalPlus, LiveCodeBench, LeetCodeDataset, KodCode
were audited on CPU; none is a certified C pool.

## Next CPU step (do not skip to GPU)

1. Record the four-lane taxonomy in a dated protocol amendment if that is
   still the plan, then relabel the first 40 families in frozen order.
2. Review through 100 families. Report direct / adapter / control / hard
   reject counts. Estimate capacity on 627 families against 260 applicable.
3. Oracle preflight only if direct+adapter can reach 260. Then history /
   near-duplicate / sandbox. Then roles. Then validator
   `phase16/scripts/validate_synthetic_pool.py`.
4. Keep secrets in `.env`. Do not rent GPU until those gates pass.

## Local agent extras (not the study)

`.grok/skills/` holds 58 untracked pstack skills copied from
`cursor/plugins` `pstack/skills` (535 files). They are project-local for
Grok. They are unrelated to Phase 16 data. `.codex/skills` was not writable
under local ACLs.
