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

## Active work: Phase 16 recovery-first definition and CPU data

Prospective intervention order is frozen in
`phase16/docs/RECOVERY_FIRST_V1.md`: deterministic output recovery first,
constrained decoding only when recovery cannot safely close the exact output
contract, then a bounded stateful harness. Self-edits are proposal data only;
they require schema, leakage, independent-oracle, near-negative, duplicate,
and human admission gates before a later batch freeze. Micro-update is allowed
only for independently verified residual `substantive_content` failures that
the frozen recovery wrapper and harness do not repair.

The update-resource curve uses nested correction counts 8/16/32/64, a fixed
evaluation set of 80 applicable probes, 20 near-negatives, and 40 unrelated
controls, with three update-training seeds per level. Stable onset requires
two adjacent passing levels plus reset, hash, audit, leakage, local-rescore,
family-coverage, and harm gates. CPU scorer:
`phase16/scripts/recovery_first_gate.py`. This is a definition only.

The CPU software contract is implemented in
`phase16/docs/RECOVERY_FIRST_CPU_CONTRACT_V1.md` and
`phase16/scripts/validate_recovery_first_cpu.py`. It validates recovery
evidence, immutable harness memory plus bounded run logs, self-edit admission,
the exact 64/80/20/40 pool, and an aggregate cross-artifact release. The
release validator recomputes hashes/decisions and explicitly does not
authorize GPU execution. No real release artifact
exists yet.

P16-G0 now separates inference-only residual discovery from update release.
All nine pinned TACO train shards are present again and match the recorded
byte sizes and SHA-256 values; TACO test remains unopened. The reproducible
private bank in ignored `phase16/data/a0_discovery_v1/` has 1,323 structural
call-based candidates (1,322 CodeWars, 1 LeetCode), canonical SHA-256
`1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352`.
The tracked public index is
`phase16/results/a0_candidate_bank_summary_v1.json`. A remote networkless
dual-reference and tokenizer preflight must freeze exactly 400 families before
four 100-item A0 shards. Until that lock exists, A0 is not authorized. Even
after it exists, it authorizes inference only: dataset tests are not independent
oracles, and training/protected opening remain closed. Protocol and runner:
`phase16/docs/A0_DISCOVERY_GPU_V1.md` and
`phase16/scripts/colab_a0_discovery.py`.
The ready-to-transfer ignored bundle is
`phase16/data/a0_discovery_v1/p16_g0_gpu_bundle.zip` (1,877,501 bytes,
SHA-256 `2c3a40c2750e93638b4399783f908178013f0864301008893f96a8922bfc6175`).

2026-11-08 remote preflight v1 accepted 1/400 and stopped. Failed audit:
`phase16/results/a0_preflight_fail_v1.jsonl` (SHA-256
`9b42a67be42c72d802e69a2c6ead1f9a81fa452386d321aef84b4c9d0866f784`).
Comparison amendment: `phase16/docs/A0_PUBLISHED_TEST_MATCH_V1.md` (exact
match plus one singleton expected-list unwrap). Copy the updated
`colab_a0_discovery.py` before the next preflight. Use a **new** output
directory. Do not splice the failed audit into a lock.

The fresh v2 preflight then accepted 400 families after two rejects and froze
lock `f655a2f3de813bc1d420d37d105dff7d881156e37b05b34a44cd155e04ae3b4c`.
All four 100-item Qwen shards completed and were downloaded to
`phase16/results/a0_discovery_run_v2/`. The hash/aggregate manifest is
`phase16/results/a0_discovery_run_v2_manifest.json`. Aggregate: 156 correct,
169 wrong answer, 57 runtime error and 18 syntax/interface error. These 244
dataset-side failures are screening candidates only.

Learner remains `Qwen/Qwen2.5-7B-Instruct`. P16-G0 is complete and the next
gate is P16-R0 CPU rescore plus independent-oracle admission. There is still
no certified study pool, micro-update, LoRA run, protected opening or effect
claim.

The model-free saved-output rescore is
`phase16/results/a0_discovery_run_v2/a0_rescore_v1.json`. It reproduced the
aggregate labels but found one payload disagreement: `taco_train_05701`
(position 171) uses unordered set iteration. Eight additional processes
returned different output hashes with match counts 6, 8 or 9/10. Evidence is
`a0_rescore_nondeterminism_v1.json`; reject this item for deterministic-rerun
failure.

The fail-closed CPU worklist is
`phase16/results/a0_admission_inventory_v1.json`; builder and protocol are
`phase16/scripts/build_a0_admission_inventory.py` and
`phase16/docs/A0_CPU_ADMISSION_V1.md`. It accounts for all 244 failures:
75 execution/scaffold records plus the nondeterministic wrong answer are
rejected, 168 deterministic `substantive_content` candidates await
task-specific independent-oracle, contract, full-history, near-duplicate and
human review, and zero corrections are admitted. `.gitattributes` pins the
tracked A0 JSON/JSONL evidence to LF so its remote byte hashes remain valid on
Windows. Training and protected opening remain unauthorized.

CPU review has begun without mutating the frozen inventory. Hash-chained
batches 1--5 are `phase16/results/a0_cpu_review_batch{1,2,3,4,5}_v1.json`.
Fourteen frozen positions have been examined: eleven have pairs of agreeing
contract-derived trusted oracles; positions 2, 16 and 36 are rejected for
under-specified contracts. The eleven oracle-ready records still need exact-
output sandbox, full-history/semantic and human gates. No untrusted model code
ran locally, zero corrections are admitted, and the next position is 37.
Review log: `phase16/docs/A0_CPU_REVIEW_LOG_V1.md`.

CPU admission progress is derived by
`phase16/scripts/track_a0_cpu_admission.py` into
`phase16/results/a0_cpu_admission_tracker_v1.json`; definition:
`phase16/docs/A0_CPU_ADMISSION_TRACKER_V1.md`. The tracker verifies the frozen
inventory and complete review hash chain before reporting 168 queued, 14
reviewed, 11 oracle-ready, three contract-rejected and zero admitted. The
11/14 oracle-ready point rate implies 82 total reviews for 64 oracle-ready
records (Wilson planning range 70--123), but this is not an admission forecast:
no admission gate has completed, so admission yield is intentionally unknown.

Batch 6 is `phase16/results/a0_cpu_review_batch6_v1.json`. It reviews the next
five frozen positions: four receive agreeing contract-derived oracle pairs and
position 37 is contract-rejected for under-specified FEN return serialization.
The cumulative funnel in `a0_cpu_admission_tracker_v2.json` is now 168 queued,
19 reviewed, 15 oracle-ready, four contract-rejected and zero admitted; next
position is 43. All remaining sandbox, history/semantic and human gates stay
pending, and no model code or GPU was used.

The earlier C-upper-inclusive source branch remains useful as capacity and
selection-bias evidence, but it is stopped and does not satisfy the new
recovery-first study roles.

Named error **C-upper-inclusive**: one explicit Python function; inclusive
`[L,R]`; output is count, sum, or select of elements with `L <= key <= R`.
The observable signature is two distinct upper-bound metamorphic pairs
(exclusive-upper mutant). Other failures are `other_failure`. Scorer:
`phase16/scripts/score_boundary.py`. Fixture: `phase16/docs/C_UPPER_INCLUSIVE.md`.

A text/shortlist hit is not eligibility. Family review is the contract gate.

KodCode-V1 pin `97d81f0695ddd7c5c02bfe16436db9a17a4a21e5`. Review unit is a
connected `seed_ids` family in
`phase16/results/kodcode_family_review_manifest.json` (627 families from 896
shortlisted rows). `phase16/docs/FAMILY_REVIEW_TAXONOMY_V1.md` records the
dated four-lane review amendment. The first 100 families in the original
hash-frozen order have been reviewed: 17 `direct_candidate`, 9
`execution_adapter_candidate`, 25 `control_candidate`, 49 `hard_reject` (24
with `deferred_multiop` status), and 0 fully certified.

Only direct+adapter counts toward applicable capacity: 26/100. The fixed pool
needs 260/627 (41.5%); the point projection is 163 and the descriptive Wilson
95% upper projection is about 222. The current 300-task C-upper-inclusive
branch therefore fails its CPU capacity gate. Do not start bulk oracle work,
roles, A0, LoRA, or GPU on this inventory. Artifacts and decision rationale are
in `phase16/docs/CPU_FAMILY_REVIEW_LOG_V1.md`.

Recall audit (CPU only): `phase16/results/kodcode_recall_sample_v1.json`
freezes 100 of 3,469 broad predicted-negative components (SHA-256
`d1a97333184f50b831703d93a019e3d251965b3e4f0c4c57287eb3f24fee632c`).
Review found 4 false negatives and 96 true negatives. Negative-stratum FNR is
4.0% (Wilson 95% 1.57--9.84%); paired descriptively with 26.0% shortlist
precision, plug-in recall is 54.02% with 25.27--80.32% sensitivity. The 627
shortlist families merge to 625 positive components in the broad graph, so
this is not formal population inference. No task was certified; the capacity
and GPU gates remain closed. Details: `phase16/docs/KODCODE_RECALL_AUDIT_V1.md`.

P16-E1 engineering smoke: Qwen 9/9 on synthetic tasks. Those tasks are
excluded from any study pool.

TACO train pin `d593ed0a2becbbc952230bb89be09189bf1056dc`. TACO test is
unopened. CodeContests, EvalPlus, LiveCodeBench, LeetCodeDataset, KodCode
were audited on CPU; none is a certified C pool.

## Next decision (CPU admission)

Continue at frozen position 37, constructing task-specific deterministic
oracles from each contract rather than copied published expected values or
dataset reference solutions. Separately, the eleven oracle-ready positions
require sandbox execution of the exact hash-frozen outputs, full history and
cross-history semantic-near-duplicate review, plus human approval. Only
admitted residuals may populate the real 64-correction prefix and 80/20/40
held-out roles. If fewer than 64 survive, report the stable-onset experiment
as infeasible. Do not run LoRA, micro-update, protected data, or claim an
effect until the aggregate CPU release and a later update execution protocol
pass.
