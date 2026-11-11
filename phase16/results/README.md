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

## A0 CPU admission review

`a0_cpu_review_batch1_v1.json` starts the fail-closed review in frozen order
at positions 0, 2 and 6. One task has a trusted contract-derived oracle ready
but still lacks sandbox, full-history/semantic and human gates; one is rejected
for an under-specified contract; and one is deferred pending a specialized
guitar-tab grid oracle. The artifact executes no untrusted model code, admits
zero corrections and authorizes neither training nor protected opening. See
`../docs/A0_CPU_REVIEW_LOG_V1.md`.

`a0_cpu_review_batch2_v1.json` hash-chains to batch 1, resolves the specialized
guitar-tab oracle at position 6, and reviews positions 10 and 12. All three
records have pairs of agreeing contract-derived trusted CPU oracles that catch
their corresponding trusted model-equivalent mutants. Cumulatively, four
records are oracle-ready for remaining gates, one is contract-rejected and
zero are admitted. Exact recovered model code was not executed locally.

`a0_cpu_review_batch3_v1.json` hash-chains to batch 2 and reviews positions 16,
21 and 22. Position 16 is contract-rejected because whitespace-run placement
is not uniquely specified; positions 21 and 22 have agreeing independent
trusted oracles. Cumulatively, six records are oracle-ready, two are contract-
rejected and zero are admitted. The next frozen position is 23.

`a0_cpu_review_batch4_v1.json` hash-chains to batch 3 and reviews positions 23,
28 and 29. Closed-form/binary-search material-loss, integer/rational race-time,
and integer/Decimal schedule oracles agree and catch their corresponding
trusted mutants. Cumulatively, nine records are oracle-ready, two are contract-
rejected and zero are admitted. The next frozen position is 30.

`a0_cpu_review_batch5_v1.json` hash-chains to batch 4 and reviews positions 30,
31 and 36. Vowel-run/regex and direct-hex/parity oracle pairs agree and catch
their corresponding trusted mutants. Position 36 is contract-rejected because
first-letter ties and punctuation-token semantics are not fully specified.
Cumulatively, eleven records are oracle-ready, three are contract-rejected and
zero are admitted. The next frozen position is 37.

`a0_cpu_admission_tracker_v1.json` is a model-free CPU aggregation of the
frozen inventory and review batches 1--5. It verifies inventory/record hashes,
review self-hashes, the cross-batch chain, identities and allowed deferral
transitions before reporting the 168 -> 14 reviewed -> 11 oracle-ready -> 0
admitted funnel. Its 82-review point projection and 70--123 Wilson range refer
only to reaching 64 oracle-ready records; admission yield remains unavailable
because no admission decision is complete. See
`../docs/A0_CPU_ADMISSION_TRACKER_V1.md`.

`a0_cpu_review_batch6_v1.json` hash-chains to batch 5 and expands the review
unit to five frozen positions (37, 39, 40, 41 and 42). Four pairs of trusted
contract-derived CPU oracles validate diamond geometry, Gordon-language word
transformation, maximal common-remainder modulus and case-grammar conversion.
Position 37 is rejected because the FEN depiction's exact return serialization
is under-specified. Cumulatively, 19 positions are reviewed, 15 are oracle-
ready, four are contract-rejected and zero are admitted; next position is 43.

`a0_cpu_admission_tracker_v2.json` extends the verified tracker chain through
batch 6. The observed oracle-ready rate is 15/19 (0.7895), with Wilson 95%
interval 0.5667--0.9149. The descriptive total-review estimate for 64 oracle-
ready records remains 82, with a 70--113 planning range. Admission yield is
still unavailable because all 15 oracle-ready records retain remaining gates.

`a0_cpu_review_batch7_v1.json` hash-chains to batch 6 and reviews positions
43, 44, 47, 48 and 51. All five contracts pass, bringing the cumulative
funnel to 24 reviewed, 20 oracle-ready, four contract-rejected and zero
admitted. `a0_cpu_admission_tracker_v3.json` is the corresponding immutable
seven-batch snapshot.

`a0_cpu_review_batch8_v1.json` hash-chains to batch 7 and reviews positions
53, 56, 58, 60 and 61. Position 53 is rejected because the Morse converter's
multi-digit input and output serialization are under-specified; four trusted
oracle pairs agree and catch their model-equivalent mutants. Cumulatively, 29
positions are reviewed, 24 are oracle-ready, five are contract-rejected and
zero are admitted; next position is 62.

`a0_cpu_admission_tracker_v4.json` verifies the complete eight-batch chain.
The observed oracle-ready rate is 24/29 (0.8276), with Wilson 95% interval
0.6545--0.9240. Reaching 64 oracle-ready records is descriptively 78 total
reviews at the point rate, with a 70--98 planning range. Admission yield is
still unavailable because all 24 oracle-ready records retain remaining gates.

`a0_cpu_sandbox_closure_v1.json` records the completed CPU-only sandbox gate
for all 24 oracle-ready records. Every exact frozen recovered output completed
inside the local Docker/bubblewrap sandbox and failed at least one independent
case: 24 residuals reproduced, zero unexpectedly correct and zero corrections
admitted. It retains only identities, hashes, counts and sandbox observations;
independent inputs, expected values and candidate code are absent. Closure
SHA-256: `71fe8dbdf483ef380b359032b020318dc877ec4cc51a28a699f29e5badfa8ff2`.

`a0_cpu_review_batch9_v1.json` hash-chains to batch 8 and reviews positions
62, 64, 66, 68 and 70. Four independent oracle pairs agree and catch the
frozen model-equivalent mutants. Position 70 is rejected because essential
pattern semantics are delegated to unavailable tests and empty/singleton
behavior is unspecified. Cumulatively, 34 positions are reviewed, 28 are
oracle-ready, six are contract-rejected and zero are admitted; next position
is 71.

`a0_cpu_admission_tracker_v5.json` verifies all nine review batches. The
observed oracle-ready rate is 28/34 (0.8235), with Wilson 95% interval
0.6649--0.9165. Reaching 64 ready records is descriptively 78 total reviews,
with a 70--97 planning range. The first 24 ready records have completed the
sandbox gate; the four batch-9 records have not.

`a0_cpu_review_batch10_v1.json` reviews positions 71, 73, 74, 76 and 77.
Three trusted oracle pairs agree and catch their frozen mutants; two contracts
are rejected for undefined domain/no-solution or transition-precedence
semantics. `a0_cpu_admission_tracker_v6.json` verifies 39 reviewed, 31 ready,
eight rejected, 129 unreviewed and zero admitted; next position is 80.

`a0_cpu_review_batch11_v1.json` validates all five positions 80, 81, 82, 85
and 86. Tracker v7 verifies 44 reviewed, 36 ready, eight rejected, 124
unreviewed and zero admitted; next position is 87.

`a0_cpu_review_batch{12,13,14,15}_v1.json` split the verified 20-position
group-B definitions into four ordinary five-record hash-chain batches. Sixteen
contracts pass and four reject fail-closed. Tracker v8 verifies 64 reviewed,
52 ready, 12 rejected, 104 unreviewed and zero admitted; next position is 144.

`a0_cpu_review_batch{16,17,18,19,20}_v1.json` continue the immutable chain
through position 198. Seventeen contracts pass independent oracle/reference
agreement and eight reject fail-closed. Tracker v10 verifies 89 reviewed, 69
ready, 20 rejected, 79 unreviewed and zero admitted; next position is 202.

`a0_cpu_review_batch{21,22,23,24}_v1.json` extend the chain through position
255. Nineteen contracts pass and one rejects fail-closed. Tracker v11 verifies
109 reviewed, 88 ready, 21 rejected, 59 unreviewed and zero admitted; next
position is 260.

`a0_cpu_review_batch{25,26,27,28}_v1.json` extend the chain through position
318. Nineteen contracts validate an independently caught residual; one record
rejects because no contract-derived frozen-output failure exists. Tracker v12
verifies 129 reviewed, 107 ready, 22 rejected, 39 unreviewed and zero admitted;
next position is 321.

`a0_cpu_review_batch{29,30,31,32}_v1.json` extend the chain through position
356. Eighteen contracts validate and two reject fail-closed. Tracker v13
verifies 149 reviewed, 125 ready, 24 rejected, 19 unreviewed and zero admitted;
next position is 357.

`a0_cpu_review_batch{33,34,35,36}_v1.json` exhaust the frozen review queue
through position 399. Seventeen contracts validate and two reject fail-closed.
`a0_cpu_admission_tracker_v14.json` verifies all 36 batches and records 168
reviewed, 142 oracle-ready, 26 contract-rejected, zero unreviewed and zero
admitted. Its chain tip is
`e977129d03252664f1c5724c7777fab58ac71445d7e7f0a94ad4749e54dc09c2` and
its tracker SHA-256 is
`cd7fc94f01e64b35f5e9277e81cea3f21442c9f56a2cd94c3c330de39e328b94`.
Queue exhaustion completes contract/oracle review only; remaining admission
gates and GPU authorization stay closed.

`a0_cpu_sandbox_plan_v2.json` freezes twelve CPU execution batches for the
118 oracle-ready positions outside the earlier 24-record closure. The plan
verifies source identities and oracle evidence but contains no sandbox
results. See `../docs/A0_CPU_SANDBOX_CLOSURE_V2.md` for the comparison and
classification changes and the limitations of the earlier closure.

`a0_cpu_sandbox_v2/` now contains the twelve immutable execution batches,
the exact rerun of the prior 24 and a verified `summary.json`: 141 completed
outputs had independent-case mismatches, zero matched all cases and position
392 is unresolved because its test used the wrong input type. No correction
is admitted. The summary SHA-256 is
`72dc0309902724f80d4a4d4c29ff797b2dce1c99da9d0b73f07fac394e9c05ef`.
