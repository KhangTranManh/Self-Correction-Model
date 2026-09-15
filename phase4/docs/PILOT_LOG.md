# Phase 4 pilot log

Phase 4 closed on 2026-09-15. Historical entries below are retained unchanged
as experiment history; they do not authorize further runs. The final report
is `FINAL_REPORT.md` and final dispositions are in `configs/experiments.yaml`.

## 2026-09-15 — blind preferences and three-round review

The blind collection completed all 16,064 hint-free reviews. It contained 108
verified fix attempts and 42 harmful revisions, but source-level coverage was
insufficient: train KEEP/REVISE pairs 14/44, dev 2/4. Balanced datasets contain
28 train and four dev rows. The frozen minimums (20 train and five dev per
action) failed; no DPO V2 training was run and no threshold was lowered.

At the user's request, selected V2 reviewed evolving answers for three rounds
on the fixed fresh 80-source diagnostic at three seeds, without verifier
feedback. All 240 trajectories completed. Mean accuracy was 50% initially,
51.25% after round 1, 49.58% after round 2, and 49.17% after round 3. Extra
rounds did not improve the policy: retained fixes declined from three to one
and initial-answer harms increased from zero to three across seeded
trajectories. See `THREE_ROUND_DIAGNOSTIC.md` and
`phase4/runs/three_round_v1/summary.json`. No model was promoted.

## 2026-09-14 — Cycle 0 baseline

- Actor: `Kxck/Self_Correction_v1` merged BF16 checkpoint.
- New math source pool: 300 train, 100 development, 400 confirmation
  candidates; normalized questions absent from committed Phase 1/3 JSONL.
- Eight-row first smoke: 0/8 strict contracts because the actor reverted to
  its Phase 1 critique format.
- Prompt-only repair: an explicit selective-policy system message produced 8/8
  strict contracts in the second smoke.
- Full 300-row rollout: initial accuracy 71.67%, final accuracy 71.67%, 287 KEEP,
  8 REVISE, 5 invalid, and 1 wrong-to-correct transition.
- One-step GRPO integration smoke passed after a BF16 output-head compatibility
  fix, but the sampled correct group had zero reward variance.
- A wrong-only, eight-generation, temperature-1.2 probe also had zero reward
  variance: mean -1.0 and gradient norm 0.0.

## Pre-training amendment

Outcome-only reward gives the same score to KEEP-on-wrong and an unsuccessful
REVISE-on-wrong. With the observed KEEP-dominant initialization, this creates no
within-group GRPO advantage. Before any substantive training, the active pilot
reward was decomposed:

- wrong → correct: +2.0;
- wrong + REVISE but still wrong: -0.25;
- wrong + KEEP: -1.0;
- invalid contract: -1.25.

The deployment/evaluation parser remains exact. During GRPO only, a decision
tag embedded in a malformed completion is recovered for the decision component
of reward; an exact-contract bonus of +0.10 preserves pressure toward valid
deployment output. Every sampled completion and both parsing outcomes are
stored in `completion_audit.jsonl`.

## Exploration warm-start data

The actor received deterministic verifier detail only while generating warm-start
targets; that detail is absent from the training prompt. Of 85 natural Cycle 0
failures, 74 self-generated corrections passed fresh verification and the strict
contract. They were paired with 74 correct/KEEP cases and split by source into
120 training and 28 development rows. This stage initializes both actions; it is
not evaluated as the Phase 4 result.

Warm-start V1 (one epoch, 1e-5) did not establish exploration on the 100-row
development set: REVISE fell from 3 to 1 and strict validity fell from 98% to
96%, with final accuracy unchanged at 73%. A bounded V2 initialization uses the
same data for two epochs at the already established Phase 3 learning rate of
2e-5. No confirmation data is consulted.

The original GPU was removed before its warm-start artifacts could be copied
back. Reconstruction on the replacement RTX 3090 used the same Cycle 0 rows,
prompts, per-request seeds, checkpoint, and filters. Kernel-level generation
variation yielded 73 rather than 74 verified corrections, so V2 uses a balanced
118-row training split and 28-row development split. The reconstructed hashes
are recorded in `data/warmstart_v1/summary.json`; the lost V1 measurements above
remain historical evidence and are not represented as rerun results.

Warm-start V2 passed its gradient preflight and changed the adapter weights.
Development completion loss fell from 0.7524 to 0.1851. On the same 100 initial
answers as the rebuilt base evaluation, strict REVISE decisions rose from 3 to
8, invalid contracts fell from 2 to 1, and final accuracy rose from 73% to 74%.
The result satisfies the pilot gate of increased exploration without KEEP
collapse, so the next bounded step is GRPO initialized from the V2 policy.

A one-step, wrong-state GRPO smoke had reward variance only from the strict
contract bonus and sampled no revisions. The follow-up five-step probe covered
10 wrong prompts and 80 completions: 41 REVISE, 39 KEEP, 66 strict contracts,
39 revise-but-still-wrong transitions, and 2 wrong-to-correct transitions.
This established outcome-level within-group variance, so the registered
50-step V2 GRPO pilot proceeds from the merged warm-start checkpoint.

The V2 configuration (four candidates, temperature 0.8) was stopped after 19
of 50 steps: every observed prompt group had zero within-group reward standard
deviation and policy loss 0.0. V3 adopts the already successful probe settings
of eight candidates, temperature 1.2, and a 512-token cap. It trains for 20
steps on the 85 wrong Cycle 0 states; the full development evaluation still
includes initially correct states and therefore gates KEEP retention.

V3 completed 20 steps in 832 seconds. Its 320-completion training audit covered
40 wrong prompts and contained 140 REVISE, 180 KEEP, 5 wrong-to-correct, 135
wrong-revise-wrong, and 279 strict-contract completions. On the same 100-row
development comparison at temperature 0.8, V3 emitted 7 REVISE, 92 KEEP, and 1
invalid output; final accuracy was 73%. Because this did not beat V2's 74%, the
V2 warm-start adapter is selected as the pilot result and V3 is retained as a
negative result.

The sealed confirmation gate was not opened. The currently prepared 400
confirmation candidates are GSM8K math problems and are not balanced by domain
or initial correctness, while `PREREGISTRATION.md` requires both. They remain
untouched; a promotion claim requires a new compliant sealed evaluation set.

## Correction expansion V1

An additional deterministic slice of 2,000 source-disjoint GSM8K training rows
was selected after the original 800-row Phase 4 allocation without reading the
confirmation-candidate file. Selected V2 produced 1,493 correct and 507 wrong
natural initial answers. Four seeded, verifier-guided correction attempts per
failure yielded 1,845 passing attempts and at least one strict verified
correction for 502 of the 507 failures. After one-target-per-source selection
and balanced KEEP matching, the expansion contains 904 training rows and 100
development rows. `configs/correction_sft_v3.yaml` continues supervised QLoRA
from the merged selected V2 policy for one epoch at 1e-5.

Rewards for initially correct answers are unchanged. This amendment is a pilot
pipeline decision. It cannot be tuned against the future sealed confirmation
set, and the original Cycle 0 configuration is frozen in
`configs/cycle000_rollout.yaml`.

Correction SFT V3 completed 113 steps with nonzero adapter gradients. Held-out
completion loss on the balanced 100-row correction split fell from 0.2428 to
0.1464. On the unchanged 100-row Cycle 0 development comparison, it produced
86 KEEP and 14 REVISE decisions with 100% contract validity. It fixed three
initially wrong answers but changed two initially correct answers to wrong, so
final accuracy remained 74%, equal to selected V2. The checkpoint is retained
as evidence but does not replace selected V2.

## Preference optimization V1

The 904/100 correction SFT splits were converted one-for-one into balanced DPO
pairs. For an initially wrong answer, the strict verifier-passing REVISE output
is preferred over KEEP. For an initially correct answer, KEEP is preferred over
an unnecessary REVISE that repeats the same answer. This directly represents
the transition reward ordering while preserving equal KEEP/REVISE coverage.
The manifest records that confirmation candidates were not read. A one-epoch,
beta-0.1 QLoRA DPO run starts from the merged selected V2 checkpoint; behavioral
promotion remains gated by the same frozen 100-row development comparison.

On 2026-09-15 the user reported closing the GPU. The last remote observation
was 79/113 DPO steps; the most recent local partial log reaches 50/113. No final
DPO adapter or report exists in the local backup, so this run has no completed
result and cannot be evaluated or selected. Completed correction SFT V3,
verified correction data, preference pairs, and development results are local.
Future DPO runs save resumable checkpoints every 25 steps (keeping two), and
the training script accepts `--resume-from-checkpoint`. Recovering progress
still requires copying those checkpoints off the GPU before it is removed.

The replacement GPU (`joyful-axolotl`, RTX 3090 24 GB) was bootstrapped from
local code and verified adapter backups. The DPO run restarted from selected
V2 and completed 113 steps in 1,607.9 seconds. Held-out preference loss fell
from 0.6931 to 0.5863, pairwise preference accuracy reached 97%, and the mean
preference margin reached 0.2353. The final adapter passed a nonzero-weight
inspection and its local/GPU SHA-256 hashes agree:
`5bad7cbe4adf144b0093cad9953c93360a157c88918462df40d898407548533e`.

On the frozen 100-row behavioral development set with reused initial answers,
DPO V1 produced 94 KEEP, 5 REVISE, and 1 invalid output: final accuracy 73%,
zero wrong-to-correct transitions, and zero correct-to-wrong transitions. A
selected-V2 rerun on the same replacement GPU produced 93 KEEP, 5 REVISE, and
2 invalid outputs, also 73% final accuracy with no fixes or regressions. The
earlier V2 74% result was not reproduced in this rerun; this small pilot does
not establish a reliable gain. DPO V1 is retained as a negative behavioral
result and selected V2 remains unchanged. Confirmation remains unopened.

Resumable checkpoints 25, 50, 75, 100, and 113, final adapter, training report,
logs, and both evaluation rollouts are copied locally. The local evidence
watcher copies stable checkpoints during training. All implementation changes
are made and checked locally before upload, as requested by the user.
