# AGI Behavior-First Research

This repository studies whether a small language model can detect and correct
its own errors under objective supervision. Math answers are checked
symbolically and code answers are executed against tests; an LLM is never used
as the correctness oracle.

## Current status (2026-10-02)

Phase 11 closed on 2026-10-02 with a negative result. DPO on contrasting
judgments of the same two solutions (542 pairs) taught the model to prefer
correct judgments in likelihood (67% of unseen validation pairs) but barely
changed which solution it picks on fresh problems (46.6% → 49.4%, not
significant), and the judge strongly favors whichever solution is shown
second. Voting over five attempts gained 12.8 points (71.4% → 84.2%). See the
[Phase 11 final report](phase11/docs/FINAL_REPORT.md).

Phase 10 closed on 2026-10-01 with a negative result. A LoRA judge trained on
the model's own verifier-correct judgments of its conflicting solutions (808
balanced examples) did not improve its choice between two solutions (52.1% →
50.7% when exactly one is right) and was 6 points worse than agreement-gated
voting at equal compute. Voting over five attempts again gained 9 points
(73.67% → 82.67%). See the [Phase 10 final report](phase10/docs/FINAL_REPORT.md).
All checkpoints can be served with [serving/](serving/README.md).

Phase 9 closed on 2026-10-01. On 400 fresh protected GSM8K sources, a
plurality vote over five independent attempts raised accuracy from 75.0% to
83.25–85.5% for all three checkpoints (Holm p < 0.001). Letting the model
judge two conflicting solutions itself did not help (72.25–74.0%) and was
significantly worse than voting. Disagreement between two independent
attempts caught 82–85% of wrong first answers, better than the Phase 5 probe,
but when exactly one of two solutions was right the model chose it only
44–52% of the time. The model can tell *that* it may be wrong, not *which*
answer is right. See the [Phase 9 final report](phase9/docs/FINAL_REPORT.md)
and [case report](phase9/docs/CASE_REPORT.md).

Phase 8 completed on 2026-09-30. On 400 fresh protected GSM8K sources, the
original solver answered 70.75% correctly on its first attempt. Routing
probe-flagged answers to a blind re-solve raised accuracy to 74.5% (original),
75.0% (V2), and 77.5% (V3); all three gains over KEEP-all pass the
preregistered Holm-corrected gate. However, blind re-solving every answer is
at least as accurate, and a plain greedy second attempt alone reaches 76.75%,
so the gain comes mainly from an answer-hidden second attempt, not from model
self-correction. Showing any candidate answer sharply reduces repair; an
irrelevant wrong answer explains most of that drop, and the extra penalty for
the model's own answer is not significant after correction. No checkpoint is
promoted. See the [Phase 8 final report](phase8/docs/FINAL_REPORT.md).

Phase 4 closed on 2026-09-15 with no reliable autonomous correction gain.
Warm-start V2's historical 74% development result did not reproduce (73%).
GRPO, expanded correction SFT and DPO did not establish a reliable gain.
Blind preferences completed 16,064 reviews but failed the preset training
coverage gate. Three-round autonomous review reduced mean accuracy from
51.25% after round 1 to 49.17% after round 3 on a balanced 80-source diagnostic.

V2 is retained as a Phase 4 pilot reference, not a promoted replacement for
the canonical Phase 3 router. No compliant sealed confirmation was run;
confirmation candidates remain unopened. Phase 3's central problem is unresolved.
See the [Phase 4 final report](phase4/docs/FINAL_REPORT.md),
[results](phase4/docs/RESULTS.md) and [code ownership](phase4/docs/CODEBASE.md).
New research must open a separate phase with new hypotheses and holdouts.

Phase 5 is complete with no checkpoint promotion. It evaluated 480 balanced,
source-disjoint natural answers under a frozen neutral/status protocol. On the
160-source protected set, V2 neutral review produced 0 fixes and 2 harms; V3
produced 4 fixes and 32 harms. Status feedback produced 5 V2 fixes and 7 V3
fixes with no observed harms, but this is assisted repair because correctness
was disclosed. Frozen pre-hint probes generalized: protected balanced accuracy
was 71.25% for the original solver and V2, and 74.38% for V3. Thus correctness
information exists internally, but the generative neutral-review policy does
not use it safely. The exact frozen probe models were later recovered under
`outputs/phase5_gpu_vllm/probe_v1/selection/` with matching SHA-256 and were
used by Phase 8; see also the
[probe recovery audit](phase5/data/probe_recovery_audit.json).

This distinction is important: the current probe is an external diagnostic
harness that reads frozen hidden states. It demonstrates that a correctness
signal is accessible, but it does not make the checkpoint itself an autonomous
self-corrector. A deployed model-plus-probe router could form an externally
controlled correction system; the standalone model has not yet demonstrated
reliable detect-decide-repair behavior without correctness feedback.
See the [Phase 5 overview](phase5/README.md),
[execution plan](phase5/docs/EXECUTION_PLAN.md), and
[final report](phase5/docs/FINAL_REPORT.md).

Phase 6 closed as a non-protected signal-readout and harness diagnostic. On
16 development sources, verbal confidence was positively rank-correlated with
the frozen probe for all three checkpoints (`ρ=0.58–0.81`) but was severely
overconfident and less predictive. Step-by-step neutral review did not improve
wrong-to-correct repair; V3 improved preservation only. This is exploratory
evidence and does not promote a model or reopen the Phase 5 protected set. See
the [Phase 6 final report](phase6/docs/FINAL_REPORT.md) and
[detailed results](phase6/docs/RESULTS.md).

Two Phase 6 follow-ups reached the same non-promotion outcome. A disjoint
8-source fuzzy-hint curve found no useful threshold from 50% through 90%:
Original/V2 ignored the hints, while V3 at 90% fixed no wrong answer and broke
two correct answers. A frozen layer audit found that probe ROC-AUC falls by an
average 4.0 points from layer 14 to final hidden layer 28 across all three
checkpoints. The final layer remains well above chance, so this supports
partial signal attenuation rather than complete representational loss.

CPU-only calibration of the stored verbal confidence scores also failed as a
sufficient remedy. Temperature scaling identified severe overconfidence but
its leave-one-out Brier scores (0.265--0.282) were worse than a constant 50%
baseline (0.250); isotonic regression overfit the 16-row samples more strongly.
This result concerns verbalized confidence, because vocabulary-token logits
were not retained by the earlier vLLM run.

A final decomposition confirms a second, independent repair bottleneck. Even
when status identifies all wrong protected answers, V2 repairs only 5/77 valid
REVISE attempts and V3 repairs 7/79. The raw frozen probe is a better detector
than verbal confidence, but among its development true-positive routes the
cached status repair succeeds 2/31 times for V2 and 5/32 for V3. A transparent
probe → router → repair harness remains a research option, not an autonomous
self-corrector or a deployable result until false-positive routed repairs are
generated and verified on a fresh non-protected holdout.

That fresh 40-source end-to-end confirmation is now complete. The holdout was
selected from a new 160-source pool and frozen before confidence, activation,
routing, or recheck generation. The frozen probe detects 65--75% of initial
errors across the three checkpoints, but preserves only 60--80% of initially
correct answers and leaves final accuracy at 50%. Verbal self-confidence is
unstable as a router (it selects 1/40, 40/40, and 0/40 rows for
original/V2/V3). Even when the verifier oracle identifies all 20 wrong
answers, repair is only 1/20 for the original solver and 2/20 for both V2 and
V3. This confirms repair as the binding bottleneck and keeps Phase 6
non-promoted. See [the fresh harness result](phase6/docs/RESULTS.md).

Phase 3 closed on 2026-09-13. Its central result is negative but informative:
preference tuning improved KEEP behavior and reduced harmful revisions, but it
did not improve detection of plausible wrong answers.

| Router | Balanced accuracy | KEEP recall | REVISE recall | Code accuracy | Unseen-template accuracy |
|---|---:|---:|---:|---:|---:|
| Decision-Only V1 | 65.0% | 81.0% | 49.0% | 62.0% | 58.0% |
| DPO Pilot V1 | 66.5% | 85.0% | 48.0% | 62.0% | 55.5% |
| DPO Semantic V2 | 65.5% | 85.0% | 46.0% | 59.0% | 53.5% |

The representation probe reinforces this conclusion. Decision-Only V1 reached
67.3% balanced accuracy on the frozen probe test, while DPO Semantic V2 reached
57.7%; probe REVISE recall fell from 53.8% to 38.5%.

A later frozen activation-steering diagnostic also failed to turn the decoded
probe direction into better semantic decisions. Probe and CAA residual
directions were tested at layers 14, 21, and 28 without updating weights. The
only apparent accuracy gain shifted predictions toward KEEP while reducing
REVISE recall, so activation steering is not promoted.

Therefore:

- `Decision-Only V1` remains the canonical Phase 3 router.
- `Kxck/Self_Correction_v1` remains the solver/repair checkpoint.
- DPO Pilot V1 and DPO Semantic V2 are retained as research artifacts, not
  promoted models.
- No additional experiment is authorized inside Phase 3. Any continuation must
  open a new phase with new holdouts and a preregistered hypothesis.

See the [Phase 3 final report](phase3/docs/FINAL_REPORT.md),
[Phase 3 README](phase3/README.md), and
[Phase 3 results](phase3/docs/RESULTS.md).

## Whole-project summary

| Phase | Core question | Durable finding | Final disposition |
|---|---|---|---|
| 1 | Can verified SFT teach critique and correction? | Guided correction is real, especially for code, but SFT mainly teaches correction format; autonomous math correction remains weak. | Closed; provides the historical solver foundation. |
| 2 | Can verifier-labeled preferences improve error discrimination? | The KTO data/training path was validated, but no completed KTO adapter was produced. | Historical pipeline only. |
| 3 | Can a router decide KEEP versus REVISE for plausible errors? | Decision-Only V1 reaches 65.0% balanced accuracy but only 49.0% REVISE recall; follow-ups trade error detection for excessive KEEP. | Closed; V1 retained only as the router baseline. |
| 4 | Can on-policy/selective-correction training improve repair? | Warm-start, GRPO, expanded SFT, DPO, blind preferences, and repeated review did not establish reliable autonomous gain. | Closed; V2/V3 are archived pilots. |
| 5 | Does a correctness signal exist before feedback? | Frozen probes generalize on protected data, but neutral generation does not use the signal safely; status gives limited assisted repair. | Closed diagnostic; probes are external readouts only. |
| 6 | Can readout, calibration, or a harness bridge signal to repair? | Fresh end-to-end routing confirms detection exists but repair remains only 5--10% even with oracle-known-wrong status. | Closed; no router, threshold, calibration, harness, or checkpoint promoted. |
| 7 | Is weak repair caused partly by seeing the earlier answer? | On 80 initially wrong protected sources per checkpoint, blind re-solving fixed 25/28/32 versus 4/5/8 when the prior answer was visible. | Completed diagnostic; answer visibility has a strong paired effect, but no autonomous router or model was promoted. |
| 8 | Is the Phase 7 effect resampling, generic distraction, or self-anchoring, and does probe routing help? | A second attempt alone gains ~6 points; any visible candidate hurts, mostly as generic distraction; probe → blind re-solve beats KEEP-all (+3.75 to +6.75 points) but not BLIND-all. | Completed; the external harness gain is supported, no checkpoint or autonomous self-corrector promoted. |
| 9 | Can the model check itself with its own independent attempts? | Voting over five attempts gains 8–10.5 points; disagreement is a strong error signal; the model's own judgment between two solutions is at chance. | Closed; voting is the best supported method, self-check failed, nothing promoted. |
| 10 | Can fine-tuning on its own correct judgments teach the model to pick the right solution? | No: the trained judge picks the right one at chance (51%) and trails voting; voting again gains 9 points. | Closed; negative result, judge adapter not promoted. |
| 11 | Can DPO on correct vs incorrect judgments of the same pair teach the choice? | DPO shifts likelihoods (67% validation preference) but not greedy choices on fresh problems (+2.8 points, n.s.); the judge has a strong order bias; voting gains 12.8 points. | Closed; negative result, DPO adapter not promoted. |

Across the completed phases, the evidence supports one narrow claim: correctness
information can be verifier-labeled and, in later checkpoints, decoded from
hidden states, and an external harness that re-solves flagged answers without
showing the old answer improves accuracy. It does **not** support the stronger
claim that the standalone model reliably recognizes, preserves, and repairs its
own errors: repair works best when the model never sees its earlier answer.

Phases 9 and 10 sharpen the distinction between **harness** and **model**.
Every accuracy gain so far — probe routing, blind re-solving, and voting over
five attempts (+8 to +10.5 points, replicated on three fresh sets) — comes from
a system wrapped around an unchanged model. At the model level, the checkpoint
notices that its own attempts disagree (82–85% of errors) but cannot reliably
tell which attempt is right (about 50%), and fine-tuning on its own correct
judgments (Phase 10) or DPO on contrasting judgments (Phase 11) did not
change that. The core Phase 0 goal, a model that recognizes,
explains, and repairs its own errors, is not yet achieved.
Any next step must start a new phase with a newly frozen evaluation set.

Phase 7 tested whether poor repair reflects answer anchoring. It compared a
fresh solution that never sees the prior answer with a paired fresh solution
that does see it, on new source-disjoint problems. The run collected 334
natural initial answers, froze a balanced development/protected split, and
completed the paired protected evaluation. Blind re-solving fixed substantially
more initially wrong answers under all three checkpoints, but the frozen probe
needed for non-oracle routing was unavailable. See the
[Phase 7 final report](phase7/docs/FINAL_REPORT.md).

Phase 8 then separated resampling, generic distraction, and self-anchoring,
and tested a non-oracle probe → blind re-solve pipeline on 400 fresh protected
sources. See the [Phase 8 final report](phase8/docs/FINAL_REPORT.md).

Canonical closures: [Phase 3](phase3/docs/FINAL_REPORT.md),
[Phase 4](phase4/docs/FINAL_REPORT.md), [Phase 5](phase5/docs/FINAL_REPORT.md),
[Phase 6](phase6/docs/FINAL_REPORT.md), [Phase 7](phase7/docs/FINAL_REPORT.md),
[Phase 8](phase8/docs/FINAL_REPORT.md), [Phase 9](phase9/docs/FINAL_REPORT.md),
[Phase 10](phase10/docs/FINAL_REPORT.md), and [Phase 11](phase11/docs/FINAL_REPORT.md).

## Next steps (proposed Phase 12)

A post-hoc analysis of Phase 11 (exploratory, section 5b of its report) shows
the judge's failure has three parts: a strong order bias toward the second
solution shown, a weak real skill (about 59% correct when it picks one of the
two solutions), and invented third answers in 15–19% of cases. Phase 12 will
test whether removing the fixable parts reveals a usable judgment:

1. **New evaluation dataset.** No unused GSM8K train source remains; freeze a
   fresh holdout from another verifiable numeric dataset (for example SVAMP,
   ASDiv, or MATH levels 1–3).
2. **Order-swap judging.** Judge every pair in both orders (A-B and B-A) and
   trust the verdict only when both orders agree; otherwise fall back to
   voting.
3. **Order-swapped DPO.** Retrain on the Phase 11 preference pairs in both
   orders (about 1,084 pairs), with invented third answers among the rejected
   judgments.
4. **Gates (preregistered):** P1 the both-orders judge picks the right
   solution significantly above chance; P2 the trained both-orders self-check
   beats equal-compute voting. Voting over five attempts remains the baseline.

Preparation runs locally at no GPU cost; the GPU run is about 3 hours. If P1
passes but P2 fails, the next lever is step-level verification or a larger
model; if neither passes, the evidence favors a larger model or step-level
supervision over further final-answer training at 7B.

## Research phases

### Phase 1 — verified self-correction

Phase 1 built the verified critique-and-correct pipeline and produced
`Kxck/Self_Correction_v1`. Guided correction worked, especially for code, but
autonomous review and resistance to false feedback remained weak. See
[phase1/README.md](phase1/README.md).

### Phase 2 — preference learning exploration

Phase 2 constructed KTO data from verifier-labelled behavior and validated the
training path. A full KTO checkpoint was not completed. This phase is historical
and is not the active training path. See [phase2/README.md](phase2/README.md).

### Phase 3 — error discrimination and selective repair

Phase 3 separated the decision `KEEP`/`REVISE` from answer repair, evaluated the
router on a frozen 200-row benchmark, and probed hidden representations. It is
now a closed, reproducible research package; Decision-Only V1 is its final
router baseline.

### Phase 4 — selective correction pilots (closed)

Exploration warm-starts, GRPO, expanded SFT, DPO, blind preferences and repeated
review did not establish a reliable autonomous correction gain. Evidence and
adapters are archived; no additional Phase 4 tuning is authorized.

### Phase 5 — guided repair and pre-hint diagnosis (complete)

The protected experiment is complete. Hidden-state probes establish a
generalizing pre-hint correctness signal, but no checkpoint achieves safe
autonomous neutral repair. V2 shows a limited assisted-repair gain under
explicit status feedback; no model is promoted.

### Phase 6 - signal readout and harness confirmation (closed)

The first small pilot compares explicit confidence with the existing frozen
probe and tests model-visible step-by-step neutral review. It finds shared but
poorly calibrated confidence/probe ranking and no autonomous repair gain.
Follow-up fuzzy-hint and layer-path diagnostics find no safe harness threshold
and a modest, replicated attenuation of probe signal after layer 14.
Calibration of verbal confidence corrects in-sample overconfidence but does not
generalize beyond a constant 50% baseline in leave-one-out evaluation.
The detector-to-repair decomposition finds that repair remains weak even under
oracle error disclosure. A fresh 40-source end-to-end confirmation then
measures probe false positives with the same non-oracle recheck prompt. Frozen
probes detect substantially more errors than verbal confidence, but repair
remains only 5--10% even under oracle-known-wrong status. This is evidence for
an external detector -> router -> repair -> verifier research harness, not an
autonomous self-corrector or deployable system. See
[the Phase 6 final report](phase6/docs/FINAL_REPORT.md).

### Phase 7 - blind re-solving and answer anchoring (completed diagnostic)

Phase 7 compared independent re-solving from the original problem with
re-solving that can see the earlier answer. Both arms used the same source and
checkpoint, and deterministic verification separated wrong-to-correct fixes
from correct-to-wrong harms. Blind re-solving fixed far more wrong answers for
every checkpoint; the probe-routed analysis was unavailable because the probe
artifacts were then missing. See
[the Phase 7 final report](phase7/docs/FINAL_REPORT.md).

### Phase 8 - resampling, distractor, and non-oracle routing (completed)

After the original RTX 3090 run was lost, Phase 8 was rerun in full on one
V100 under a logged amendment (batched generation, regenerated distractor
donors, exact recovered Phase 5 probes). A second attempt alone gains about
six points; visible candidates hurt mostly as generic distraction; and probe →
blind re-solve beats KEEP-all for all three checkpoints but not BLIND-all. See
[the Phase 8 final report](phase8/docs/FINAL_REPORT.md) and
[run record](phase8/RUN_STATUS.md).

### Phase 9 - independent attempts, voting, and self-check (closed)

Phase 9 compared keeping the first answer, voting over three or five
independent attempts, agreement-gated voting, and a model self-check that
sees two conflicting solutions. Voting over five attempts is the only large,
significant gain; the self-check writes plausible error explanations but picks
the right solution at chance. See
[the Phase 9 final report](phase9/docs/FINAL_REPORT.md).

### Phase 10 - training the judgment step (closed)

Phase 10 fine-tuned a LoRA judge on the original solver with its own
verifier-correct judgments of its own conflicting solutions, then tested it on
300 fresh holdout problems. The judgment did not improve and the trained
self-check trailed voting. See
[the Phase 10 final report](phase10/docs/FINAL_REPORT.md).

### Phase 11 - preference training of the judgment step (closed)

Phase 11 trained a DPO LoRA to prefer the correct over the incorrect judgment
of the same pair of conflicting solutions and tested it on the last 234
never-used GSM8K problems. The preference was learned in likelihood but did
not translate into better choices, and the judge's order bias remained. See
[the Phase 11 final report](phase11/docs/FINAL_REPORT.md).

## Non-negotiable data rules

1. Correctness comes from a deterministic verifier, not an LLM judge.
2. Wrong answers must be natural model failures; synthetic wrong answers are not
   accepted for canonical training.
3. Frozen evaluation sources never enter training or dataset selection.
4. KEEP and REVISE must share the same neutral-review template distribution.
5. Model-visible rationales may be stored for audit, but they must not be called
   hidden chain-of-thought.
6. Every promoted adapter must have a reproducible config, dataset hash, frozen
   evaluation, and local backup.

## Repository layout

```text
AGI/
├── README.md                    # Project status and phase routing
├── instructionAI/              # Cross-phase architecture and data rules
├── phase1/                     # Verified correction pipeline and history
├── phase2/                     # KTO exploration and data
├── phase3/                     # Closed router/selective-repair research
│   ├── configs/                # Training configs and experiment registry
│   ├── data/                   # Sources, immutable attempts, built datasets
│   ├── docs/                   # Codebase and results documentation
│   ├── lib/                    # Shared provenance and verifier utilities
│   ├── runs/                   # Frozen evaluations and human-readable reports
│   └── scripts/                # Data, training, evaluation, and serving CLIs
├── phase4/                     # Closed selective-correction pilots and evidence
├── phase5/                     # Complete guided-repair and pre-hint probe diagnostic
├── phase6/                     # Closed signal-readout and harness diagnostic
├── phase7/                     # Completed blind re-solving diagnostic
├── phase8/                     # Completed resampling/distractor/probe-routing study
├── phase9/                     # Closed voting and self-check study
├── phase10/                    # Closed judge-training study (negative)
├── phase11/                    # Closed DPO judge study (negative)
├── serving/                    # vLLM serving profiles and strategy client
└── outputs/                    # Local adapters and runtime artifacts (gitignored)
```

The canonical Phase 3 structure and ownership rules are defined in
[phase3/docs/CODEBASE.md](phase3/docs/CODEBASE.md). Model status is machine-readable
in [phase3/configs/experiments.yaml](phase3/configs/experiments.yaml).

## Quick local audit

From the repository root:

```bash
python phase3/scripts/validate_local_state.py
python phase3/scripts/validate_local_state.py --hash-adapters
```

Phase 4 closure audit (aligned verification environment):

```powershell
.venv-phase4-verify/Scripts/python.exe phase4/scripts/validate_local_state.py --hash-adapters
```

The audit checks canonical datasets, reports, row counts, adapter presence, and
optionally the recorded adapter hashes. It does not read `.env`.

## Serving

All solver checkpoints and the Phase 10/11 judge adapters are served by
[`serving/serve.sh`](serving/README.md) (profiles `original` and `v3`), and
`serving/client.py` runs the `single`, `vote`, and `self_check` strategies with
the exact experimental prompts. Phase 3 routers keep their own launcher,
`phase3/scripts/serving/serve_phase3.sh`. Both launchers bind
to localhost by default. Example:

```bash
PHASE3_LORA_MODULES="phase3-decision-only-v1=/root/agi/outputs/phase3_decision_only_v1/final_adapter" \
bash phase3/scripts/serving/serve_phase3.sh
```

Only set `PHASE3_HOST=0.0.0.0` on a network you intend to expose. The launcher
does not add authentication by itself.

## Secrets and large artifacts

- `.env` is local-only and must never be read for documentation work or committed.
- `outputs/` is gitignored; adapters must be backed up separately.
- Raw attempts and frozen result JSONL files are research evidence. Do not
  rewrite or delete them during documentation cleanup.
