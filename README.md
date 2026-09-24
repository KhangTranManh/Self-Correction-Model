# AGI Behavior-First Research

This repository studies whether a small language model can detect and correct
its own errors under objective supervision. Math answers are checked
symbolically and code answers are executed against tests; an LLM is never used
as the correctness oracle.

## Current status (2026-09-24)

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
not use it safely. Raw evidence is under `outputs/phase5_gpu_vllm/`.

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

Across all six phases, the evidence supports one narrow claim: correctness
information can be verifier-labeled and, in later checkpoints, decoded from
hidden states. It does **not** support the stronger claim that the standalone
model reliably recognizes, preserves, and repairs its own errors. Any next step
must treat detection and repair as separate capabilities, start a new phase,
and use a newly frozen evaluation set.

Canonical closures: [Phase 3](phase3/docs/FINAL_REPORT.md),
[Phase 4](phase4/docs/FINAL_REPORT.md), [Phase 5](phase5/docs/FINAL_REPORT.md),
and [Phase 6](phase6/docs/FINAL_REPORT.md).

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

The general vLLM launcher is `phase3/scripts/serving/serve_phase3.sh`. It binds
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
