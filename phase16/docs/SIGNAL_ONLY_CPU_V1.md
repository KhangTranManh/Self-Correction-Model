# P16-S0: CPU-defined signal-only pilot (2026-11-05)

Status: **definition and synthetic scorer tests only**. This is a separate,
smaller development branch; it does not reinterpret the Phase 15 outputs,
replace the 40-episode Phase 16 gate in `PROTOCOL.md`, or claim a model result.
No source pool, GPU generation, LoRA update, or hidden-state extraction is
authorized by this document alone. The exact source manifest, prompts,
verifier, model revision, decoding, seeds, and episode assignment must be
locked before the first study-model output. A failed capacity gate is reported,
not repaired by selecting tasks after observing model answers.

## Two propositions and claim vocabulary

**S1, autonomous repair:** Given a task and its own earlier answer, but no
correctness label, oracle, test result, or gold answer, the model improves on
the same task more than a matched *blind independent retry*. This is a
behavioral contrast, not a claim that the model knows the unique answer.
Separately, **S1-guided** gives a genuine verifier counterexample restricted
to that task. A guided improvement cannot be called autonomous repair.

**S2, rule application:** After one verified correction on an A0-natural
failure, the model improves full correctness on new, source- and
scaffold-disjoint tasks where the named rule applies, without a repeated
instruction to apply the rule. A1 carries the correction in context; A2
receives it through a temporary LoRA update and sees no correction in its
probe prompt. Thus A1 can support *in-context application*, while A2 can
support *temporary weight-mediated application*. Neither is evidence of
permanent learning. A superiority claim for A2 over A1 needs a separate
paired A2-vs-A1 contrast; A2 merely passing its own gate is not enough.

The named error types remain `C-upper-inclusive` and
`M-negate-difference`, with the exact oracles and exclusions in
`PROTOCOL.md`. A generic failure, missing final answer, or only cleaner
format is not a named-error repair. `W-functional-counterexample` in
`DIVERSE_CODE_D0.md` is a broad failure label, not by itself a transferable
rule. No math and code scores are pooled.

## Capacity and assignment (per feasible domain/error type)

At most **300 unique source tasks**: 160 frozen-order A0 screening tasks,
40 rule-applicable probes (two per episode), 20 same-domain near-negative
tasks where the rule does *not* apply, 20 unrelated controls, and 60 sealed
follow-up reserves. From the 160-screen prefix, require at least 20 natural
named-error failures (at least 12.5%) and 20 initially correct tasks. The
first 20 of each in the frozen order are used for the 20 episodes; no
resampling or post-hoc substitution. Teaching failures and correct H1 checks
remain inside the 160-screen allocation. All roles are source-disjoint;
probes, near-negatives, and controls are assigned before their A0 outputs
are viewed. Split by source task and template/contest family; exact and
fixed near-duplicate checks exclude Phase 1-15 reuse. The 60 reserves stay
sealed during this pilot and are not automatically opened if a gate passes.

If there are fewer than 20 distinct failures, fewer than 20 correct checks,
or insufficient validated probes/controls, report **infeasible**. In
particular, the CodeContests metadata candidates and generated math Q/A
inventory are not yet verified S0 pools (`MULTISOURCE_F0_RECHECK.md`).

## S1 paired arms on the 20 wrong and 20 correct screen tasks

- `blind`: fresh conversation, original task only; independent second solve.
- `visible`: original task, own A0 answer marked *unverified*, neutral request
  to inspect and give a final answer; no correctness hint.
- `guided`: on the 20 wrong tasks only, same visible answer plus one genuine
  verifier counterexample, with no reference solution or probe information.

Use matched model, item seed, output budget and verifier. Save raw text/code.
S1-autonomous **follow-up signal** requires, on the 20 wrong tasks, at least
4 visible-only wins, a visible-minus-blind net gain of at least 3 tasks, and
wins from at least 3 episodes. On the 20 initially correct tasks, visible
may cause at most 2 more paired harms than blind. S1-guided is reported
separately: it needs at least 4 guided-only wins and net gain at least 3
against **both** blind and visible. These are deliberately small *triage*
thresholds, not hypothesis tests or evidence of reliable autonomous repair.
If there are repairs but the paired gate fails, report them as isolated
repairs or independent-retry ability, not S1.

The first three wrong-task conditions alone are also reported as an
`autonomous_behavioral_hint`. If correct-task preservation fails, this is a
*harmful or unstable hint*, not a follow-up signal. The harms are never
hidden by the hint label.

## S2 matched arms on all probes and controls

- `a0`: unchanged model, no teaching material.
- `a1`: the verified teaching correction in context.
- `a1_sham`: a length-matched verified correction of another rule/family in
  context, with no probe answer leakage.
- `a2`: one temporary LoRA update from the teaching correction; reset and
  hash-check after the episode.
- `a2_sham`: same update recipe using a length-matched unrelated correction;
  independent reset and hash-check.

All five arms answer identical task IDs with matched seeds, scaffold and
output budget. Teaching material contains only the teaching task, its natural
A0 failure, task-local verifier evidence, and verified repair. All five arms
are scored by **full verified task correctness**; named-error frequency and
format-only or rescued-answer correctness are diagnostics. The sham arms
exclude a generic “any correction/update changes behavior” explanation but
do not prove a unique internal mechanism.

For each of A1 and A2 separately, an S2 **follow-up signal** needs on 40
applicable probes: at least 4 treatment-only wins and net gain at least 3
versus A0, at least 2 net versus its own sham, and wins across at least 3
episodes. On each set of 20 near-negatives and 20 unrelated controls, it may
turn at most 2 A0-correct tasks wrong. Report all paired fixes and harms,
even if a gate passes. The near-negative set tests *when not to apply* the
rule; unrelated controls test broader regression. A2-vs-A1 is descriptive
unless a later confirmation run freezes it as a primary contrast.

The positive-probe and sham conditions alone are reported as a
`behavioral_hint`. If either control harm limit fails, this is a potentially
interesting but **unsafe/unstable hint**, not a follow-up signal. This
separation preserves weak evidence without relabeling damage as success.

These criteria only authorize a **new, separately locked follow-up**. They
cannot promote an adapter, establish statistical significance, or imply
fully applied/self-correcting behavior. If one signal passes with some
control damage within the triage allowance, the conclusion must explicitly
state that damage. If a primary correctness gain is absent, a positive probe
or reduced named-error frequency cannot rescue the gate.

## Transformer readout: optional secondary diagnostic, not a gate

After behavior and source roles are frozen, hidden states may be extracted
at the last prompt token *before generating the probe answer*. The label is
whether the named rule is applicable (positive versus same-domain
near-negative), not whether the model's final answer happened to be right.
Restrict candidate layers to 14, 21, 28 based on previous phases; fit any
probe only on development episodes and test on held-out source/template
families with a surface-text and shuffled-label control. Do not choose a
layer/threshold by inspecting the held-out outcomes. At 20 episodes this is
exploratory and may be underpowered. A decodable signal is **representation
evidence only**, not proof that generation used it. Phase 3 already tested
probe/CAA activation steering for KEEP/REVISE and found bias shifts, so no
steering or causal-use claim is part of S0.

## CPU scorer and integrity

`../scripts/signal_gate.py` accepts verified Boolean correctness labels for
the fixed 20-episode structure and reports paired wins/losses, gates, and
limits. It rejects missing arms, duplicate source IDs, invalid role counts,
and non-Boolean labels. It never runs generated code or reads a gold answer
from a model completion. `../tests/test_signal_gate.py` uses synthetic cases
only. The scorer cannot establish source-family independence from Boolean
labels; that requires a separately audited split manifest. Real verdict
production still requires a separately audited oracle.
