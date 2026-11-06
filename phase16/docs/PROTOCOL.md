# Phase 16 protocol v0 — substantive error types

Recorded 2026-11-03 before creating a Phase 16 source pool or generating any
Phase 16 model output. This is a development design, not a frozen main-run
preregistration. Amendments must be dated here and in `../RUN_STATUS.md` before
the work they affect. Phase 15 results are historical motivation, not Phase 16
observations.

An explicitly separate nine-task A0 engineering smoke is frozen in
[`ENGINEERING_A0.md`](ENGINEERING_A0.md). Its synthetic tasks are excluded
from all future Phase 16 study pools, so its model outputs cannot be used to
decide the protocol's feasibility gate.

Source discovery was later specified in
[`TACO_SOURCE_AMENDMENT.md`](TACO_SOURCE_AMENDMENT.md): TACO train is a
candidate inventory, not an automatically eligible or frozen study pool.

The separate, smaller **signal-only** development definition was recorded on
2026-11-05 in [`SIGNAL_ONLY_CPU_V1.md`](SIGNAL_ONLY_CPU_V1.md). It does not
retroactively alter this 40-episode protocol or open any source pool.

## Question and learner

Can a single verified correction reduce a specific content error on unseen
tasks, without making unrelated tasks worse? Compare no correction (A0), the
same correction placed in context (A1), and one temporary direct LoRA update
from that correction (A2). The initial learner is
`Qwen/Qwen2.5-7B-Instruct`, the model used in the Phase 15 pilot. A later
model, update method, or self-edit arm requires its own recorded amendment.

Unit of assignment and resampling is an episode. Each episode contains one
teaching task on which A0 naturally exhibits the named error, two new probes
where the rule applies, and one unrelated control. A0, A1 and A2 answer the
same probes and control under the same prompt scaffold, decoding settings,
per-item seed and output budget. A1's only added information is the teaching
correction. A2 receives the same teaching correction through a temporary
weight update; its test prompts contain no correction. Each episode starts
from the same base model and adapter initialization, and the reset is checked
afterward.

## First candidate: C-upper-inclusive (Python)

**Contract.** A task explicitly specifies an inclusive interval `[L, R]`
with `L <= R`; its target program selects, counts, or sums elements whose key
obeys `L <= key <= R`. The problem statement supplies the key, data type,
function signature and return type. Reference implementations and tests are
fixed before screening. A task with ambiguous boundary wording is rejected
before any model output is inspected.

**Observable error signature.** A submitted Python function must parse and
run within the fixed resource limit. It must pass the ordinary in-range and
below/above-range tests, yet fail a paired metamorphic boundary check: adding
one eligible element whose key equals `R` changes the oracle result as
specified but does not change the candidate result accordingly. For count
tasks the expected change is +1; for sum tasks it is the added value; for
selection tasks it is inclusion of that uniquely identified element. Sum
tests must use a nonzero added value. Require at least two distinct upper
boundary cases before assigning the signature. An optional syntax/AST
inspection can describe a `< R` mistake, but the functional signature is the
label. Any other failing pattern is recorded as `other_failure`; it cannot be
called C-upper-inclusive merely because a hidden test failed. Manually audit
a fixed sample of signature-positive outputs before relying on the label.
The exact mechanical rule and a worked fixture are in
[`C_UPPER_INCLUSIVE.md`](C_UPPER_INCLUSIVE.md).

**Correctness.** The program must pass the complete frozen unit-test suite,
including lower and upper boundary, `L == R`, no qualifying elements,
duplicate boundary keys, and ordinary interior cases. Record runtime and
syntax failures separately. Absence of the upper-boundary signature alone is
not a pass. Do not expose probe/control tests, oracle code, or expected
outputs to any arm. Execute model-written code in a networkless process with
time and memory limits and a fresh work directory per completion.

**Transfer.** The source pool will contain distinct task statements and
implementation scaffolds, with count, sum and selection variants. Teaching
and probes cannot be the same function with only constants or variable names
changed. At least one probe in each episode must use a different variant from
the teaching task. Human-reviewed specifications and generated oracle tests
are allowed; wrong model answers must arise naturally from A0, never from
injected faulty code. Keep source-task identifiers and text-similarity checks
in the manifest. Reserve an independent source family for later confirmation
before inspecting pilot outcomes.

## Separate candidate: M-negate-difference (symbolic math)

Given an expression containing `-(a-b)` with symbolic or numeric `a,b`, the
task asks for its equivalent expanded expression and final evaluated value
where applicable. The target error is a reported intermediate expansion
equivalent to `-a-b` instead of `-a+b`. SymPy checks both the submitted
intermediate expression and the final result against the oracle; require
values that distinguish the two formulas. A wrong final number without an
identifiable expansion is `other_failure`. Ambiguous parsing is `unscorable`.
Teaching and probes use disjoint expressions, coefficients and surrounding
operations. This candidate gets its own screen, episodes and report; it is
never pooled with code scores. It advances only if it has enough natural A0
errors and a reliable expression parser. It is a fallback, not an assumed
successful error type.

## Source preparation and feasibility screen

1. Before model screening, document provenance/licence, task schema,
   reference implementation, public contract, deterministic tests, source
   family and stable ID. Deduplicate against Phases 1–15 and within Phase 16
   by exact text and a fixed near-duplicate rule. Record manifest hashes.
2. Partition by source task and scaffold family into a teaching-screen pool,
   an unscreened development probe/control pool, and a sealed confirmation
   pool. No teaching item, probe or control may share a source task.
   Confirmation tasks and their oracle outputs remain inaccessible during
   pilot selection and tuning.
3. Run A0 only on the teaching-screen pool with frozen decoding and execution limits.
   Report prevalence of the exact error signature over all applicable tasks,
   plus syntax, runtime, other-failure and correct rates. Audit a preselected
   sample of positive and negative labels for verifier mistakes.
4. A candidate is feasible for the 40-episode pilot only if the observed
   signature rate is at least 10% and there are at least 40 distinct natural
   teaching failures, 80 disjoint applicable probes, and 40 disjoint unrelated
   controls after deduplication. Probe and control tasks are assigned from
   their unscreened pool before any of their A0 outcomes are viewed; they are
   not selected because A0 happened to fail.
   If these conditions fail, report infeasibility and screen the other
   candidate without loosening the threshold after seeing scores.

The source pool, prompts, parser, execution limits, test cases, split rule,
screening seed and item-assignment algorithm must be implemented and frozen
with hashes before the first Phase 16 model generation. No source pool is
declared frozen by this document alone.

## 40-episode development pilot, conditional on feasibility

Use 40 disjoint episodes per feasible error type: one correction, two probes,
one control each. This yields 80 probe and 40 control responses per arm.
The correction contains the teaching task, A0's failed response, verifier
feedback restricted to that task, and a verified repaired response. It must
not contain any probe/control text, test result, or answer. Check text and
code overlap before generation. Keep the correction content equivalent
across A1 and A2; record token counts and training settings.

The pilot's primary outcome is full verified task correctness on probes.
The named error-signature rate is diagnostic, not a substitute for task
correctness. The safety outcome is full verified correctness on controls.
Report absolute counts for all arms, A1−A0 and A2−A0 paired fixes/harms,
per-episode differences, syntax/runtime rates, and the named error rate.
Audit every output and rescore locally from saved text/code. An arm that
removes the named signature but increases other failures has not succeeded.

The following are development resource gates, not significance claims:

- The error type is useful for further study only if A1 gains at least 4/80
  full-correct probe items over A0, has more paired fixes than harms, and
  loses no more than 1/40 full-correct controls.
- A2 merits a separate, locked confirmation run only if it independently
  meets the same three conditions. Compare A2 with A1 descriptively in the
  pilot. Do not try more adapter methods merely because the named error rate
  improved while full correctness or controls worsened.
- A partial pilot, failed adapter reset, verifier disagreement, leakage, or
  missing audit is inconclusive. Correct the infrastructure under a dated
  amendment before restarting; never splice partial and retry episodes.

If a gate passes, define the confirmation pool size and statistical endpoint
from observed episode-level variability, then freeze a new protocol before
opening that pool once. Confirmation must assess both probe gain and control
damage with episode-clustered uncertainty. No general-learning claim follows
from the 40-episode pilot alone.

## Current state

The C-upper-inclusive scorer and synthetic fixtures passed a Colab CPU smoke
([record](COLAB_DEFINITION_SMOKE.md)). An isolated code runner was validated,
and the separate nine-task A0 engineering GPU smoke completed 9/9 correct
([record](ENGINEERING_A0.md)). A train-only TACO metadata/exact-overlap audit
has begun ([record](TACO_F0_AUDIT.md)), but the actual error-specific study
source pool, its split manifest, a feasibility screen, math parser and
confirmation sample size are not yet built or chosen. Phase 16 has no pilot
or confirmation result.
