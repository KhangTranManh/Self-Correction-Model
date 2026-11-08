# P16-R0: recovery-first system and gated micro-update (2026-11-07)

Status: **prospective development amendment; CPU definition only**.

This amendment changes the order of future Phase 16 interventions. It does
not rewrite Phase 15, reopen the failed 300-task C-upper-inclusive pool, open
protected data, certify a source task, or authorize model generation, weight
updates, or GPU work. Existing Phase 16 error definitions and verifier gates
remain valid where referenced; this document controls which intervention is
attempted first after a task and oracle are valid.

## Motivation from Phase 15

P15-A4 preserved unchanged A0 generations and safely normalized an answer
already present in the text. Probe joint correctness rose from 31/80 to 45/80,
control joint correctness rose from 30/40 to 36/40, all 120 outputs became
format-clean, and no correctness harm was observed. P15-A5 then found only a
+4/80 rescued-answer probe difference for LoRA, accompanied by a -2/40
control difference. That evidence makes output recovery the default systems
intervention and weight update a residual, conditional intervention.

The Phase 16 question is therefore ordered:

1. What can a reference-free recovery layer fix without changing weights?
2. What additional errors can a bounded, stateful harness prevent?
3. Which failures remain substantive after both layers?
4. For those residual failures, after how many verified examples does a
   micro-update beat the harness stably without regression?

## Error decomposition before intervention

Every independently verified failure receives exactly one pre-intervention
class. The class is assigned from raw output and frozen verifier evidence,
never from another treatment arm.

- `presentation_only`: the correct semantic payload is present but violates a
  declared serialization, suffix, schema, delimiter, or tool-call contract.
- `recoverable_payload`: a deterministic, reference-free extractor can
  preserve and expose a checkable payload already present in the output.
- `substantive_content`: the recovered payload is still wrong under the
  independent oracle; this includes reasoning and algorithmic errors.
- `execution_or_scaffold`: syntax, runtime, resource, interface, or harness
  failure that is not evidence for a content update.
- `ambiguous_or_unscorable`: the source contract or output cannot support a
  unique verifier verdict.

Only `substantive_content` may contribute examples to a weight-update curve.
Format, extraction, scaffold, and ambiguous failures cannot be relabeled as
reasoning failures to increase update capacity.

## Intervention ladder

### R0: raw generation

Retain the complete raw completion and generation metadata. Raw generation is
a diagnostic baseline, not the deployed system endpoint.

### R1: deterministic output recovery

Apply a task-family-specific canonicalizer or extractor after generation.
It must be frozen before evaluation and must not consult a reference answer,
hidden test, another arm, or another item. It may append or serialize a payload
already found under an explicit rule, but may not invent, replace, or repair
semantic content. Preserve both raw and recovered outputs.

Recovery is admissible only after synthetic fixtures prove payload
preservation, idempotence, byte-for-byte noninterference on already-valid
outputs, deterministic rerun, and zero changes driven by gold values. Report
accepted, rejected, unchanged, fixed, harmed, and still-wrong counts.

### R2: constrained decoding

Use grammar, schema, token, or tool-call constraints only when the output
contract is exact and R1 cannot safely recover the remaining acceptance
failures. First test constraints on excluded engineering fixtures. A
constraint that changes semantic payload, creates a tokenizer dead end, or
harms an already-valid output is rejected. Postprocessing precedes hard token
blocking because P15-A4 met its format objective without generation-time
distribution shift.

The selected recovery wrapper, `R*`, is frozen and hashed before comparing
harness and update arms. Every later arm uses the identical `R*`, so a format
advantage cannot be credited to weights.

### H: bounded stateful harness

The harness accumulates only previously verified development corrections. A
memory record contains an error-family ID, applicability trigger,
non-applicability boundary, concise rule, permitted action, at least one
verified positive example, at least one near-negative example, provenance,
oracle hash, and record hash. Rules are versioned and immutable after a run
starts.

At inference, retrieval is limited to a frozen top-k and token budget. The
harness records which rule IDs were retrieved and the exact prompt patch. It
cannot read the current task's verdict, hidden tests, probe answer, or later
memory. Near-negative and unrelated controls measure rule over-application.
The complete memory may not simply be pasted into every prompt.

Rule drafting may use a model, but admission requires deterministic schema
checks, independent verifier evidence for examples, leakage checks, and human
approval. The model is never the correctness oracle.

### S: self-edit data generation

A model may propose a general rule, rewritten explanation, positive examples,
and near-negative examples after seeing a verified development correction.
The proposal is saved as raw data and is **not** applied as an update in that
episode.

An accepted self-edit must:

1. parse under a frozen schema;
2. contain no probe, control, sealed, or protected text or answer;
3. avoid source IDs and task-specific answer copying;
4. pass an independent oracle on every generated example;
5. include an applicability boundary and a verified near-negative;
6. pass exact and semantic-near-duplicate checks; and
7. retain raw, normalized, verdict, rejection reason, and hashes.

Accepted records enter a versioned candidate corpus. A human freezes a later
batch before either harness construction or training. Rejected self-edits
remain in the audit but are never silently normalized into training data.

### U: gated micro-update

A micro-update is allowed only for residual `substantive_content` failures
that remain after `R*` and the stateful harness. It trains on a frozen batch
of verified corrections and accepted self-edit records; it never trains on a
current probe, control, sealed task, Phase 5--14 protected source, or Phase 15
reserved final task. Adapter initialization, trainable modules, optimizer,
precision, steps, seeds, and reset checks must be frozen separately before
GPU use.

## Matched arms and endpoints

For each evaluation item, use matched prompts, inference seed, output budget,
and verifier. The primary arms are:

- `recovery`: unchanged model plus `R*`;
- `harness_n`: stateful harness built from the first `n` frozen correction
  records, plus `R*`;
- `update_n`: micro-update trained from the same first `n` records, no runtime
  correction context, plus `R*`;
- `harness_update_n`: optional secondary combination, defined only after the
  three primary arms are complete.

The primary outcome is full independent-verifier correctness after `R*`.
Report raw correctness, recovered correctness, recovery yield and harm,
named-error frequency, syntax/runtime status, latency, generated tokens,
retrieved-rule tokens, paired fixes and harms, and source-family coverage.
Cleaner format or lower named-error frequency cannot replace full correctness.

## Nested sample-size curve

Use hash-frozen, family-stratified nested prefixes of exactly
`n = 8, 16, 32, 64` verified correction records. Each larger level contains
the complete smaller prefix. The evaluation set is held fixed across levels:
80 applicable probes, 20 same-domain near-negatives, and 40 unrelated
controls, all source- and family-disjoint from correction records. Their
assignment and oracle hashes are frozen before any harness or update output.

Run three independent update-training seeds per level. Inference item seeds
are matched to the harness. The harness at one level is deterministic with
respect to the frozen memory and item seed. If the source cannot supply all
four correction levels and the held-out roles, report the threshold question
as infeasible; do not shrink the ladder after viewing outcomes.

For one level `n`, `update_n` passes the development gate only when:

1. at least two of three update seeds have paired probe net gain of at least
   4/80 over `harness_n`;
2. no update seed has negative paired probe net gain;
3. each passing seed has update-only wins in at least three source families;
4. every seed turns at most 1/20 harness-correct near-negative and at most
   1/40 harness-correct unrelated control wrong; and
5. reset, hash, audit, leakage, and deterministic local rescore checks pass.

The **stable update onset** `n*` is the smallest level for which both that
level and the next larger level pass. Passing only one level is an isolated
development hint. If only `n=64` passes, the onset is unresolved because no
larger frozen level confirms stability. These thresholds are resource gates,
not significance claims.

The CPU scorer `../scripts/recovery_first_gate.py` evaluates only independent
Boolean verifier labels for this curve. It never executes generated code,
infers correctness from text, or authorizes training.

## Stop and release rules

- If R1 safely closes the measured objective, stop for that error type; do
  not run constrained decoding or update merely to demonstrate them.
- If R2 is needed and safely closes it, stop before weight work.
- If the stateful harness removes the residual error without unacceptable
  near-negative or unrelated harm, deploy/recommend the harness and record
  update as unnecessary for that objective.
- If fewer than 64 verified residual correction records or the held-out roles
  exist, the stable-onset question is unanswered.
- A passed `n*` authorizes only a separately written micro-update execution
  protocol. It does not itself authorize GPU, protected opening, deployment,
  or a learning claim.
- Any verifier disagreement, leakage, reset failure, missing raw output, or
  unlogged prompt/memory change fails closed.

## Immediate CPU work

The executable schemas, cross-artifact release validator, and synthetic tests
are now implemented in `RECOVERY_FIRST_CPU_CONTRACT_V1.md`. Remaining CPU work
is empirical: inventory independently verifiable residual error families,
measure source capacity, and populate real artifacts that pass the aggregate
release gate. The existing KodCode recall audit may inform discovery but does
not certify the 64 corrections or held-out roles required here. No Phase 16
model output, GPU execution, or update is authorized by this amendment.
