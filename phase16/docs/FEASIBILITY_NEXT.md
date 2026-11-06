# Phase 16: next gate — source audit before A0 screening

Defined 2026-11-03, after the P16-E1 engineering smoke and before any
scientific Phase 16 source pool or screening generation. This document
specifies the next work item; it does not declare a dataset frozen or
authorize interpreting the nine engineering outputs as study data.

The later [TACO source amendment](TACO_SOURCE_AMENDMENT.md) specifies the
primary candidate inventory and F0 provenance/overlap checks. The C-specific
capacity gate and conditional F1 screen below remain unchanged.

## Why this is next

P16-E1 verified the Colab model/verifier pipeline, but Qwen solved all nine
synthetic tasks. It therefore supplied no natural C-upper-inclusive teaching
failure and says nothing reliable about prevalence. The next uncertainty is
whether enough independent, unambiguous tasks and naturally occurring A0
errors exist. Do not train an adapter or run A1/A2 until that is answered.

## P16-F0: read-only source audit (no GPU)

1. Inventory candidate sources and licences, stable source IDs, original
   statements, reference solutions and test provenance. Inspect Phase 1–15
   manifests for exact source-ID reuse, prompt reuse, and protected/held-out
   sources. Phase 3's APPS/MBPP *planning* scan is an inventory lead, not an
   automatically eligible Phase 16 pool.
2. Retain only tasks with an explicit inclusive upper-bound contract, a
   specified Python function interface, and an independently checkable
   oracle. Generic code tasks or tasks whose boundary meaning must be
   inferred are ineligible. Record exclusions by reason.
3. Normalize whitespace/case for exact prompt hashes; additionally flag
   near duplicates with token 5-gram Jaccard similarity >= 0.80 and shared
   task/scaffold family. A reviewer resolves flags **before** A0 outputs are
   viewed; all variants of one source task remain in one partition. Record
   both raw and normalized hashes and reviewer decisions.
4. Build and manually inspect two distinct upper-boundary metamorphic
   pairs, lower/equal/ordinary/outside cases, and an oracle for each retained
   task. Do not expose tests or expected answers to the model. Apply the
   existing sandbox preflight to the verifier; reject ambiguous or
   oracle-disputed tasks before screening.
5. Report counts by source, family and count/sum/select variant, plus all
   exclusions. Freeze immutable task and split manifests with SHA-256, the
   prompt template, test generator, model settings, seeds, and scoring code
   before the first screening model output.

The capacity gate for the *proposed* C-upper-inclusive screen is at least
400 eligible teaching-screen tasks, 80 disjoint applicable probes, 40
unrelated controls, and a separately reserved confirmation family with at
least 80 eligible tasks. The confirmation reserve is not opened in the
development pilot; its final sample size and endpoint will be locked only
after the pilot, per `PROTOCOL.md`. Partition by source/scaffold family,
not by a random row split. If the source audit cannot meet the capacity and
separation requirements, report that limitation and do not spend GPU on a
screen assembled by recycling held-out or near-duplicate tasks. Close this
narrow branch and, if useful, write a separate broader-code or math protocol
before collecting new model evidence, as specified in the TACO amendment.

## P16-F1: conditional A0 screen (GPU only after F0 passes)

Run one greedy A0 completion per frozen teaching-screen task on the same
Qwen/Qwen2.5-7B-Instruct configuration used in P16-E1. Use a frozen item
order and seed, append-only per-item audit, the isolated code runner, and
the C-upper-inclusive functional label. Save raw prompts, completions,
execution transcripts, scorer labels and hashes. Monitor stage and completed
item counts without changing selection or prompts mid-run. Infrastructure
failure leaves the screen incomplete, not a negative result.

After **all 400** items have been scored, calculate the exact signature
count / 400, full-correct count, syntax/runtime/timeout counts and
`other_failure` count. Audit a prespecified systematic sample separately in
each label stratum: sort by frozen task ID, select positions 1, 11, 21, ...;
if that selects fewer than 10, add the lowest unselected IDs until 10 are
selected (or audit the entire stratum if it has fewer than 10). Also audit
every scorer/oracle disagreement.
If any label error is found, correct the verifier under a dated amendment
and rescore all saved outputs in the sandbox; do not silently relabel a
subset. The 40-episode pilot becomes feasible only if at least 40 distinct
natural signature-positive teaching tasks remain after audit (equivalent
to >=10% of this fixed 400-task screen), and the unscreened probe/control
capacity still satisfies `PROTOCOL.md`. Otherwise stop C-upper-inclusive
development and report the observed failure taxonomy. This screen is a
development feasibility decision, not a population estimate or treatment
effect.

An initial TACO train metadata/exact-overlap audit is now recorded in
[`TACO_F0_AUDIT.md`](TACO_F0_AUDIT.md). The error-specific source review,
frozen study pool and F1 GPU screen have not occurred.
