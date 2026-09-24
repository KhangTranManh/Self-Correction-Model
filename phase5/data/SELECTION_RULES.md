# Phase 5 local source selection rules

This file defines the CPU-only source work before any GPU allocation.

The raw candidate pool is unused GSM8K **train** questions. Build a complete
normalized-question and source-ID exclusion index from every locally available
Phase 1–4 source, including protected and unopened confirmation manifests.
Never use GSM8K test or a source already selected for an earlier phase.

Use the original problem text and reference numerical answer. Deterministically
rank remaining source IDs by SHA-256 of `phase5_source_v1|<source_id>` and take
the first 1,200 as a CPU-prepared candidate pool. Store all 1,200 IDs, original
dataset revision, normalized-question hashes, exclusion-index hash and a frozen
selection script version. This is a maximum acquisition budget, not a promise
that 1,200 eligible records exist. If fewer remain, preserve the count and stop
without reusing an excluded source.

Pre-GPU CPU checks:

1. Parse each reference final answer with the existing math verifier, rejecting
   ambiguous or unparsable references by a written reason code.
2. Apply a fixed short-problem criterion: question at most 120 whitespace words;
   reference solution contains between two and six GSM8K `<<...>>` arithmetic
   annotations; each annotation must evaluate to its declared value with the
   pinned verifier/runtime. These reference annotations define available step
   structure; they are never shown to the reviewing model.
3. Deduplicate questions by a fixed normalized text hash and retain only
   records whose source IDs and hashes are absent from the exclusion index.
4. Publish eligible/rejected counts and a SHA-256 manifest before requesting GPU.

The first GPU job would generate **one** natural initial answer per eligible
source using the original solver, and verify it. It would not generate reviews.
Frozen correct/wrong buckets and source splits can then be made from those
natural answers. Initial correctness cannot be known from the reference alone;
therefore the 240/80/160 balanced split remains conditional until this cheap
one-answer pass. If a source lacks a reliably alignable erroneous calculation
step, it can enter the status-hint and probe analyses but not location/type
hint analyses. Exact alignment rules and coverage must be frozen locally before
that first GPU pass.

At the time these rules were written, no Phase 5 model data had been generated.
The candidate count was subsequently checked by the local inventory and
verifier audit in `CANDIDATE_REPORT.md`.

## Execution note (2026-09-23)

The CPU source audit selected 1,200 candidates, and a later GPU pass collected
815 natural answers (575 correct, 240 wrong). The statement above describes
the pre-execution state, not the current state. A model-error alignment rule
was not frozen before that GPU pass, despite the requirement above. Preserve
that deviation; location/type analyses on the current collection are
exploratory unless a new preregistered holdout resolves it. See
`../docs/EXECUTION_PLAN.md`.

On 2026-09-24, before any guided-review or probe outcome existed, the collected
pool was frozen into balanced, source-disjoint train/development/protected
splits under `splits/v1/`. The manifest records the deterministic selection and
assignment namespaces, seed, exact counts, and SHA-256 hashes.
