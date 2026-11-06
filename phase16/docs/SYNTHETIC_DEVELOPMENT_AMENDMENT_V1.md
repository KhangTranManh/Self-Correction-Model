# P16-S0 synthetic-development pool amendment v1 (2026-11-05)

Status: **definition only; GPU gate closed**. This document admits pinned
KodCode-V1 candidates as a synthetic development distribution for P16-S0. It
does not alter Phase 15, replace the original 40-episode Phase 16 protocol, or
turn a text match into an eligible task. The current inventory contains 896
shortlisted rows collapsed into 627 connected seed families, but **zero tasks
are presently contract-, oracle-, history-, and sandbox-certified**.

## Claim boundary

This branch can discover a behavioral hint for autonomous repair or applying
the named `C-upper-inclusive` rule. Because KodCode includes synthetic and
source-derived material, a positive result is evidence on this development
distribution only. It cannot establish transfer to natural programming tasks.
Any broader claim requires a separately frozen confirmation set, preferably a
post-cutoff LiveCodeBench release or another provenance-audited natural source.
LeetCodeDataset may support discovery but is not an independent confirmation
source where its task families overlap KodCode/LeetCode-derived seeds.

## Candidate unit and provenance

The pinned KodCode-V1 revision is
`97d81f0695ddd7c5c02bfe16436db9a17a4a21e5`. The review inventory is
`../results/kodcode_family_review_manifest.json`; its 627 units are connected
components of `seed_ids` within a KodCode subset. A component, not a rewritten
prompt, is the indivisible family. At most one task from a component may occur
anywhere in the 300-task pool.

The subsets `Taco`, `Codeforces`, `Code_Contests`, `Leetcode`, and `Apps` are
marked source-derived. They are allowed for synthetic development only after
ID/text/family checks against Phases 1--15 and all Phase 16 engineering
fixtures. An unresolved origin, license, historical match, or seed relation is
a rejection, not an `unknown` value carried into the run. Dataset-provided
tests are supporting evidence and never the sole oracle.

## Exact 300-task manifest

The manifest has these fixed roles before any A0 model answer is observed:

| role | tasks | rule applies | count/sum/select |
|---|---:|---|---:|
| `screen` | 160 | yes | 54 / 53 / 53 |
| `positive_probe` | 40 | yes | 14 / 13 / 13 |
| `near_negative` | 20 | no | `none` |
| `unrelated_control` | 20 | no | `none` |
| `reserve` | 60 | yes | 20 / 20 / 20 |

Thus the pool has 260 rule-applicable tasks and 40 controls. The screen prefix
must still yield at least 20 natural model A0 failures with the named signature
and 20 initially correct tasks, as required by `SIGNAL_ONLY_CPU_V1.md`.
"Natural" here describes the model's unprompted A0 error; it does not claim
that the underlying KodCode task is naturally authored.

Applicable tasks must span at least five KodCode subsets, with no subset
contributing more than 60 of the 260. This cap limits a single generator/source
style without pretending that subset labels guarantee independence. All 300
family IDs, normalized prompt hashes, and task IDs must be unique. The 60
reserves remain sealed.

## Contract admission

An applicable task must have one explicit Python function signature, defined
input and return types, and exactly one checkable output. It must state an
inclusive interval `[L,R]` and require count, sum, or selection over elements
whose key satisfies `L <= key <= R`. Reject tasks whose interval convention is
implicit, whose expected output depends on unstated ordering/tie behavior, or
whose surrounding requirements prevent isolating the named rule.

Two universe modes are admitted:

- `explicit_collection`: the input supplies the elements/records to filter.
- `implicit_integer_range`: the task quantifies integers in a defined finite
  domain. Each upper-bound pair must choose an `R` that satisfies the task's
  predicate, so changing `[L,R-1]` to `[L,R]` adds the observable endpoint.

Near-negatives must be structurally matched but have a contract where
`C-upper-inclusive` does not apply (for example, an explicitly half-open
interval). Unrelated controls must exercise a different skill. Neither may be
relabeled after model output.

## Independent oracle and locked cases

Every task needs two independently written computations: a simple brute-force
oracle and a reference implementation. They must agree on at least ten frozen
cases covering ordinary inclusion, the lower boundary, outside values, empty
and singleton behavior, duplicates where legal, and at least two distinct
upper-bound pairs. The trusted reference must pass every case. For applicable
tasks, an exclusive-upper mutant must pass ordinary/lower/outside cases and
fail the two upper pairs, producing the exact `C-upper-inclusive` signature.
At least one unrelated mutant must also be caught, to show that the suite is
not merely a signature detector.

Generated code is executed only in the Phase 16 no-network, resource-limited
sandbox. A sandbox smoke and deterministic rerun are required. Oracle source,
reference source, frozen tests, and normalized prompt each receive SHA-256
hashes in the manifest. Reviewers must not use a study-model completion as an
oracle or edit tests after viewing A0.

## Mechanical release gate

`../scripts/validate_synthetic_pool.py` checks the frozen manifest structure,
exact roles/variants, unique families and prompts, subset diversity/cap, clean
history and duplicate decisions, and the recorded oracle/sandbox evidence. It
does not execute code or prove that the reviews are truthful. A passing report
therefore follows, rather than replaces, manual contract review and independent
oracle execution.

GPU work remains forbidden until all 300 rows pass, the validator passes, the
manifest and test bundle are hashed, and the screen order and all roles are
frozen. The first permitted GPU stage is A0 screening only; it does not
authorize LoRA. If fewer than 300 rows qualify, report the measured shortfall.
A model-generated replacement pool requires another amendment with generator
separation and cannot silently fill this manifest.

