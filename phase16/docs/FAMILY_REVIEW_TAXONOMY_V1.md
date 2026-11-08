# P16-S0 family-review taxonomy amendment v1 (2026-11-06)

Status: **CPU review amendment; GPU gate remains closed**.

This amendment refines the manual contract-review decision used for the
frozen KodCode family inventory. It does not certify a task, assign a study
role, open a protected split, or authorize model generation. The family order,
source pins, one-family/one-task rule, error definition, and 300-task quotas in
`SYNTHETIC_DEVELOPMENT_AMENDMENT_V1.md` are unchanged.

## Why the old decision is insufficient

The first review batch used `applicable_candidate`,
`near_negative_candidate`, and `reject`. That is sufficient to exclude false
positive text hits, but it conflates three materially different cases:

1. a task whose semantic contract matches but whose interface needs a fixed
   mechanical adapter;
2. a clean task suitable only as a near-negative or unrelated control; and
3. a task that is unusable because its contract is ambiguous or outside the
   study.

The old labels remain in their original artifacts. This amendment adds a
separate review layer; it does not rewrite those artifacts.

## Four active lanes

Every reviewed family receives exactly one active lane.

### `direct_candidate`

The source already specifies one callable target with defined inputs, one
checkable output, an explicit inclusive upper endpoint, and a count, sum, or
select operation admitted by the locked C contract. Normal test-fixture
construction (for example a `TreeNode`) is not an adapter.

### `execution_adapter_candidate`

The semantic task satisfies the same C contract, but a deterministic,
model-independent interface adapter is needed. Permitted adapter work is
limited to:

- giving an otherwise explicit single operation a canonical function wrapper;
- converting one standard-input/single-output operation to that wrapper;
- selecting the single observable entry point when a helper is also required;
- removing a purely syntactic scaffold blemish without changing prose,
  examples, bounds, types, or output semantics.

The raw source text and adapted prompt must both be retained and hashed. The
adapter and its exact diff must be frozen before A0. It may not add an
algorithm, an endpoint convention, ordering or tie behavior, missing examples,
or a semantic assumption. Multi-operation state machines, multi-query
programs, and ambiguous contracts cannot enter this lane.

This is a synthetic-development interface normalization. Any result remains a
claim about the P16-S0 development distribution, not natural-task transfer.

### `control_candidate`

The family is execution-ready but is not a positive C task. It must also have
one of two frozen subtypes:

- `near_negative`: structurally matched, but the C rule does not apply (for
  example, the upper endpoint is explicitly exclusive);
- `unrelated`: a different skill with no C-upper-inclusive target behavior.

Controls do not contribute to the 260 applicable-task capacity. A task for
which the inclusive rule still applies but whose output merely falls outside
count/sum/select is not a near-negative; it is rejected from this pool.

### `hard_reject`

The task is unusable for the active pool: ambiguous or contradictory contract,
missing semantic information, excluded output, non-isolatable operation,
historical/provenance failure, or another gate that cannot be repaired by the
permitted mechanical adapter. Hard rejects stay in the inventory.

## Deferred multi-operation queue

`deferred_multiop` is an exclusion status, not a fifth active lane. It records
otherwise interesting multi-method, multi-query, or mutable-state tasks for a
possible future protocol. These tasks are excluded from all P16-S0 capacity,
oracle, and role counts unless a later dated amendment defines a separate
error and verifier.

## Evidence and capacity accounting

Contract review is performed in the manifest's existing SHA-256 frozen order
without viewing study-model output. One connected `seed_ids` family receives
one lane. Each row retains the representative ID and question hash, a concise
reason, and pending statuses for oracle, history, semantic-near-duplicate, and
sandbox gates.

Only `direct_candidate + execution_adapter_candidate` count toward the 260
applicable-task requirement. A lane is not certification. Oracle preflight is
allowed only after the first 100 frozen families have been reviewed and the
observed lane counts make 260 applicable families plausible. If the 100-family
projection remains below 260 or is unstable, continue CPU review or close the
branch; do not use GPU output to rescue capacity.
