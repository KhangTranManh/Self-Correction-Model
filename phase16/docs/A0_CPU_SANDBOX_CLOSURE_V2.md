# A0 CPU sandbox closure v2

Status: **defined and input-verified; execution pending**.

The v2 plan covers the 118 oracle-ready positions that were not included in
the prior 24-record closure. It contains 12 ordered batches, each with at
most ten records, in `../results/a0_cpu_sandbox_plan_v2.json`. Its plan hash
is `334c5a9fe4d2132e60f5cc93fbdab109fa89ee824805621ab922ee125017f9c9`.
Creating the plan did not execute recovered code.

Before execution, `../scripts/close_a0_cpu_sandbox_v2.py` verifies the frozen
inventory, all 36 review hashes, the earlier closure's hash and chain tip,
all completion identities and recovered-code hashes, and each current
oracle/reference/mutant transcript against its frozen review evidence.
The CLI also verifies the discovery manifest, lock, shard hashes and ordering.
It refuses to overwrite an existing output. Each batch is independently
restartable and records identities and hashes, never case inputs or expected
values.

Independent expected results use exact equality. The discovery-only rule
that unwraps a singleton published expected list is not applied. A completed
execution with at least one wrong result receives
`substantive_residual_reproduced`; an entirely correct execution receives
`residual_not_reproduced`. Runtime errors, timeouts, missing output and other
execution failures receive `execution_or_scaffold_unresolved` and must not be
counted as substantive. All observations remain subject to later admission
and human review.

The existing v1 closure is preserved. Its 24 records were executed using the
discovery comparison and the old classifier. Their observations all had
completed status, but the v1 results have not been revalidated under v2's
strict comparison. Do not silently combine their classifications with v2
results as if they used the same verifier.

## Execution requirements

Run batches only inside the previously described CPU Docker/bubblewrap
environment: networkless, read-only repository mount and root filesystem,
resource limits, and a writable output mount separate from repository inputs.
The sandbox preflight must pass. A local Windows Python process is not a
substitute for that boundary. Use the `start` and `count` in the plan with
`close_a0_cpu_sandbox_v2.py`, the same inventory, all 36 reviews in order,
discovery run and manifest, and previous closure. Save each output under a new
name, then verify its hash and matching positions before moving to the next
batch. The plan is not an execution result.

The sandbox engine was unavailable when this definition was prepared. No
new sandbox results, correction admission, protected opening or GPU
authorization are implied.
