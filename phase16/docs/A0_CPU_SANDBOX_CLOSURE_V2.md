# A0 CPU sandbox closure v2

Status: **executed on CPU; one oracle/contract mismatch remains unresolved**.

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

## Execution result

All twelve planned batches completed in a CPU Docker/bubblewrap sandbox. The
first 24 records were also rerun using exact equality. The verified aggregate
in `../results/a0_cpu_sandbox_v2/summary.json` contains 142 checked: 141
completed with at least one independent case mismatch, zero passed every case,
and one had an execution error. These are sandbox observations, not final
admission decisions; oracle correctness and remaining history, duplicate,
correction and human gates still matter.

The one error at position 392 (`pay_cheese`) exposed a flaw in the frozen
batch-35 oracle: it supplies a number, while the captured prompt specifies an
array of five daily values and a currency-formatted string. The runner
therefore exercised the function with the wrong input type; this is not
evidence of a model runtime failure on a valid input. A later code-point check
confirmed that the captured currency symbol is `£` (U+00A3); its earlier display
as a replacement character was a terminal rendering error. Exact string
serialization still needs clarification. Position 392 remains unresolved;
do not treat its earlier oracle-ready designation as validated admission.
The earlier review and execution artifacts remain immutable historical
evidence. A corrected contract assessment requires a new versioned artifact.

The run used Docker `--network none`, a read-only repository and root
filesystem, two CPUs, 1 GiB RAM, a 64-process cap, and bubblewrap with an
unprivileged UID. The container needed `SYS_ADMIN` and `NET_ADMIN` capabilities
and an unconfined seccomp profile for nested namespace creation; candidates
ran inside the additional bubblewrap boundary. No GPU or model generation was
used. No correction was admitted and GPU training remains unauthorized.
