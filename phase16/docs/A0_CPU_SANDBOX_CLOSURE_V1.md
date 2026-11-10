# P16-R0 A0 CPU sandbox closure v1 (2026-11-09)

Status: **completed on the local CPU Docker/bubblewrap sandbox**.

This model-free step closes the exact-output sandbox and frozen-recovery gates
for the 24 records marked `oracle_ready_remaining_gates_pending` through
review batch 8. Batch 9 later added four more ready records that are outside
this immutable closure artifact.
It does not generate corrected code, perform human approval, certify training
data, use a GPU, or authorize training.

`../scripts/close_a0_cpu_sandbox_gates.py` verifies the frozen admission
inventory and all eight review batches through the existing tracker validator,
then binds each oracle-ready position to its task-specific contract cases and
trusted oracle. It verifies the exact recovered-output hash and requires the
frozen recovery status `accepted` before executing the output in the same
UID-65534, networkless bubblewrap sandbox used by P16-G0.

The resulting artifact retains only identities, hashes, case counts, sandbox
status and pass counts. It must not retain candidate code, independent input
values or expected values. A residual passes this gate only when the recovered
output still fails at least one independent case or fails execution. An output
that passes every independent case is rejected as `residual_not_reproduced`.

Passing this step is not admission. Full semantic-history review, manual
cross-history near-duplicate review, human approval and a verified corrected
implementation remain required. Independent cases remain held out from any
future correction prompt or training record.

The real run completed on 2026-11-10 in a CPU-only Docker Linux container.
The outer container had no network, a read-only root filesystem, a read-only
repository mount, two CPUs, 1 GiB RAM and a 64-process cap. Bubblewrap created
another network/PID/filesystem boundary for each candidate, and `setpriv`
dropped it to UID/GID 65534. All 24 candidates completed execution and failed
at least one independent case: 24 residuals reproduced, zero unexpectedly
correct and zero admitted. Artifact:
`../results/a0_cpu_sandbox_closure_v1.json`; closure SHA-256
`71fe8dbdf483ef380b359032b020318dc877ec4cc51a28a699f29e5badfa8ff2`.

The reproducible image definition is
`../docker/a0-cpu-sandbox.Dockerfile`. Docker mode is opt-in through
`P16_A0_SANDBOX_BACKEND=docker` and refuses to activate outside a root process
inside a container. The original user-namespace backend remains the default.

## Design decision

The closure keeps an explicit fail-closed registry from each frozen position
to its independent cases and oracle. A distributed registry exported by every
historical batch would reduce this central list, but would also rewrite the
review modules that produced existing evidence. The explicit registry leaves
those modules immutable and makes any tracker/spec disagreement stop before
untrusted code executes.
