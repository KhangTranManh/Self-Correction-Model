# P16-R0 A0 CPU sandbox closure v1 (2026-11-09)

Status: **implementation ready; real sandbox execution pending**.

This model-free step closes the exact-output sandbox and frozen-recovery gates
for the 15 records currently marked `oracle_ready_remaining_gates_pending`.
It does not generate corrected code, perform human approval, certify training
data, use a GPU, or authorize training.

`../scripts/close_a0_cpu_sandbox_gates.py` verifies the frozen admission
inventory and all six review batches through the existing tracker validator,
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

The real run is pending because the previous Linux sandbox host is not
currently reachable and the local Windows host has neither bubblewrap nor an
active Docker Linux engine. Do not substitute unrestricted local execution of
untrusted recovered code.
