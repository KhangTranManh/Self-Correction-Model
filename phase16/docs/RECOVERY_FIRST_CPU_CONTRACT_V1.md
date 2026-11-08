# P16-R0 CPU artifact contract v1 (2026-11-07)

Status: **implemented and covered by synthetic tests; no real study pool is
certified**.

This contract turns `RECOVERY_FIRST_V1.md` into executable, fail-closed CPU
checks. It does not run a model, execute candidate code, reopen protected
data, authorize GPU execution, or claim that recorded oracle evidence is
true. The deterministic verifier remains the correctness oracle.

## Executable contract

`../scripts/validate_recovery_first_cpu.py` dispatches on `schema_version` and
accepts five artifact types:

| Schema | Purpose |
|---|---|
| `p16_recovery_batch_v1` | Frozen reference-free recovery wrapper, raw/recovered outputs, independent verdicts, preservation/idempotence/noninterference fixtures, and exact outcome accounting. |
| `p16_harness_memory_v1` | Immutable stateful-memory records, positive and near-negative evidence, provenance, deterministic bounded retrieval patches, and append-only run logs. |
| `p16_self_edit_batch_v1` | Raw and normalized self-edit proposals, leakage/duplicate/oracle/human gates, deterministic rejection reasons, and the same-episode update prohibition. |
| `p16_recovery_first_pool_v1` | Exactly 64 residual corrections, 80 positive probes, 20 near-negatives, and 40 unrelated controls with unique source/task/family IDs and frozen nested correction order. |
| `p16_recovery_first_cpu_release_v1` | Cross-artifact release check tying memory and self-edits to pool corrections and run logs to frozen evaluation tasks. |

Every object uses exact keys. Content, records, batches, memory, pool, and
release hashes are recomputed rather than trusted. Unknown fields fail so an
unlogged intervention cannot silently enter the contract.

## Recovery invariants

The validator requires at least one payload-recovery fixture, one
already-valid noninterference fixture, and one deliberate rejection fixture.
All records retain raw and recovered output, use the same frozen family
wrapper, prove deterministic rerun, idempotence, and payload preservation,
and record no reference-answer, hidden-test, or other-arm access.

An accepted recovery must change bytes. An unchanged or rejected recovery
must preserve endpoint bytes. A `substantive_content` record must remain
incorrect after recovery; otherwise it was misclassified. Any correctness
harm fails the batch.

## Stateful-memory invariants

Each immutable memory record has a correction ID, error family, applicability
trigger, non-applicability boundary, rule, permitted action, positive and
near-negative verifier evidence, provenance, leakage status, and human
approval.

The frozen `memory_sha256` covers only memory configuration and immutable
records. It intentionally excludes append-only run logs. Each run separately
hashes the retrieved rule IDs and exact deterministic prompt patch. Retrieval
cannot exceed frozen `top_k` or `token_budget` and may not access the current
verdict, hidden tests, probe answer, or later memory.

`whitespace_v1` is a deterministic CPU accounting method, not a claim about
the learner tokenizer. A later GPU protocol must freeze tokenizer-specific
accounting separately if it uses a different budget.

## Self-edit admission invariants

Every proposal preserves its raw text and normalized fields. Acceptance is
computed from the schema, leakage flags, at least one verified positive and
one verified near-negative, all-example verifier success, exact and semantic
duplicate checks, and human approval. `decision` and ordered
`rejection_reasons` must match the computed result.

`applied_in_same_episode` must always be false. Rejected records remain in the
batch audit; they are not dropped or silently repaired into accepted data.

## Pool and release invariants

The pool contains exactly 204 unique records:

- 64 `correction` records;
- 80 `positive_probe` records;
- 20 `near_negative` records; and
- 40 `unrelated_control` records.

Every correction is independently recorded as `substantive_content` and
still wrong after recovery. Harness correctness is deliberately absent from
the pre-run pool manifest: it is a treatment outcome measured later on the
held-out arms, never a field used to select correction data. All records must
pass recorded oracle, history, near-duplicate, and sandbox gates and have no
protected or Phase 15 reserved overlap. The validator emits SHA-256 values
for the frozen 8/16/32/64 correction prefixes.

The aggregate CPU release additionally requires harness memory size to be one
of 8/16/32/64, in the exact frozen correction order. Self-edits must point to
pool corrections; harness runs must point only to frozen evaluation tasks.
The release report explicitly returns `gpu_execution_authorized: false`.

`recovery_first_gate.py` now also requires the frozen pool SHA-256 and complete
64-item correction order. Every `training_ids` level must equal the matching
8/16/32/64 prefix, preventing a curve run from silently substituting a
different correction batch.

## Commands

Validate any individual artifact or aggregate release:

```powershell
.\.venv-verify310\Scripts\python.exe -B phase16\scripts\validate_recovery_first_cpu.py INPUT.json
```

Write a new report without overwriting an existing file:

```powershell
.\.venv-verify310\Scripts\python.exe -B phase16\scripts\validate_recovery_first_cpu.py INPUT.json --output REPORT.json
```

Run the synthetic contract tests:

```powershell
.\.venv-verify310\Scripts\python.exe -B -m unittest phase16.tests.test_validate_recovery_first_cpu phase16.tests.test_recovery_first_gate
```

## Boundary before GPU work

The CPU implementation is complete as software only. A real CPU release does
not yet exist because the failed KodCode branch supplies neither 64 certified
residual corrections nor the fixed held-out roles.

The later `A0_DISCOVERY_GPU_V1.md` amendment separates inference-only residual
discovery from micro-update authorization. Its source-side preflight may
authorize A0 generation on a frozen development lock without pretending that
the as-yet-unobserved residual pool already exists. That narrow authorization
does not pass this aggregate release, certify dataset tests as independent
oracles, open protected data, or authorize training. LoRA, micro-update,
protected opening, and any update-effect claim remain closed until real
artifacts pass this aggregate validator and a separate update execution
protocol is frozen.
