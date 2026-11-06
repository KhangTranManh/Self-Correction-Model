# Phase 16 — verified correction of substantive errors

**Status: source-audit and CPU-definition stage, 2026-11-05. A nine-task Colab T4 A0 engineering smoke, train-only dataset audits, and a synthetic CPU signal-gate check completed; no Phase 16 feasibility screen, scientific pilot, or protected evaluation has run.**

Phase 15 closed after its S-format development diagnostics. Phase 16 asks
whether one verified correction helps Qwen/Qwen2.5-7B-Instruct avoid a
repeatable error in the content of a new answer while preserving performance
on unrelated tasks. The first candidate is a Python interval-boundary error;
a symbolic-math sign error is defined as a separate fallback. Each candidate
must pass a feasibility screen before an intervention pilot.

Read [the protocol](docs/PROTOCOL.md) for the error signatures, test design,
data separation, scoring, and decision gates. The chronological record is
[RUN_STATUS.md](RUN_STATUS.md). Phase 15's closed report is
[../phase15/docs/FINAL_REPORT.md](../phase15/docs/FINAL_REPORT.md).

A separate [CPU-defined signal-only pilot](docs/SIGNAL_ONLY_CPU_V1.md) now
defines autonomous repair versus blind retry and one-correction application
versus matched sham arms. Its Boolean-verdict scorer and synthetic tests are
ready, but no model or source-pool result exists. It does not replace the
original 40-episode gate.

The [AWS GPU budget/run definition](docs/AWS_GPU_BUDGET_V1.md) limits a first
domain to a planned USD 45 ceiling and both domains to USD 80, with stage
stops. It is planning only: no AWS instance has been launched, and the CPU
source/verifier gates must pass first.

The first error type now has a [worked test fixture](docs/C_UPPER_INCLUSIVE.md),
a [test-transcript scorer](scripts/score_boundary.py), and synthetic verifier
checks in `tests/test_score_boundary.py`. No model-written program is executed
by the scorer.

A [Colab CPU definition smoke](docs/COLAB_DEFINITION_SMOKE.md) verified the
three variants and saved its audit in `results/`. It used no model or GPU.

A [Colab T4 engineering run](docs/ENGINEERING_A0.md) then checked nine
synthetic tasks end-to-end. Qwen solved 9/9, and a saved-output rescore found
zero disagreements. Those tasks are excluded from the future study pool.

The [next gate](docs/FEASIBILITY_NEXT.md) is a read-only source audit before
any GPU feasibility screen. It defines source capacity, overlap checks,
frozen splits and the 400-item A0 screening decision rule; neither step has
run yet.

The [TACO source amendment](docs/TACO_SOURCE_AMENDMENT.md) broadens the
candidate inventory while keeping the narrow error label and no-leakage
gates intact. The [initial TACO train audit](docs/TACO_F0_AUDIT.md) has now
downloaded and inventoried the train split only; it has not screened model
outputs or established a verified error-specific study pool.

A separate [diverse-code D0 branch](docs/DIVERSE_CODE_D0.md) now defines a
verified-counterexample failure class and reports a source-only 18-task
structural preflight. Its sandboxed oracle smoke and any model screen remain
pending.
# Current CPU pool/verifier decision

The 2026-11-05 source and error-scorer recheck is recorded in
`docs/POOL_AND_ERROR_GATE_CPU_V1.md`. Its decision is **not GPU-ready**:
no 300-task/domain manifest or independently validated oracle exists yet.

A [four-source CPU candidate mix](docs/FOUR_SOURCE_MIX_F0.md) now inventories
TACO, CodeContests, EvalPlus and one LiveCodeBench shard with pinned source
hashes and overlap flags. It remains an unassigned metadata inventory, not a
certified study pool; the C-upper-inclusive capacity gate is still closed.

An [extended capacity audit](docs/EXTENDED_SOURCE_CAPACITY_V1.md) adds a
complete LeetCodeDataset scan and a complete projected-column KodCode scan.
It finds plausible capacity only if KodCode is explicitly admitted as a
synthetic development distribution; zero added task has yet passed the full
oracle gate, so this does not release GPU work.

The [synthetic-development amendment](docs/SYNTHETIC_DEVELOPMENT_AMENDMENT_V1.md)
now fixes a 300-task S0 manifest contract: 260 rule-applicable tasks, 40
controls, unique connected seed families, independent oracles, and a mechanical
validator. The current 627-family KodCode review inventory has zero certified
tasks, so this definition still does not release A0 or GPU work.

CPU family review has now [started](docs/CPU_FAMILY_REVIEW_LOG_V1.md). The
first 20 frozen-order families yielded 6 applicable candidates, 1
near-negative and 13 rejects. One count candidate passed a trusted ten-case
oracle/mutation audit, but history-near-duplicate and sandbox gates remain
open; no task is fully certified.
