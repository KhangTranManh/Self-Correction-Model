# Phase 16 — recovery-first correction of substantive errors

**Status: recovery-first CPU software contract and a structural A0 discovery
bank are implemented on 2026-11-07. A0 inference remains conditional on a
remote source-only preflight lock; training, protected data and effect-claim
gates remain closed. The earlier fixed 300-task KodCode branch failed its CPU
capacity gate. No Phase 16 feasibility screen, scientific pilot, weight
update, or protected evaluation has run.**

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

The prospective [recovery-first amendment](docs/RECOVERY_FIRST_V1.md) makes
deterministic output recovery and constrained decoding the primary systems
interventions, followed by a bounded stateful harness. Self-edits are saved as
candidate data and independently filtered; they are not applied immediately.
Micro-updates are allowed only for verified substantive residual errors. A
nested `8/16/32/64` correction curve asks when an update begins to beat the
harness at two consecutive levels without near-negative or unrelated-control
regression. Its CPU Boolean-label scorer has synthetic tests only.

The [executable CPU artifact contract](docs/RECOVERY_FIRST_CPU_CONTRACT_V1.md)
now validates frozen recovery evidence, immutable stateful memory and bounded
retrieval logs, self-edit admission audits, the exact 64/80/20/40 study pool,
and a cross-artifact release bundle. Hashes and decisions are recomputed, and
the release report always keeps GPU execution closed. This completes the CPU
software layer, not the missing real-data certification.

The [inference-only A0 handoff](docs/A0_DISCOVERY_GPU_V1.md) resolves the
otherwise circular requirement to possess residual failures before generating
the baseline outputs that reveal them. All nine pinned TACO train shards were
restored and hash-verified. A reproducible bank contains 1,323 structural
call-based candidates; a remote tokenizer plus networkless reference-solution
preflight must freeze exactly 400 before any generation. That lock can
authorize A0 inference only. Dataset tests are not treated as independent
oracles, and no discovered failure enters the 64-correction prefix without
the full CPU admission gates.

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

The earlier [source gate](docs/FEASIBILITY_NEXT.md) defined a read-only audit
before any GPU feasibility screen. The general recovery-first discovery bank
has now been built separately under the A0 handoff above; the original narrow
C-upper-inclusive 400-item eligibility gate was not passed or reinterpreted.

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

CPU family review [started with a 20-family batch](docs/CPU_FAMILY_REVIEW_LOG_V1.md). The
first 20 frozen-order families yielded 6 applicable candidates, 1
near-negative and 13 rejects. One count candidate passed a trusted ten-case
oracle/mutation audit, but history-near-duplicate and sandbox gates remain
open; no task is fully certified.

The review subsequently reached 100 families under the dated
[family-review taxonomy](docs/FAMILY_REVIEW_TAXONOMY_V1.md): 17 direct, 9
execution-adapter, 25 control and 49 hard reject. Direct plus adapter is
26/100, projecting below the fixed requirement of 260/627 even under a
descriptive upper sensitivity bound. The current 300-task branch is therefore
closed at the CPU capacity gate; bulk oracle work and GPU screening are not
authorized without a new dated design decision.

A subsequent [negative-stratum recall audit](docs/KODCODE_RECALL_AUDIT_V1.md)
froze 100 of 3,469 broad components omitted by the shortlist. Four were false
negatives, giving a 4.0% negative-stratum false-negative rate. Combined with
the 26.0% shortlist precision, the descriptive two-stratum recall estimate is
54.02%, with wide 25.27--80.32% sensitivity. This is an audit estimate, not a
certified population result, and it does not reopen the CPU or GPU gate.
