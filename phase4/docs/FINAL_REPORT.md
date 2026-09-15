# Phase 4 final report

Closed on 2026-09-15 with a negative research result. Phase 4 did not establish
reliable autonomous error recognition and safe correction. No model is promoted
to replace Phase 3 Decision-Only V1 or the canonical Phase 1 solver.

## Experiments and findings

The actor was `Kxck/Self_Correction_v1`. Strict KEEP/REVISE contracts were scored
by deterministic math/code verifiers. References and verifier outcomes were
absent from deployment prompts. Guided target construction is separate from
autonomous review and cannot demonstrate independent error detection.

Warm-start V2 improved one 100-source math development run from 73% to 74%,
with one fix. This gain did not reproduce: the replacement GPU gave 73%.
V2 remains the archived pilot reference only. Initial GRPO suffered zero
within-group reward variance and zero policy gradient. Exploration probes
produced real fixes, but completed GRPO V3 still scored 73% on development.

On 2,000 additional math sources, the model made 507 natural initial errors.
Guided attempts found a strict verified fix for 502. The balanced 904/100 split
supported completed SFT V3 and DPO V1 runs. SFT fixed three answers and harmed
two, ending at 74%. DPO achieved 97% held-out preference-pair accuracy but only
73% behavioral accuracy with no fixes. Better loss and pair ranking did not
transfer to reliable correction.

Blind preferences completed all 16,064 reviews: 108 fix attempts and 42 harmful
revisions. Source-level available train KEEP/REVISE pairs were 14/44 and dev
2/4. Balancing left 28 train and four dev pairs. The preset minimums of 20 train
and five dev pairs per action failed. DPO V2 was not trained; sampling and
thresholds were not changed after inspection.

Three-round autonomous review used 80 fixed sources balanced across math/code
and initial correct/wrong states at three seeds. Mean accuracy was 50%
initially, 51.25% after round 1, 49.58% after round 2 and 49.17% after round 3.
Round 3 declined 2.08 percentage points versus round 1 (stratified source
bootstrap 95% interval: -4.58 to -0.42 points). Across 240 seeded trajectories,
retained fixes fell from three to one and initial-answer harms rose from zero
to three. Code accuracy stayed 50%. Repeated review failed its benefit gate.

## Evidence limits

These are bounded development pilots. The original 100-row set is math-only;
the fresh 80-row diagnostic deliberately starts at 50%. Their accuracies are
not directly comparable. Three seeds reuse the same sources, not 240
independent problems. Original confirmation candidates did not meet the
preregistered balance requirements and remain unopened. The primary sealed
promotion gate was not evaluated. No confirmed solution is claimed.

GPU replacements lost an early warm-start artifact and interrupted an earlier
DPO run. Those historical observations remain labeled in the pilot log. The
completed replacement DPO and final diagnostic have local backups and hashes.
Verification uses aligned CPython 3.10 / SymPy 1.14.0 following a Python 3.13
parsing discrepancy; historical verifier code was preserved.

## Closure

Phase 3's central problem remains unresolved. Preserve adapters, checkpoints,
raw generations, manifests, configs and negative results. No additional Phase 4
experiment or tuned threshold is authorized by closure. Future research must
open a separate phase with new hypotheses and protected evaluation.

See [RESULTS.md](RESULTS.md) for measurements and evidence paths,
[CODEBASE.md](CODEBASE.md) for ownership, and [PILOT_LOG.md](PILOT_LOG.md) for history.

Local closure validation passed: all four final adapter hashes, 16,064 blind
attempts, balanced diagnostic hashes, all 240 review trajectories and recomputed
metrics. Fresh verification of the real blind preference pairs also passed;
pair validity does not override the failed coverage gate. Refactored helpers
and affected CLI entry points passed smoke checks and syntax compilation.

At closure maintenance, the replacement GPU SSH endpoint refused connections
on three attempts. Final experiment evidence was copied and hash-checked before
this outage; the latest documentation and code cleanup are local and prepared
for later synchronization. No remote deletion or instance termination was made.
