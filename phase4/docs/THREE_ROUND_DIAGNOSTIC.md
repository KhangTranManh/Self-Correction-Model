# Three-round autonomous review diagnostic

Status: completed on the replacement GPU, after blind preference collection.
Code and transition/metric checks were completed locally before upload.

## Result (2026-09-15)

All 240 trajectories (80 sources, three seeds) completed, totaling 720 reviews.
Mean accuracy was 50% initially, 51.25% after round 1, 49.58% after round 2,
and 49.17% after round 3. Round 3 versus round 1 declined 2.08 percentage
points; the stratified source-bootstrap 95% interval was [-4.58, -0.42] points.
These are development diagnostic results, not confirmation.

Round 1 accuracy by seed was 52.5%, 50%, and 51.25%; round 3 was 48.75%,
50%, and 48.75%. Round 1 retained three fixes and zero initial-answer harms
across seeded trajectories; round 3 retained one fix and three initial-answer
harms. Code accuracy stayed 50%. Extra reviews did not meet the benefit gate.
No weights were trained or promoted. The initial 50% is fixed by the balanced
diagnostic construction and is not an estimate of natural task accuracy.

Evidence: `phase4/runs/three_round_v1/{protocol.json,trajectories.jsonl,summary.json,run.log}`.
The trajectories SHA-256 is
`8eee741af05730576ea9224116eba98349f2ba25b2c829a2b2a2e125f91c978b`.
Local checks validate all trajectory state transitions and recompute aggregate
metrics from the saved raw reviews.

## Frozen protocol

Question: does reviewing the evolving answer three times improve correctness
over reviewing it once, without increasing harmful changes?

Use selected warm-start V2 with frozen weights and the existing fixed 80-source
fresh math/code diagnostic. Reuse its natural initial answers. Use the deployment
system and neutral review prompt at every round, with only the problem and current
answer as context. Do not provide references, verifier outcomes, error labels,
hidden tests, or previous evaluation scores to the model. KEEP retains the current
answer; REVISE replaces it with the strict parsed revision. Invalid reviews retain
the current answer and are counted separately. Continue all three rounds even
after KEEP; do not stop according to correctness.

Freeze generation settings to the existing Cycle 0 rollout configuration and use
seeds 20260914, 20260915, and 20260916. Record request seeds separately per source
and round. The first-round result is the one-review baseline; rounds two and
three extend the same trajectory. Never choose the best round using correctness.

Persist every raw review, action, answer, and offline verification result. Verify
with the aligned CPython 3.10 / SymPy 1.14.0 runtime. Record input/config hashes
before generation. Evaluation outputs must not enter training.

Report initial and round 1/2/3 accuracy for every seed, math/code accuracy,
strict-contract validity, consecutive-round fixes and harms, and initial-to-final
fixes and harms. Compare round 3 against round 1 on the same sources. Average
seeds within each source for paired accuracy differences; bootstrap by source,
stratified by the four domain/correctness buckets, for a 95% interval. Do not
treat the three seeds as independent new problems.

A preliminary benefit requires higher mean round-3 accuracy than round 1,
no increase in initial-to-final harms, and no decrease in code accuracy or strict
contract validity. Report uncertainty and seed consistency even if these conditions
hold. A positive diagnostic is not proof that Phase 3 is solved: confirmation
requires a separately frozen compliant holdout. If performance stagnates or
declines, retain the negative result and do not select a favorable seed or round.

Verifier-guided repair is a different experiment. It may be tested separately,
but feedback-assisted improvement cannot establish autonomous error detection.
