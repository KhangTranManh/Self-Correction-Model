# P16-D0: diverse code, witnessed functional failure

Recorded 2026-11-03. This is a **new exploratory branch** using TACO train,
not a relabeling of the C-upper-inclusive protocol. It is defined before any
TACO model output. Its first structural source test is reported below; no
GPU or model output was used to choose this definition.

## Question and claim boundary

Does one verified correction to a naturally wrong Python solution improve
performance on *different* tasks that share an algorithmic skill, while
preserving performance on unrelated controls? Compare base A0, in-context
correction A1, and temporary LoRA A2 only after the source/verifier gates
below pass. The primary endpoint is full task correctness on unseen probes,
not removal of a particular bug pattern.

The enrollment label is **W-functional-counterexample**: the A0 program
parses, reaches the requested function, completes within the sandbox limit,
passes at least one trusted valid case, and fails at least two distinct
trusted cases under the same unambiguous public contract. Each failed case
must have an independently checked expected answer; two copies of one
input do not count. Syntax errors, timeouts, I/O mismatches, ambiguous
specifications and oracle disputes are separate labels. This is a broad
*observable failure class*, not a claim that all failures share one cognitive
or algorithmic mechanism. Report failure taxonomy separately (boundary,
empty/degenerate, duplicates, ordering, arithmetic, complexity, other) only
when a blinded reviewer can justify it from the contract and witnesses;
taxonomy is descriptive and cannot be used to cherry-pick the primary pool.

This differs from Phase 3's repair/router work: the outcome here is transfer
to disjoint new tasks under a single correction and a matched A0/A1/A2
comparison. Do not interpret a within-task repair as transfer.

## D0-A: source/verifier gate before model screening

- Use only pinned TACO train revision from `TACO_F0_AUDIT.md`. Exclude the
  nine C engineering tasks, all 18 D0 structural-test IDs, exact and
  manually adjudicated near duplicates, historical Phases 1–15 matches,
  image/interactive/ambiguous tasks, and unresolved provenance. Keep TACO
  test unopened. Initially support **call-based** problems only. Standard
  input/output is a separate verifier-extension project, not implicitly
  eligible.
- A task needs a public callable contract, a valid function name and
  unambiguous argument/return encoding. The dataset's test count and
  reference solutions are evidence to check, not a trusted oracle by
  themselves. Require at least five distinct valid cases per admitted task,
  with two held-out potential witnesses. Review case validity against the
  statement and cross-check expected outputs with a reference solution in
  the isolated Colab sandbox and an independent solution/property or manual
  derivation. Do not execute dataset code on Windows or expose expected
  outputs to the model.
- Test and record a sandboxed reference-runner smoke on the 12 list-argument
  cases from the 18 excluded D0 engineering tasks. The six scalar/string
  encoding cases stay out of scope until their encoding is specified and
  separately tested. Log per-task parse, execution, agreement, timeout and
  oracle-dispute states. A structural pass alone is insufficient to admit
  tasks; any disagreement triggers source review, not automatic acceptance.
- Inventory independent task capacity by source, difficulty, skill and
  scaffold family. Before any model output, freeze a 200-task A0
  teaching-screen pool, 80 disjoint same-skill probes, 40 different-skill
  controls matched on source/difficulty where possible, and a sealed
  80-task confirmation reserve. If this cannot be achieved with validated
  cases and at least two source sites plus three skill strata, report source
  infeasibility; do not replace missing strata after seeing A0 outcomes.

## D0-B: conditional model screen and episode pilot

Only after D0-A passes, freeze a separate run lockfile with exact IDs,
partitions, prompt, model revision/settings, sandbox, cases/oracles, order,
seed and hashes. Run one greedy Qwen/Qwen2.5-7B-Instruct A0 completion per
200 teaching-screen tasks, with append-only audit. The 40-episode pilot is
possible only if at least 40 *distinct* naturally occurring
W-functional-counterexample failures survive blinded oracle/label audit
(>=20% of this fixed 200-task screen). Probe/control tasks are assigned
before viewing their A0 outcomes. If the gate fails, report the complete
failure distribution and stop; no threshold relaxation after seeing scores.

For each admitted episode, build one teaching correction from its verified
A0 failure, a task-specific counterexample and a fully verified repair.
The teaching material must not reveal probe/control tasks or tests. Match
two different-task probes by predeclared skill and one unrelated control by
predeclared different skill; no source task or near-duplicate family crosses
roles. A0, A1 and A2 answer identical probes/controls under matched
decoding, seed and budget. A2 starts from the same base checkpoint for each
episode and must pass adapter reset checks. Use full verified correctness
as primary outcome and paired fixes/harms plus control damage as safety
outcomes. Apply the development gate in `PROTOCOL.md` (>=4/80 net probe
gain, more paired fixes than harms, <=1/40 control loss) separately to A1
and A2; no general-learning claim from a pilot.

## Completed source-only structural test

`../scripts/taco_diverse_preflight.py` selected two rows per available
source-by-difficulty cell by fixed SHA-256 rank (`P16-D1-source-preflight-
2026-11-03`) from the preliminary 2,518 call-based metadata candidates.
The 18 selected IDs are permanently excluded from D0 study pools. The
sample contains 6 CodeWars, 6 LeetCode, 6 GeeksForGeeks and 6 each EASY,
MEDIUM, MEDIUM_HARD. All 18 have a parseable nonempty solution list and
nonempty public statement/function-name reference by the structural check.
Twelve have fewer than three published tests; six GeeksForGeeks rows encode
call inputs as strings rather than argument lists. No reference solution
was executed, no test-answer correctness was checked, and no eligibility
rate is inferred from this deliberately stratified sample.

Audit: `../data/taco_inventory_v2/diverse_preflight.json`.
Next executable step: freeze and run the 12-task reference-runner CPU smoke
in the existing Colab sandbox, then audit case/oracle quality. Do not start
TACO A0 GPU screening yet.
