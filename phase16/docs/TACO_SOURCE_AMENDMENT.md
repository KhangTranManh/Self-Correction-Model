# P16-F0 amendment: broader TACO source inventory

Recorded 2026-11-03 before downloading TACO or generating any Phase 16 study
output. This replaces the **source-discovery part** of `FEASIBILITY_NEXT.md`;
it does not change the C-upper-inclusive error definition, the P16-E1
engineering exclusion, or the conditional A0/A1/A2 comparison. No study
pool is frozen by this document.

## Decision and rationale

Use `BAAI/TACO` **train** as the primary *candidate inventory*, not as an
automatically eligible study set. Its published card lists 25,443 train and
1,000 test problems, with source URL, difficulty, tags/skills, Python
solutions, and input/output tests. It explicitly reports that some problems
were curated from APPS and CodeContests. Thus the larger nominal count may
not translate into independent, well-verified examples of our narrow error.
Do not touch TACO `test` during development; it is not automatically a valid
confirmation set either. LiveCodeBench is only a possible later external
evaluation source, with a separately locked protocol, not part of F0/F1.

Source card: https://huggingface.co/datasets/BAAI/TACO/blob/main/README.md

## F0-A: acquisition and provenance (CPU, no model)

1. Pin the exact TACO dataset revision/commit and file hashes at download.
   Record retrieval date, loader version, row count, schema and splits.
   Read train rows only. Do not execute dataset-provided solutions or use
   `eval` on serialized metadata; parse JSON where applicable and reject
   malformed fields.
2. Inventory each row with original URL, source site, title, difficulty,
   tags/skills, statement hash, test count, `fn_name` presence, solution
   count, and image/interactive/ambiguous-input flags. Treat URL as a useful
   source ID, not proof of uniqueness. Keep source licence/provenance notes;
   do not republish raw third-party statements or solutions in reports.
3. Build a Phase 1–15 exclusion registry from source URLs/IDs and normalized
   problem text, including protected/held-out sets and Phase 16's nine
   engineering tasks. Exclude matching rows. Canonicalize URLs (host/path,
   remove fragment and tracking query); exact-deduplicate normalized
   statements; flag near duplicates with token 5-gram Jaccard >= 0.80.
   Manually resolve flags and record decisions before viewing any new A0
   response. All duplicates/templates stay in one family; no family crosses
   teaching, probe/control, and confirmation partitions.
4. Report the complete funnel: raw rows, parseable rows, excluded by each
   reason, unique remaining rows, and counts by source/difficulty/skill.
   Missing IDs, contradictory tests, image-dependent or interactive tasks,
   and unclear rights/provenance are separately flagged; they are not silently
   treated as eligible.

## F0-B: error-specific eligibility and verifier (still no model)

For **C-upper-inclusive only**, two reviewers inspect source-only evidence:
the statement must explicitly require an inclusive upper bound, have a
deterministic Python callable contract, and permit independent oracle and
two discriminating metamorphic boundary pairs. Record reject/accept reasons
and disagreements. Dataset tests alone are insufficient: independently
validate boundary pairs and ordinary/lower/equal/outside cases against the
contract. The published `input_output` field may be function-call or
standard-input format; do not silently coerce the latter into the current
function-only verifier. Reject unsupported interfaces until a separate
sandboxed verifier is implemented and tested.

Publish counts by count/sum/select variant, source site, difficulty, skill,
and scaffold family. The F0 capacity gate remains the one in
`FEASIBILITY_NEXT.md`: 400 eligible teaching-screen tasks, 80 disjoint
applicable probes, 40 unrelated controls, and an independent confirmation
reserve of at least 80. These are *minimum independent task counts*, not
rows created by changing constants or variable names. Check that each
teaching episode can have two nonduplicate probes with at least one different
variant. If this cannot be constructed, report C-upper-inclusive as
source-infeasible under the current design; do not lower the gate after
looking at available rows.

If F0 passes, freeze the selected IDs and split assignments, prompts, tests,
oracle hashes, verifier, model settings and screening order in a separate
F1 lockfile **before** any A0 study generation. Then the conditional 400-item
screen in `FEASIBILITY_NEXT.md` may run. Dataset size alone is not a reason
to launch GPU.

## Diversity branch if the narrow error has insufficient capacity

Do **not** relabel arbitrary wrong answers as C-upper-inclusive or pool
different errors to reach 40. Instead close this branch with its observed
source counts. A later, separate P16-D protocol may use TACO's broader
task diversity, but must first define a measurable new error family, its
counterexample/oracle rules, disjoint transfer probes, controls, sample size,
and primary outcome. Any exploratory reading of model failures used to
discover that family is development data and cannot serve as its confirmatory
sample. The symbolic-math candidate remains a separate alternative.

## Current state

The initial train-only metadata and exact-overlap audit has since been run
([record](TACO_F0_AUDIT.md)). Near-duplicate review, source-family
adjudication, error-specific eligibility, independent oracle verification,
and any A0 model screening are still pending. No study pool is frozen.
