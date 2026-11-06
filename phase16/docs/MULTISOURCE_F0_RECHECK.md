# Phase 16 multisource F0 recheck (2026-11-03)

This is source discovery, not a revised preregistration, frozen pool, model
screen, or efficacy result. The existing 40-episode capacity gates in
`PROTOCOL.md`, `FEASIBILITY_NEXT.md`, and `DIVERSE_CODE_D0.md` remain in force
until explicitly amended before model screening.

## CodeContests metadata check

Source: `deepmind/code_contests`, repository revision
`802411c3010cb00d1b05bad57ca77365a3c699d6`. Only the first three of
39 train parquet shards were downloaded; validation/test were not accessed.
The reproducible read-only audit is
`../scripts/audit_codecontests_subset.py`; its output is
`../results/codecontests_three_shard_metadata_audit.json`.

The three shards contain 1,026 rows. Of these, 965 have at least five *raw*
paired test cases and 683 exactly match a normalized TACO train description.
After requiring Python 3 correct and incorrect submissions, no special file
I/O, five raw paired tests, and no exact TACO-text match, 302 unique
description candidates remain. This is **not** a validated 240-task pool:
submission labels and test oracles were not executed; near-duplicate,
historical-source, contest-family, and verifier checks remain. Only 62
metadata candidates would remain if 240 survived, so more source shards may
be needed to absorb those exclusions. The first three shards are not a
random sample of the full dataset.

CodeContests uses standard input/output; the current Phase 16 verifier is
function-call only. A separate sandboxed standard-I/O verifier and trusted
reference-vs-incorrect-solution checks are prerequisites to model screening.
Published incorrect submissions cannot substitute for naturally occurring
A0 errors or define the error mechanism on their own.

## Mathematics Dataset generator check

The official `google-deepmind/mathematics_dataset` source was cloned at
`427f45075f84b8b9774950196ad63867ca20ffb3`; its code is Apache-2.0.
`absl-py` was installed only under `../data/python_deps`. The generator ran
for `arithmetic__add_or_sub` and `arithmetic__mixed` on the local Python 3.13
environment. A `polynomials__expand` sample failed because upstream calls
`ndarray.itemset`, removed in the installed NumPy 2.x. No generated items
were admitted to a study pool.

More importantly, this source emits question/answer pairs, not required
intermediate expansions. The Phase 16 `M-negate-difference` label requires an
explicitly observable `-(a-b) -> -a-b` intermediate step. Arbitrary generated
math questions therefore do **not** qualify as 240 M-negate-difference
items. A compatible, pinned generator, a predeclared task subset or separate
construction, a step-output prompt/parser, and independent family splits
would all need an audit before a math source-capacity claim.

## Decision

Continue source/verifier feasibility only. Do not freeze a 240+60 design,
run A0, claim a math error pool, or open protected evaluation data based on
these metadata checks. A 20-episode/300-task design would be a protocol
amendment, not a silent implementation of the current 40-episode design.
