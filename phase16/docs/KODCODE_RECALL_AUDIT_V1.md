# P16 KodCode semantic-shortlist recall audit v1 (2026-11-06)

Status: **CPU sampling definition; GPU gate closed**.

The first 100 predicted-positive families gave shortlist precision 26/100
under `FAMILY_REVIEW_TAXONOMY_V1.md`. Recall cannot be computed from positive
predictions alone. This audit freezes a family-level sample from the broad
candidate population that the semantic shortlist predicted negative.

## Source and population

Source: KodCode-V1 train revision
`97d81f0695ddd7c5c02bfe16436db9a17a4a21e5`. Only projected columns
`question`, `question_id`, `subset`, `metadata`, `benchmark_similarity`, and
`benchmark_task_id` are streamed. Dataset code, solutions, and tests are never
executed. TACO test and all protected project data remain unopened.

The broad candidate gate is unchanged from the extended-source audit:

1. all four C text flags;
2. element-operation and range-entity flags;
3. `benchmark_similarity < 0.8`.

Rows are joined into connected components by `(subset, seed_id)`. A broad
component is a predicted-positive component if it contains any row ID retained
by `kodcode_c_semantic_shortlist_all.json`. Components without a retained row
are the predicted-negative recall-audit population. This component definition
prevents a rejected paraphrase from being sampled as negative when another row
from the connected seed family was shortlisted.

## Frozen sample

Each predicted-negative component receives a stable family ID from its subset
and sorted connected seed IDs. Sample order is SHA-256 of
`"p16-kodcode-recall-v1:" + family_id`; the first 100 components form the
sample. The representative row is the member with the smallest SHA-256 of
`"p16-kodcode-recall-representative-v1:" + row_id`.

The sampling script scans the projected columns but writes only counts and the
100 sampled representatives. It refuses to overwrite an existing artifact.
The sample manifest is frozen before manual review of its text.

## Labels and estimands

Review uses the exact taxonomy in `FAMILY_REVIEW_TAXONOMY_V1.md`:

- `direct_candidate` or `execution_adapter_candidate` = false negative;
- `control_candidate` or `hard_reject` = true negative.

The negative-sample false-negative rate is `FN_sample / 100`. Shortlist
precision remains `26/100`; the earlier 50-row text-contract diagnostic is not
mixed into this estimate.

Let `N_pos = 627` existing shortlist families and `N_neg` be the number of
broad predicted-negative components reported by the sampler. The plug-in
family-level recall estimate is:

`(0.26 * N_pos) / ((0.26 * N_pos) + (FN_sample / 100) * N_neg)`.

This is a two-stratum audit estimate, not a certified population fact. Report
the positive precision and negative false-negative rate separately, their
Wilson intervals, the plug-in recall, and sensitivity using interval bounds.
Do not use the recall audit to admit tasks, assign study roles, or release GPU
work. Any sampled false negative still needs contract, oracle, history,
near-duplicate, and sandbox gates.

## Frozen result

The pinned projected-column scan processed 484,097 rows. Exactly 6,645 rows
passed the broad gate, and all 896/896 shortlist row IDs were observed. The
broad graph contained 4,094 components: 625 contained at least one shortlist
row and 3,469 formed the predicted-negative population. The frozen 100-family
sample is `../results/kodcode_recall_sample_v1.json`, SHA-256
`d1a97333184f50b831703d93a019e3d251965b3e4f0c4c57287eb3f24fee632c`.

Manual taxonomy review found 3 direct candidates, 1 execution-adapter
candidate, 51 controls and 45 hard rejects; 33 rejects are deferred
multi-operation tasks. Thus the negative sample contains 4 false negatives
and 96 true negatives. Its false-negative rate is 4.0% (Wilson 95%:
1.57--9.84%). The positive sample precision remains 26.0% (Wilson 95%:
18.40--35.37%).

Using the preregistered two-stratum plug-in formula gives estimated recall
54.02%. Combining the two Wilson endpoints as a descriptive sensitivity check
gives 25.27--80.32%. This wide interval, the hash-frozen rather than random
positive prefix, and the mismatch between 627 shortlist families and 625
broad positive components prevent a formal population claim. The result does
not reopen the failed 260-task capacity gate. Review rows and estimates are in
`../results/kodcode_recall_review_v1.json`; no task is fully certified.
