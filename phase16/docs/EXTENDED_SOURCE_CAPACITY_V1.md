# Phase 16 extended code-source capacity audit (2026-11-05)

**Decision: candidate capacity is plausible only if KodCode is admitted as a
synthetic development source. The GPU release gate remains closed.** This is
a CPU text/provenance audit. It does not certify tests, execute dataset code,
assign study roles, estimate the study model's error prevalence, or open a
protected confirmation set.

## Measured inventories

### LeetCodeDataset

Pinned revision `215604aeed660029df7de2fea5a4d7b6ed476a08` was downloaded
completely: 2,869 unique tasks (2,641 train, 228 temporal test). All tasks
have function entry points and at least five declared examples/assertions in
this snapshot. The source-text audit found 63 four-flag C candidates: 57
train and six test; one has an exact normalized-text match in the earlier
four-source inventory. Reading these candidates shows many semantic false
positives (subarrays, paths, interval scheduling and derived range queries).
Thus **63 is a review ceiling, not an eligible count**. The temporal test
rows should be reserved rather than used to repair development capacity.

Artifact: `../results/leetcodedataset_text_audit.json`. Source revision and
file hashes are recorded by `../scripts/audit_extended_candidate_sources.py`.

### KodCode-V1

Pinned revision `97d81f0695ddd7c5c02bfe16436db9a17a4a21e5` was scanned through
projected columns without executing its generated tests. Counts:

- 484,097 total train rows;
- 6,938 rows pass the common four text flags;
- 6,864 pass the additional element-operation/entity text flags;
- 6,645 remain after excluding `benchmark_similarity >= 0.8`;
- 4,094 connected seed-family components among those 6,645 rows;
- 1,101 components remain if source-derived subsets named TACO,
  CodeContests, APPS, Codeforces and LeetCode are excluded.

A stricter semantic-text heuristic rejects obvious subarray/path/grid/query,
optimization, random-generation and downstream-statistic tasks. It leaves
470 rows in 323 connected seed components under the strict subset exclusion.
If source-derived KodCode subsets are retained, it leaves 896 rows in 627
connected seed components. Exact text and `seed_ids` are both used; a
different generated prompt sharing a seed remains the same connected family.

The first 50 rows in a precomputed SHA-256 review order were manually checked
against the exact contract. Twenty-four satisfy the **text contract only**;
26 were rejected for incidental range language, ambiguous/malformed target,
wrong output operation, or an oracle contradiction. Zero of the 24 has yet
passed independent test/oracle validation. The observed 24/50 rate has a
descriptive Wilson 95% interval of approximately 0.348--0.615. It is not a
formal population estimate because the review sampled rows, not one row per
connected component, and only the strict-subset shortlist was reviewed.

This diagnostic is nevertheless useful for capacity:

- 323 strict-subset components times that interval gives roughly 112--199,
  so the source-independent KodCode branch is not a credible route to 200
  applicable tasks by itself;
- 627 all-subset components gives roughly 218--386, so a **synthetic
  development pool** may have enough capacity, subject to full seed-history,
  contract, test and oracle audit.

Artifacts: `../results/kodcode_text_audit_v2.json`,
`../results/kodcode_c_semantic_shortlist.json`, and
`../results/kodcode_manual_contract_review_50.json`. KodCode-V1 is licensed
CC BY-NC 4.0, so it is unsuitable for an unrestricted commercial derivative
without separate permission.

### Other proposed sources

- OpenCodeReasoning has 28,319 unique questions but explicitly aggregates
  TACO, APPS, CodeContests and open-r1/Codeforces. Its many reasoning samples
  do not create independent source problems.
- rStar-Coder reports 418K problems, but much is synthetic from seed tasks;
  testcase shards are extremely large. It is useful as a candidate generator
  or cross-check, not an automatic independent pool.
- open-r1/Codeforces reports 8,760 verifiable problems and explicit aliases,
  but it is standard-I/O and generated tests are about 110 GB. A projected
  local scan was stopped after excessive local memory/startup without a
  completed result. It needs a metadata-only export or isolated cloud audit.
- Project CodeNet has 4,053 problems and many judged submissions, but does
  not supply the dense public test oracle required here.
- Full LiveCodeBench release_v6 has 1,055 tasks according to its official
  release table. It is best kept versioned and sealed for confirmation; its
  rows still require per-task contract review.
- APPS is already represented in TACO/KodCode and earlier phases, so it adds
  volume but little independent provenance.

## Recommended source composition

For an exploratory **signal-only** development run, the viable candidate
composition is:

1. KodCode semantic shortlist, one representative per connected seed-family,
   including source-derived subsets but excluding any seed mapped to Phase
   1--15 usage;
2. natural TACO/CodeContests/LeetCodeDataset candidates for diversity and
   oracle cross-check, not to inflate the family count;
3. LiveCodeBench temporal tasks held out for a later confirmation run;
4. near-negative and unrelated controls selected independently of A0 output.

This requires a protocol amendment stating that the development evidence is
on a synthetic/source-derived distribution. It cannot be described as a
natural-code population result. Before GPU, review at least one row from all
candidate families needed for the fixed roles, build independent oracles,
and demonstrate that the inclusive implementation passes while an exclusive
upper-bound mutant produces the two-pair signature.

## Controlled generation fallback

If fewer than 200 applicable families survive, a different model may generate
**task wording and scaffolds only**. It must not be trusted to generate the
gold answer or decide eligibility. The safe construction is:

1. sample a formal specification from predeclared operation (`count`, `sum`,
   `select`), collection structure, key type, predicate and return schema;
2. generate the oracle and boundary/metamorphic cases programmatically from
   that formal specification;
3. ask a model different from the Phase 16 study model to produce multiple
   problem statements/scaffolds without seeing study outputs;
4. parse each statement back into the formal schema and reject disagreement;
5. require two independent reference implementations, correct/exclusive
   mutation checks, sandbox execution and blinded human contract review;
6. cluster by generator prompt, formal schema, seed and text similarity;
   keep whole families in one role and cap the number per family;
7. label all results `controlled-synthetic`; reserve natural tasks for
   confirmation.

Model generation can increase surface diversity, but it cannot manufacture
independent evidence merely by paraphrasing or changing constants. No such
generated task has been created in this audit.

Primary sources: [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset),
[KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1),
[rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder),
[OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning),
[open-r1/Codeforces](https://huggingface.co/datasets/open-r1/codeforces),
[Project CodeNet](https://github.com/IBM/Project_CodeNet), and
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).
