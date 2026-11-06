# Phase 16 four-source mix: CPU F0 candidate inventory (2026-11-05)

**Decision: FAIL the current C-upper-inclusive pool gate; no GPU screen.**
This is a reproducible *candidate inventory*, not a certified study dataset.
The audit script is `../scripts/audit_four_source_mix.py`; the unified
per-task metadata manifest and source-file hashes are in
`../results/four_source_candidate_audit_v2.json`. It contains no model outputs,
does not execute dataset code, assigns no study roles, and does not open a
protected evaluation split. Only the local source shards listed below were
read. The release/version and raw-file hash are recorded for every source.

| Source | Snapshot inspected | Rows | Preliminary candidates for current function-call C branch | All four C text flags |
|---|---:|---:|---:|---:|
| TACO train | pinned, all 9 shards | 25,443 | 2,518 call-based metadata candidates | 15 among those candidates |
| CodeContests train | pinned, 3 of 39 shards | 1,026 | 302 metadata candidates after exact TACO overlap | 15 among those candidates |
| EvalPlus HumanEval+ | pinned, complete | 164 | 0 automatically admissible: all 164 HumanEval tasks previously used in Phase 1 | 0 total |
| EvalPlus MBPP+ | pinned, complete | 378 | 0 automatically admissible until task-ID history audit | 0 total |
| LiveCodeBench lite release_v1 | pinned, first of 3 shards | 134 | 0 automatically admissible: source and verifier audit pending | 3 total |

The unified inventory has **27,145 rows and no duplicate manifest IDs**.
CodeContests has 683 rows with an exact normalized-description match to
TACO. The 302 CodeContests and 2,518 TACO preliminary candidates are **not**
2,820 eligible tasks: they have not passed contract, family, near-duplicate,
historical-source, oracle or sandbox checks. The 30 strict text hits among
those candidates are likewise not 30 eligible tasks. Regex flags are neither
necessary nor sufficient for the semantic label; their count is not an
upper bound on possible eligible tasks.

The C-upper-inclusive definition in `PROTOCOL.md` requires counting,
summing or selecting individual elements with a key in an inclusive `[L,R]`,
so adding one upper-bound element changes the oracle by one count, its
nonzero value, or one selected ID. A source problem involving a segment,
subarray, calendar date, or upper bound in a different operation does not
automatically meet this contract. A spot read of the 15 CodeContests and
three LiveCodeBench text hits found such semantic mismatches. For example,
the LiveCodeBench bounded-submultiset problem counts *subsets*; adding one
input element can change the answer by more than one, so it cannot use the
current `score_boundary.py` count-pair invariant. It remains an unreviewed
candidate for a different rule, not a certified C task.

## What each source can contribute under the current protocol

- TACO: potential function-call teaching/screen/probe tasks **only after**
  contract/oracle review. Its standard-I/O rows are outside the current
  function-call C scorer, though a separately preregistered verifier could
  revisit them.
- CodeContests: potential **standard-I/O** code tasks and controls after
  sandbox and independent reference validation. Its incorrect submissions
  are not the study model's natural A0 errors. The local 3-shard check is not
  a sample of all 39 shards.
- EvalPlus: augmented tests for HumanEval/MBPP tasks and potential regression
  diagnostics. It does not introduce new base problem IDs. HumanEval was
  fully used in Phase 1; MBPP ID-level reuse must be resolved before any
  role assignment. EvalPlus tests must not be blindly executed on Windows.
- LiveCodeBench: dated, versioned independent source/confirmation candidate.
  Only 134/400 release_v1 rows were opened for schema preflight; this is not
  a full inventory. The inspected shard has 125 LeetCode-style call-based
  and nine Codeforces-style standard-I/O tasks, determined from `func_name`
  metadata; each item needs an I/O-specific runner. Its dataset and original
  contest-platform usage terms need review before redistributing a merged
  dataset; a repository code license alone does not settle data rights.

## Requirement-by-requirement release check

| Requirement from S0/CPU gate | Current state |
|---|---|
| 300 unique source tasks in one feasible domain, 160/40/20/20/60 frozen roles | **Fail**: zero certified tasks; all roles unassigned |
| At least 200 screen/probe tasks where the same named rule applies | **Unknown/unsupported**: 30 preliminary strict text hits in TACO + CodeContests; no task certified |
| Historical, cross-source exact and near-duplicate/family exclusion | **Partial**: TACO history and cross-source exact text checked; near duplicates, MBPP IDs and contest families unresolved |
| Independent oracle and paired boundary tests | **Fail**: none built for four-source tasks |
| Safe code execution with appropriate I/O mode | **Fail**: no integrated sandboxed four-source runner |
| Natural A0 named-error prevalence | **Not run**; may only follow a frozen eligible pool |

The best next CPU action is **not** to concatenate more rows. First manually
review the 30 strict TACO/CodeContests candidates plus a preregistered sample
of the broader TACO/CodeContests standard-I/O candidates, record precise
rejection reasons, and estimate whether the exact named mechanism has any
credible capacity. If it does, expand the source inventory and build the
stdlib/sandboxed oracle. If it does not, stop this C branch and preregister
a different transferable rule with its own verifier and sample-size target.
Do not relabel a generic wrong answer as a C-upper-inclusive error.

Sources: [TACO dataset card](https://huggingface.co/datasets/BAAI/TACO),
[CodeContests repository](https://github.com/google-deepmind/code_contests),
[EvalPlus repository](https://github.com/evalplus/evalplus), and
[LiveCodeBench repository](https://github.com/LiveCodeBench/LiveCodeBench).
