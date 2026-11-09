# P16-R0 A0 CPU admission tracker v1 (2026-11-09)

Status: **CPU tracker active; zero corrections admitted**.

This tracker measures progress through the frozen 168-record independent-
oracle queue without changing the inventory or any review decision. It is
model-free and GPU-free. It does not execute recovered model code, construct
task-specific oracles, perform human review, certify a correction pool, or
authorize training.

## Verified inputs

`../scripts/track_a0_cpu_admission.py` verifies before aggregating:

- the canonical inventory hash and all 244 per-record hashes;
- the frozen 168 pending / 76 pre-review-rejected accounting;
- every review artifact's canonical self-hash;
- the ordered cross-batch hash chain;
- review identity against the frozen inventory;
- that a position is re-reviewed only after an explicit deferral; and
- that the current v1 review chain admits no correction.

The output contains identities, gate states, counts, hashes and descriptive
projections only. It contains no prompt text, published test values, expected
values, reference solution or recovered model code.

## Current funnel

The five review batches resolve 14 unique frozen positions:

| Stage | Count |
|---|---:|
| A0 dataset-side failures | 244 |
| Rejected before oracle review | 76 |
| Frozen independent-oracle queue | 168 |
| Unique positions reviewed | 14 |
| Oracle-ready, remaining gates pending | 11 |
| Contract-rejected | 3 |
| Unreviewed queue | 154 |
| Sandbox pending among oracle-ready | 11 |
| Full semantic-history review pending | 11 |
| Manual cross-history review pending | 11 |
| Human approval pending | 11 |
| Corrections admitted | 0 |

The next frozen position is 37. The tracker does not convert oracle-ready
records into admitted corrections.

## Capacity projection

The observed oracle-ready rate among 14 resolved contract/oracle reviews is
11/14, or 0.7857. Its Wilson 95% interval is 0.5241--0.9243. Descriptively,
64 oracle-ready records would require approximately 82 total reviews at the
point rate, 123 at the interval's lower bound, or 70 at its upper bound.

These are small-sample planning estimates, not an admission forecast. There
are zero completed admission decisions, so the tracker intentionally reports
the admission rate as unavailable. The target remains 64 **admitted** residual
corrections, not 64 oracle-ready records.

Milestones are frozen at 24, 48, 80, 120 and 168 unique reviewed positions.
Review remains in original A0 position order; quantity may increase only by
reviewing more frozen records, never by lowering a gate or relabelling a
rejection.

## Reproduction

From the repository root:

```powershell
python -B phase16/scripts/track_a0_cpu_admission.py `
  --inventory phase16/results/a0_admission_inventory_v1.json `
  --reviews `
    phase16/results/a0_cpu_review_batch1_v1.json `
    phase16/results/a0_cpu_review_batch2_v1.json `
    phase16/results/a0_cpu_review_batch3_v1.json `
    phase16/results/a0_cpu_review_batch4_v1.json `
    phase16/results/a0_cpu_review_batch5_v1.json `
  --output phase16/results/a0_cpu_admission_tracker_reproduction.json
```

The writer refuses to overwrite an existing output. Compare the reproduced
`tracker_sha256` with the tracked artifact, then remove only the explicitly
created reproduction file if it is no longer needed.

## Next CPU work

1. Close sandbox, recovery-residual, full-history, semantic-near-duplicate and
   human gates for the eleven oracle-ready records.
2. Continue task-specific contract/oracle review from position 37 in the
   frozen order.
3. Append each new hash-chained batch to the tracker input list and regenerate
   to a new tracker version; never overwrite this v1 artifact.
4. Estimate admission yield only after at least one oracle-ready record has a
   completed admission decision.

