# P16-R0 A0 full CPU contract/oracle review

**Historical snapshot:** subsequent checks found incorrect frozen oracles at
positions 392 and 398. See `A0_CPU_FOLLOWUP_V1.md` for the amended evidence and
current pending gates; the counts below describe the original review only.

## Outcome

The hash-frozen 168-record independent-oracle review queue is exhausted.
Batches 1--36 resolve every record in frozen order:

- 142 records are oracle-ready but retain admission gates.
- 26 records are contract-rejected fail-closed.
- 0 records are deferred or unreviewed.
- 0 corrections are admitted.

Tracker: `../results/a0_cpu_admission_tracker_v14.json`

Tracker SHA-256:
`cd7fc94f01e64b35f5e9277e81cea3f21442c9f56a2cd94c3c330de39e328b94`

Review-chain tip:
`e977129d03252664f1c5724c7777fab58ac71445d7e7f0a94ad4749e54dc09c2`

## Scope and interpretation

The original A0 discovery produced 244 dataset-side failures. Inventory
construction rejected 76 execution, scaffold or nondeterministic records
before independent-oracle review and froze the remaining 168. This review
checked only whether the captured contract supported an independent expected
behavior and whether two trusted implementations agreed on designed witnesses
that catch a local frozen-output mutant. It did not execute recovered model
code or admit corrected implementations.

The 142 oracle-ready records are therefore candidates for later admission,
not training data. Each still requires CPU sandbox reproduction, full semantic
history review, manual cross-history/near-duplicate review and human approval.
Only the first 24 had a separately recorded sandbox closure before this full
review; tracker v14 conservatively records the current review rows themselves
as pending for all 142.

## Final four batches

Batch 33 passes all five records at positions 357, 358, 360, 361 and 364.
Batch 34 passes all five at 366, 373, 374, 377 and 378. Batch 35 passes 381,
387 and 392, and rejects 382 and 395 because no complete contract can be
recovered without unavailable tests or a missing formula. Batch 36 passes the
final four records at 396--399 and hands off to `null`.

Every passing group-G oracle/reference pair agrees on its designed cases and
every associated mutant is caught. The dynamic common-batch test rebuilds all
saved batches 11--36 exactly and verifies their canonical hashes.

## Gate status

Queue exhaustion completes the contract/oracle review milestone only. The
correction pool is not certified; protected data opening and GPU training are
not authorized. The next valid work unit is CPU admission closure over the 142
oracle-ready records, preserving their frozen identities and the existing
hash chain.
