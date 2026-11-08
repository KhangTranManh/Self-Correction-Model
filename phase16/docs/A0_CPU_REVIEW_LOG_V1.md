# P16-R0 A0 CPU review log v1 (2026-11-08)

Status: **review begun in frozen order; zero corrections admitted**.

The first review batch examines A0 positions 0, 2 and 6, the first three
records marked `pending_independent_oracle_review`. It uses task contracts but
does not use published expected values or dataset reference solutions to
derive a verdict. The frozen admission inventory is not modified.

## Batch 1 decisions

| Position | Item | Review result |
|---:|---|---|
| 0 | `taco_train_14050` / `remove_parentheses` | Contract passes. Two independently implemented trusted CPU oracles agree on eight contract-derived cases. A boolean nesting mutant is caught on nested-parentheses cases. Actual model execution, full semantic history, cross-history near-duplicate review and human approval remain pending, so the item is not admitted. |
| 2 | `taco_train_18158` / `order` | Rejected fail-closed at contract review. Exact scheduling, resource overlap and completion semantics are delegated to unavailable example tests; using published tests to infer them would violate the independent-oracle boundary. |
| 6 | `taco_train_24349` / `transpose` | Contract passes, but a dedicated grid oracle is still required to preserve simultaneous-note alignment while fret widths expand or contract. Review stops here before later A0 positions. |

For position 0, the frozen candidate bank already records a singleton
near-duplicate component. A fresh bank-local token 5-gram comparison found a
maximum similarity below 0.011. This is useful automated evidence, not a
substitute for the required cross-history semantic and human review.

The reproducible artifact is
`../results/a0_cpu_review_batch1_v1.json`, built by
`../scripts/audit_a0_cpu_review_batch1.py`. It explicitly records that no
untrusted model code was executed locally and no correction was admitted.

## Next gate

Complete the specialized independent oracle for position 6 before reviewing
position 10. Position 0 additionally needs sandbox execution of the exact
hash-frozen recovered output, full history and semantic-near-duplicate review,
and human approval. Training, protected opening and effect claims remain
unauthorized.
