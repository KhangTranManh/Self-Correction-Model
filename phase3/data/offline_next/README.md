# Phase 3 offline preparation

Everything in this directory was produced locally without loading an LLM or
using a GPU. No training has started.

## Ready now

- `classifier_manifest.jsonl`: 256 balanced frozen-probe examples (128
  KEEP/128 REVISE; 128 code/128 math) linked by row index to the saved
  Decision-Only V1 and Semantic V2 activation arrays. This is ready for a
  class-weighted linear classifier or small focal-loss MLP on CPU.
- `decision_token_train.jsonl`: 160 rows from 80 problem groups.
- `decision_token_dev.jsonl`: 40 rows from 20 disjoint problem groups.
  Every problem contributes one KEEP and one REVISE row. A later trainer must
  mask all loss except the decision output tokens.
- `hard_revise_existing.jsonl`: all 100 verified plausible-wrong cases ranked
  by an explicit hardness score, including 45 near-miss math and 6 code cases
  that pass part of the test suite.
- `manual_shortcut_audit_queue.jsonl`: 40 full correct/wrong pairs with empty
  human-review fields.
- `shortcut_audit.json`: grouped 5-fold CPU diagnostics for answer text,
  surface features, and model origin.

## Prepared for the next GPU session

`future_same_origin_generation_manifest.jsonl` contains 100 exact generation
tasks (762 requested samples). Each task samples from the model that already
gave the verified-correct answer, looking for a plausible verified-wrong answer
to form a truly same-model-origin contrastive pair.

Run generation and fresh verification only after renting a GPU. First process
the rows marked `priority: pilot`; do not accept formatting-only, empty,
compile-only, or reference-derived wrong answers.

## Rebuild

```powershell
python phase3/scripts/data/prepare_offline_next.py
```

The command is deterministic and rewrites only this directory. See
`offline_preparation_summary.json` for counts, validation, hashes, and shortcut
risk estimates.
