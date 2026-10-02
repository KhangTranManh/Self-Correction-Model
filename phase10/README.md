# Phase 10 — training the judgment step

**Status: closed on 2026-10-01. Negative result; no adapter promoted.** Read
the [final report](docs/FINAL_REPORT.md).

Phase 9 found that the model notices when its two independent attempts
disagree but picks the right one only about half the time. Phase 10 trained a
LoRA judge on the original solver using the model's own verifier-correct
judgments of its own conflicting solutions (808 balanced examples), then
tested it on 300 fresh holdout problems.

## Result in brief

- Picking the right solution when exactly one is right: 52.1% untrained →
  50.7% trained (no change; P1 failed).
- Trained self-check 73.0%, identical to untrained, and 6 points below
  agreement-gated voting at the same cost (P2 failed, Holm p = 0.006).
- Voting over five attempts: 73.67% → 82.67% (+9.0 points), replicating
  Phase 9.

**What improved:** nothing in the trained skill itself. The phase did confirm
voting on a third fresh set, showed that a correct judgment is usually among
four sampled ones (70–87% of one-right pairs) even though the greedy judge
picks it only half the time, and produced 1,621 verified disagreement pairs
with both correct and incorrect judgments — the data a preference objective
would need. See section 5 of the final report.

## Documents and code

- [Preregistration](docs/PREREGISTRATION.md) · [Run record](RUN_STATUS.md)
- `data/sources_v1/` — training pool and holdout; `data/execution_lock_v1.json` — lock
- `scripts/prepare_sources.py` — sources; `scripts/generate.py` — vLLM stages
- `scripts/build_data.py` — pairs and rejection-sampled SFT data
- `scripts/train_judge.py` — LoRA training; `scripts/analyze.py` — single opening
- `scripts/ops/` — pipeline and retry wrapper; `scripts/sync_remote.py` — backup

The trained adapter is served as `phase10-judge` by `../serving/serve.sh`.
The holdout is opened: never train on it or tune against it.
