# Phase 12 run record

2026-10-06, local CPU (no GPU):

- Phase 11 committed (`b160208`) before Phase 12 work began.
- New evaluation dataset: SVAMP downloaded from the authors' repository
  (`github.com/arkilpatel/SVAMP`, `SVAMP.json`, SHA-256 `5be77703…`, 1,000
  problems, all integer answers). The exclusion scan found 40 problems already
  used in Phase 3's frozen evaluation (`phase3/data/two_stage_selective_repair/
  frozen_eval.jsonl` and its run copies) and excluded them; 960 remain, and all
  references passed the Phase 1 verifier (`data/sources_v1/`).
- Order-swapped DPO data built from Phase 10 training-pool judgments with
  Python 3.10.11 + SymPy 1.14.0: 1,822 pairs (1,592 training, 230
  validation). The A/B relabeling was spot-checked: in a swapped pair the same
  faulty step reads "Solution B", matching the swapped prompt.
- The analysis was tested on synthetic data with hand-checked expected
  values (both-orders verdicts, abstain fallback, bias table, invented
  answers) before freezing.
- Preregistration written and execution lock frozen
  (`data/execution_lock_v1.json`, 24 files, `cc49da15…`).
- Waiting for a GPU. On the new host: `scripts/ops/bootstrap.sh`, a short DPO
  smoke test, then `scripts/ops/supervise.sh`.
- 2026-10-08: endpoint table revised before any Phase 12 generation — P1
  becomes accuracy of consistent both-orders verdicts with a coverage floor of
  30%; P2 becomes non-inferiority (lower 95% bound > −2 points) against
  compute-matched vote@k; a single-order Secondary and diagnostic thresholds
  (position-2 rate 40–60%, invented answers < 5%) were added. `analyze.py` was
  rewritten and retested on synthetic data; the lock was refrozen
  (`4f42f926…`).

2026-10-08, remote GPU (Tesla V100-SXM2-32GB, 94 GB RAM, Ubuntu 22.04,
Python 3.10.12; project root `/root/AGI_phase12/`):

- `scripts/ops/bootstrap.sh` completed; the lock verified on the host
  (`4f42f926…`).
- DPO smoke test (temporary script and output; four longest training pairs,
  3,889–4,697 tokens, including a swapped pair): peak 18.6 GB, 3.6 s per pair;
  margins rose from 0 on every step with no skipped updates.
- 16:13 UTC: pipeline attempt 1 started under `scripts/ops/supervise.sh`.
- Stage times (UTC): samples 16:13–16:38 (4,800); untrained judge
  16:38–16:47 (501 pairs × 2 orders); DPO training 16:47–17:56 (reference pass
  ~17 min, 100 steps; validation preference accuracy 76.1% at step 50, 77.8%
  at step 100); DPO judge 17:56–18:12; protected analysis 18:12.
  `PHASE12_PIPELINE_COMPLETE` at 18:12:56 in one attempt.
- All 15 result files match the GPU copies by SHA-256.
- 18:13 UTC: `serving/serve.sh original` started on the host (now also loading
  `phase12-dpo-judge`); the client demo ran `single`, `vote`, and
  `self_check` through the server.
- Closed on 2026-10-09; see `docs/FINAL_REPORT.md`.
