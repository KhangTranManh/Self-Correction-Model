# AGI Behavior-First Research

This repository studies whether a small language model can detect and correct
its own errors under objective supervision. Math answers are checked
symbolically and code answers are executed against tests; an LLM is never used
as the correctness oracle.

## Current status (2026-09-13)

Phase 3 closed on 2026-09-13. Its central result is negative but informative:
preference tuning improved KEEP behavior and reduced harmful revisions, but it
did not improve detection of plausible wrong answers.

| Router | Balanced accuracy | KEEP recall | REVISE recall | Code accuracy | Unseen-template accuracy |
|---|---:|---:|---:|---:|---:|
| Decision-Only V1 | 65.0% | 81.0% | 49.0% | 62.0% | 58.0% |
| DPO Pilot V1 | 66.5% | 85.0% | 48.0% | 62.0% | 55.5% |
| DPO Semantic V2 | 65.5% | 85.0% | 46.0% | 59.0% | 53.5% |

The representation probe reinforces this conclusion. Decision-Only V1 reached
67.3% balanced accuracy on the frozen probe test, while DPO Semantic V2 reached
57.7%; probe REVISE recall fell from 53.8% to 38.5%.

A later frozen activation-steering diagnostic also failed to turn the decoded
probe direction into better semantic decisions. Probe and CAA residual
directions were tested at layers 14, 21, and 28 without updating weights. The
only apparent accuracy gain shifted predictions toward KEEP while reducing
REVISE recall, so activation steering is not promoted.

Therefore:

- `Decision-Only V1` remains the canonical Phase 3 router.
- `Kxck/Self_Correction_v1` remains the solver/repair checkpoint.
- DPO Pilot V1 and DPO Semantic V2 are retained as research artifacts, not
  promoted models.
- No additional experiment is authorized inside Phase 3. Any continuation must
  open a new phase with new holdouts and a preregistered hypothesis.

See the [Phase 3 final report](phase3/docs/FINAL_REPORT.md),
[Phase 3 README](phase3/README.md), and
[Phase 3 results](phase3/docs/RESULTS.md).

## Research phases

### Phase 1 — verified self-correction

Phase 1 built the verified critique-and-correct pipeline and produced
`Kxck/Self_Correction_v1`. Guided correction worked, especially for code, but
autonomous review and resistance to false feedback remained weak. See
[phase1/README.md](phase1/README.md).

### Phase 2 — preference learning exploration

Phase 2 constructed KTO data from verifier-labelled behavior and validated the
training path. A full KTO checkpoint was not completed. This phase is historical
and is not the active training path. See [phase2/README.md](phase2/README.md).

### Phase 3 — error discrimination and selective repair

Phase 3 separated the decision `KEEP`/`REVISE` from answer repair, evaluated the
router on a frozen 200-row benchmark, and probed hidden representations. It is
now a closed, reproducible research package; Decision-Only V1 is its final
router baseline.

## Non-negotiable data rules

1. Correctness comes from a deterministic verifier, not an LLM judge.
2. Wrong answers must be natural model failures; synthetic wrong answers are not
   accepted for canonical training.
3. Frozen evaluation sources never enter training or dataset selection.
4. KEEP and REVISE must share the same neutral-review template distribution.
5. Model-visible rationales may be stored for audit, but they must not be called
   hidden chain-of-thought.
6. Every promoted adapter must have a reproducible config, dataset hash, frozen
   evaluation, and local backup.

## Repository layout

```text
AGI/
├── README.md                    # Project status and phase routing
├── instructionAI/              # Cross-phase architecture and data rules
├── phase1/                     # Verified correction pipeline and history
├── phase2/                     # KTO exploration and data
├── phase3/                     # Active router/selective-repair codebase
│   ├── configs/                # Training configs and experiment registry
│   ├── data/                   # Sources, immutable attempts, built datasets
│   ├── docs/                   # Codebase and results documentation
│   ├── lib/                    # Shared provenance and verifier utilities
│   ├── runs/                   # Frozen evaluations and human-readable reports
│   └── scripts/                # Data, training, evaluation, and serving CLIs
└── outputs/                    # Local adapters and runtime artifacts (gitignored)
```

The canonical Phase 3 structure and ownership rules are defined in
[phase3/docs/CODEBASE.md](phase3/docs/CODEBASE.md). Model status is machine-readable
in [phase3/configs/experiments.yaml](phase3/configs/experiments.yaml).

## Quick local audit

From the repository root:

```bash
python phase3/scripts/validate_local_state.py
python phase3/scripts/validate_local_state.py --hash-adapters
```

The audit checks canonical datasets, reports, row counts, adapter presence, and
optionally the recorded adapter hashes. It does not read `.env`.

## Serving

The general vLLM launcher is `phase3/scripts/serving/serve_phase3.sh`. It binds
to localhost by default. Example:

```bash
PHASE3_LORA_MODULES="phase3-decision-only-v1=/root/agi/outputs/phase3_decision_only_v1/final_adapter" \
bash phase3/scripts/serving/serve_phase3.sh
```

Only set `PHASE3_HOST=0.0.0.0` on a network you intend to expose. The launcher
does not add authentication by itself.

## Secrets and large artifacts

- `.env` is local-only and must never be read for documentation work or committed.
- `outputs/` is gitignored; adapters must be backed up separately.
- Raw attempts and frozen result JSONL files are research evidence. Do not
  rewrite or delete them during documentation cleanup.
