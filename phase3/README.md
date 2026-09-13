# Phase 3 — Error Discrimination and Selective Repair

**Status: CLOSED (2026-09-13).**

Phase 3 tested whether `Kxck/Self_Correction_v1` could reliably route a
previous answer to `KEEP` or `REVISE` before a separate repair step. The phase
is complete. No further data collection, tuning, calibration, or benchmark
selection should be performed under the Phase 3 name.

## Final verdict

Decision-Only V1 remains the final Phase 3 router baseline:

| Router | Frozen BA | KEEP recall | REVISE recall | Code accuracy |
|---|---:|---:|---:|---:|
| Decision-Only V1 | 65.0% | 81.0% | 49.0% | 62.0% |
| DPO Pilot V1 | 66.5% | 85.0% | 48.0% | 62.0% |
| DPO Semantic V2 | 65.5% | 85.0% | 46.0% | 59.0% |
| Decision-Token V2 | 64.0% | 94.0% | 34.0% | — |
| Same-Origin Router V1 | 61.0% | 98.0% | 24.0% | — |
| Verifier-Reward Pilot | 62.0% | 98.0% | 26.0% | — |

The central finding is stable: correctness is partially decodable from frozen
hidden states, but the signal is not converted into a reliable decision policy.
Weight updates repeatedly increase KEEP bias. Platt calibration does not
transfer, and Probe/CAA activation steering changes bias rather than semantic
error detection.

Read [the final report](docs/FINAL_REPORT.md) for the concise conclusion and
[the full results](docs/RESULTS.md) for experiment-level evidence.

## Final artifacts

### Models

- Solver/repair checkpoint: `Kxck/Self_Correction_v1`.
- Final router baseline: `outputs/phase3_decision_only_v1/final_adapter/`.
- All other adapters are historical negative results and are not promoted.

### Data

- Decision-Only V1: 240 rows, source-disjoint 168/36/36 train/dev/test.
- Frozen behavior benchmark: 200 protected rows.
- Representation probe: 256 protected probe rows.
- Router Recovery V2: 108 verified same-origin pairs / 216 behavior rows;
  preserved as incomplete data, never trained because the 200-pair gate failed.

The authoritative data inventory, lifecycle labels, row counts, and hashes are
in [data/CATALOG.md](data/CATALOG.md) and `data/catalog.json`.

## Closed codebase

```text
phase3/
├── configs/       immutable experiment configs and final registry
├── data/          canonical, protected, historical, and incomplete datasets
├── docs/          final report, detailed results, and codebase contract
├── lib/           verifier and provenance primitives
├── runs/          immutable evaluation evidence
└── scripts/
    ├── data/       dataset reproduction
    ├── training/   historical training reproduction
    ├── evaluation/ frozen evaluation and diagnostics
    ├── serving/    historical adapter serving
    └── maintenance/closed-artifact catalog and validation
```

The code remains runnable for reproduction, but its presence does not reopen
the phase. See [docs/CODEBASE.md](docs/CODEBASE.md).

## Reproduction and audit

Run from the repository root:

```bash
python phase3/scripts/maintenance/build_data_catalog.py
python phase3/scripts/validate_local_state.py --hash-adapters
```

Serving the final baseline for comparison:

```bash
PHASE3_LORA_MODULES="phase3-decision-only-v1=/root/agi/outputs/phase3_decision_only_v1/final_adapter" \
bash phase3/scripts/serving/serve_phase3.sh
```

## Immutable rules

1. Frozen evaluation and probe rows never enter training or sample selection.
2. Correctness labels come from deterministic verifiers, never an LLM judge.
3. Raw attempts and scored per-row evidence remain immutable.
4. `.env`, SSH credentials, and provider secrets are outside the audit surface.
5. Any continuation creates a new phase, new registry entry, and new holdouts.
