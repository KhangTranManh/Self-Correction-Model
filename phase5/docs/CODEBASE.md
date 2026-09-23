# Phase 5 ownership and execution boundary

Phase 5 has CPU-selected candidates and a completed initial GPU collection.
The local checked archive and final initial-answer evidence are recorded in
`configs/experiments.yaml`; guided reviews and probes have not run. Prefer
calling archived verifier primitives from Phase 1/3/4 without changing closed
phase artifacts.

Ownership:

| Path | Purpose |
|---|---|
| `configs/` | Experiment settings, checkpoint identity and status |
| `data/raw/` and `data/candidates_v1/` | GSM8K source, checked candidate manifest, exclusions and verifier audit |
| `data/initials_v1/` on GPU; `outputs/phase5_remote_v100/` locally | Natural original-solver answers, verifier results and final hashes |
| `data/hints/` | Independent step annotations and exact truthful hint text |
| `data/splits/` | Pending frozen source-disjoint probe and review splits |
| `runs/reviews/` | Every raw answer, hint, contract parse and verification |
| `runs/probes/` | Frozen activations, fits, controls and layer selection |
| `runs/confirmation/` | One protected read after all choices are frozen |
| `scripts/` | Checked candidate/initial collection and validation CLIs; review and probe CLIs pending |

Every generated artifact needs the source manifest hash, model checkpoint hash,
prompt/config hash, seed, software versions and parent artifact hashes. Adapters
are not standalone checkpoints; record their exact parent. Copies of code and
evidence should exist both locally and on the assigned GPU. GPU training is not
part of the initial diagnostic.

The initial collection followed local code and CPU validation, GPU upload,
generation, and immediate local evidence backup. Its as-run BF16 emulation on
the V100 is documented in `docs/GPU_HANDOFF.md`. The next sequence is local
split and hint-feasibility audit, checkpoint/contract freeze, GPU review and
activation extraction, immediate local backup, then independent analysis.
`docs/EXECUTION_PLAN.md` owns the current step order and gates.
