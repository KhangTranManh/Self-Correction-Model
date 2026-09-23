# Phase 5 ownership and execution boundary

Phase 5 is a local design package. No scripts, datasets, model outputs or GPU
runs exist yet. Add them only as their input contracts and validation checks
are implemented. Prefer calling archived verifier primitives from Phase 3/4
without changing the closed phase artifacts.

Planned ownership:

| Path | Purpose |
|---|---|
| `configs/` | Frozen experiment settings, checkpoint identity and status |
| `data/source/` | New eligible source manifest and exclusion audit |
| `data/initials/` | Natural original-solver answers and fresh verification |
| `data/hints/` | Independent step annotations and exact truthful hint text |
| `data/splits/` | Frozen source-disjoint probe and review splits |
| `runs/reviews/` | Every raw answer, hint, contract parse and verification |
| `runs/probes/` | Frozen activations, fits, controls and layer selection |
| `runs/confirmation/` | One protected read after all choices are frozen |
| `scripts/` | Local checked collection, validation, review and probe CLIs |

Every generated artifact needs the source manifest hash, model checkpoint hash,
prompt/config hash, seed, software versions and parent artifact hashes. Adapters
are not standalone checkpoints; record their exact parent. Copies of code and
evidence should exist both locally and on the assigned GPU. GPU training is not
part of the initial diagnostic.

The sequence is local code and CPU validation, GPU upload, generation and
activation extraction, immediate local evidence backup, then independent
analysis. The GPU address and execution commands remain unset until assigned.
