# Phase 3 scripts

Phase 3 is closed and these entry points are maintenance/reproduction code.

| Directory | Purpose |
|---|---|
| `data/` | Rebuild immutable sources, attempts, and historical datasets |
| `training/` | Reproduce completed or rejected GPU experiments |
| `evaluation/` | Reproduce frozen behavior, probes, calibration, and steering diagnostics |
| `serving/` | Serve the canonical Decision-Only V1 adapter for historical comparison |
| `maintenance/` | Validate and catalog the closed artifact set |

Canonical entry points are documented in `../docs/CODEBASE.md`. Scripts for
rejected or incomplete experiments remain because their data and reports must
be auditable; their presence does not authorize a new run.
