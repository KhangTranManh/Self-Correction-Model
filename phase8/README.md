# Phase 8 — reproducibility, distractor, and probe-routed blind re-solve

The CPU source preparation and Phase 7 as-run protocol audit are complete.
The preregistration is in `docs/PREREGISTRATION.md`. First-pass generation was
started on the new RTX 3090 host; it must be treated as incomplete until its
append-only audit and summary report confirm all 400 rows. No Phase 8 result
or pipeline accuracy is available yet.

`data/fresh_source_pool_v1/candidate_problems.jsonl` contains 400 fresh
verifier-checked GSM8K training questions. The manifest SHA-256 is recorded in
`candidate_report.json`. The selection excluded all recorded Phase 1–4 sources,
the entire Phase 5 candidate manifest, the Phase 6 source pools, and all 600
Phase 7 candidate sources. Selection was deterministic and happened before any
new initial answer or probe score existed. All 400 sources are reserved as a
new protected test pool, with no tuning on these rows. Reference answers are
evaluation data; never place them in model-visible prompts.

The pool has no initial-correctness labels yet. Freeze the complete router
policy before generating blind outputs. Report natural correctness prevalence;
do not select a post hoc balanced subset as the primary accuracy measure.

The exact Phase 5 probe models and activation files are absent locally; see
`../phase5/data/probe_recovery_audit.json`. The selected historical probe
settings are in `../phase5/data/protocol/protected_opening_v1_lock.json`:
layer 14, C=0.01 for original/V2, and layer 14, C=1.0 for V3, with threshold
0.5. Settings alone do not reproduce the learned coefficients. Search for and
hash-check the original `.joblib` files before calling a probe frozen Phase 5.

If the exact artifacts cannot be recovered, re-extract train and development
activations with `phase5/scripts/extract_prehint_activations.py`, then fit with
`phase8/scripts/fit_locked_probe.py`. Use the frozen Phase 5 train/development
rows only, keep the three checkpoint lineages and input template unchanged,
and save hashes for every resulting model. Label the result a **newly rebuilt
probe**. Do not use Phase 5 protected, Phase 6 holdout, Phase 7 data or this
new source pool to choose layer, C, threshold, prompts or weights. Evaluate the
new probe on this protected pool only after those choices lock.
