# Phase 5 data status

The CPU source inventory and first 1,200 candidate problems are selected in
`candidates_v1/`. This excludes all recorded Phase 1-4 source/evaluation
questions, including unopened Phase 4 confirmation candidates. All candidate
references passed the project math verifier under CPython 3.10.
The original GSM8K train rows are in `raw/gsm8k_train.jsonl`; reference steps
are stored separately and must never enter model-visible prompts.
See `CANDIDATE_REPORT.md` for exact counts and hashes.

Natural original-solver answers, model-answer step annotations, hint text and
frozen source splits remain to be collected after the GPU is assigned. The
prospective 240/80/160 split is a maximum and feasibility target, not a claim
that enough eligible problems exist. Raw generations and test outcomes cannot
be used to rewrite eligibility or prompt rules.
