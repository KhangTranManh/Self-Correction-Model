# Phase 5 data status

The CPU source inventory and first 1,200 candidate problems are selected in
`candidates_v1/`. This excludes all recorded Phase 1-4 source/evaluation
questions, including unopened Phase 4 confirmation candidates. All candidate
references passed the project math verifier under CPython 3.10.
The original GSM8K train rows are in `raw/gsm8k_train.jsonl`; reference steps
are stored separately and must never enter model-visible prompts.
See `CANDIDATE_REPORT.md` for exact counts and hashes.

The original solver generated 815 natural answers on a V100: 575 correct and
240 wrong. The checked local backup is `../../outputs/phase5_remote_v100/`,
with its hashes and as-run precision in `../configs/experiments.yaml`. A
balanced 480-source train/development/protected split is now feasible but has
not been frozen. Model-answer step annotations, verified hint text, guided
reviews, and probe outcomes do not yet exist. Raw generations and protected
outcomes cannot be used to rewrite eligibility or prompt rules.
