# Phase 16 engineering results

These files are engineering artifacts. They are excluded from the future
40-episode pilot and protected confirmation set.

- `colab_definition_smoke.json`: CPU-only scorer mechanics on three trusted
  task variants, with no model output.
- `colab_runner_smoke.json`: CPU-only isolated runner mechanics on nine
  trusted tasks, with no model output.
- `p16_e1_audit.jsonl`: append-only Colab T4 A0 engineering run, including
  metadata, stages, all nine raw model completions and their scores; SHA-256
  `11b5f13417fe5af11b353512f1df227c1ff30d7d725417a2c32979c29433e420`.
- `p16_e1_summary.json`: nine completed, nine correct, no boundary errors.
- `p16_e1_rescore.json`: repeat execution of all saved completions in the
  isolated verifier; zero disagreements.

The live partial snapshot used during monitoring was removed after the full
audit was downloaded and verified. See `../docs/ENGINEERING_A0.md` for the
frozen specification and interpretation.
