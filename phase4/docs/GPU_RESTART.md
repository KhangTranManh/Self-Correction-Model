# Restart the interrupted preference stage

Phase 4 closed on 2026-09-15. The commands below are historical recovery
instructions, not authorization to restart training. Completed artifacts are
archived; see `FINAL_REPORT.md` and `CODEBASE.md`.

The selected policy remains warm-start V2. Correction SFT V3 is backed up under
`outputs/phase4_correction_sft_v3`, with its 74% development result under
`phase4/runs/expansion_v1`. The interrupted old-GPU DPO run had no local final
adapter. A fresh run on the replacement GPU completed; its final adapter and
resumable checkpoints are now backed up in `outputs/phase4_preference_dpo_v1`.
It did not improve behavioral accuracy; see `PILOT_LOG.md`.

On a replacement GPU, bootstrap the existing training environment and upload
the project code, preference data, and selected V2 adapter. Reconstruct the
selected V2 merged checkpoint at `/root/models/phase4_warmstart_v2_merged` using
the original `Kxck/Self_Correction_v1` checkpoint and the V2 adapter; do not use
the original Qwen checkpoint directly, because the selected policy includes
the earlier self-correction training.

Start a fresh run from `/root/agi`:

```bash
/root/venv-train/bin/python -u phase4/scripts/training/train_preference_dpo.py \
  --config phase4/configs/preference_dpo_v1.yaml
```

If a complete resumable checkpoint has been recovered, add:

```bash
--resume-from-checkpoint outputs/phase4_preference_dpo_v1/checkpoint-75
```

The revised configuration saves every 25 steps and retains two checkpoints.
Copy checkpoints, logs, and final artifacts to the workstation during the run.
After completion, inspect nonzero LoRA weights, compare generated reviews on
the frozen Cycle 0 development problems with reused initial answers, and copy
all results locally. Confirmation candidates remain unopened.
