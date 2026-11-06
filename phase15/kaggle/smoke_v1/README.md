# Phase 15 Kaggle engineering smoke v1

Private, non-evidentiary pipeline check for temporary QLoRA updates on
`Qwen/Qwen2.5-7B-Instruct`. It runs eight synthetic episodes and validates:

- NF4 model loading on a Kaggle T4;
- A0/A1/A2/A3 prompt and update paths;
- two fresh adapter updates per episode;
- exact adapter reset before each update and after each episode;
- append-only completion/update audits;
- source-number separation inside every synthetic episode.

It does not read or create a Phase 15 protected evaluation set. Its scores are
engineering diagnostics and must not be reported as Phase 15 evidence.

## Completed run

Kaggle kernel: `khangkxcp/phase-15-qlora-engineering-smoke-v1`

- Version 1 stopped before model load because `bitsandbytes==0.44.1` was
  incompatible with Kaggle's Python 3.13/CUDA 12.8 image.
- Version 2 used `bitsandbytes==0.50.2` and completed in 606.57 seconds.
- It produced 96/96 completion records and 16/16 adapter-update records.
- All trained adapter hashes differed from initialization, and all eight final
  reset hashes exactly matched the same frozen initialization.
- All outputs were parseable; all 32 control evaluations were correct; no
  output hit the token cap or triggered the loop detector.
- On the 16 judge probes per arm, A0/A1/A2/A3 scored 11/9/8/9. These small
  synthetic counts show pipeline behavior only and provide no efficacy claim.
- Audit SHA-256:
  `adb376c1271209835844403fd0823faa27f8096d2fab3026999332ab95c008f6`.

Local artifacts are under `outputs/phase15_kaggle_smoke_v1/v1/` and `v2/`.
They are ignored by Git.

The smoke script is the run that already completed. The later self-edit gate
lives in `phase15/scripts/self_edit_gate.py` and was checked on the saved
lessons; it was not pushed back into this kernel.

The Kaggle output is `/kaggle/working/phase15_smoke_v1/`. Pull it with:

```powershell
kaggle kernels output khangkxcp/phase-15-qlora-engineering-smoke-v1 `
  -p outputs/phase15_kaggle_smoke_v1
```
