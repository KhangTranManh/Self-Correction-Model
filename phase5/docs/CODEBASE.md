# Phase 5 ownership and execution boundary

Phase 5 has completed candidate selection, initial collection, guided reviews,
probe controls, and one protected opening. The frozen pre-run registry remains
unchanged so its protocol-lock hash stays valid; post-run status lives in
`data/results_v1.json`, `docs/FINAL_REPORT.md`, and the opening receipt. Prefer
calling archived verifier primitives from Phase 1/3/4 without changing closed
phase artifacts.

Ownership:

| Path | Purpose |
|---|---|
| `configs/` | Experiment settings, checkpoint identity and status |
| `data/raw/` and `data/candidates_v1/` | GSM8K source, checked candidate manifest, exclusions and verifier audit |
| `data/initials_v1/` on GPU; `outputs/phase5_remote_v100/` locally | Natural original-solver answers, verifier results and final hashes |
| `data/hints/v2/` | Strict feasibility audit; neutral/status ready, location/type infeasible |
| `data/splits/v1/` | Frozen balanced 240/80/160 source-disjoint split and hashes |
| `data/protocol/` | Frozen protocol lock, protected-opening lock, and completion receipt |
| `data/results_v1.json` | Compact post-run status outside the immutable pre-run registry |
| `outputs/phase5_gpu_vllm/review_v1/` | Every train/development prompt, output, parse and verification |
| `outputs/phase5_gpu_vllm/probe_v1/` | Frozen activations, fits, controls and layer selection |
| `outputs/phase5_gpu_vllm/protected_v1/` | The single protected review/probe opening |
| `scripts/` | Candidate, review, vLLM, activation, probe, control, lock and report CLIs |

Every generated artifact needs the source manifest hash, model checkpoint hash,
prompt/config hash, seed, software versions and parent artifact hashes. Adapters
are not standalone checkpoints; record their exact parent. Copies of code and
evidence should exist both locally and on the assigned GPU. GPU training is not
part of the initial diagnostic.

The initial collection followed local code and CPU validation, GPU upload,
generation, and immediate local evidence backup. Its as-run BF16 emulation on
the V100 is documented in `docs/GPU_HANDOFF.md`. Later reviews used pinned
vLLM 0.7.0 FP16 because the V100 is compute capability 7.0; probes used frozen
Transformers FP16 forward passes. All evidence was backed up locally before
reporting. `docs/FINAL_REPORT.md` owns the result and disposition.

The probe runtime is deliberately outside the checkpoint's generative path. It
is a read-only diagnostic harness over frozen activations, not part of the
standalone model and not an autonomous self-correction result. Any future
pipeline that uses probe predictions to invoke repair must identify that probe
as an external controller and evaluate detection, preservation, and verified
repair end to end on a new holdout.
