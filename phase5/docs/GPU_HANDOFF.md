# Phase 5 GPU handoff

The CPU preparation is complete: 1,200 hash-ranked GSM8K train candidates,
1,200 fresh reference-verifier passes, and no overlap with recorded Phase 1-4
sources. The first GPU answer collection is complete: 815 natural answers,
575 correct and 240 wrong. Guided reviews and probes have not run.

The active GPU endpoint is a Linux Docker container with one Tesla V100-SXM2
32GB (32,768 MiB, compute capability 7.0), driver 580.82.07, Python 3.10.12,
and approximately 100 GB available under `/root`. Current
vLLM requires compute capability 7.5 or higher, so this pilot uses the
Transformers collector. The previous Windows RTX 3090 RDP handoff was not run.
Do not store remote credentials in the project or transfer package.

The local `outputs/phase5_transfer/phase5_v5_linux.zip` archive includes
Phase 5 code and selected CPU data, plus the Phase 1 prompt/verifier and
Phase 4 rollout library. It excludes model weights, old experimental evidence,
secrets and caches. On the GPU, extract under `/root/AGI_phase5` and run
`bash phase5/scripts/ops/bootstrap_linux.sh`. The script creates `.venv`
inside the project, installs pinned PyTorch 2.7.1 with CUDA 11.8 and inference dependencies,
then writes `phase5/runs/gpu_preflight.json`. It does not download the model
or start generation.

The initial-answer collection used full BF16 weights with PyTorch software
emulation on this V100. PyTorch's default `is_bf16_supported()` includes
emulation; the as-run collector picked BF16 despite the FP16 setup plan. The
setting is recorded in the final audit, so retain it as a documented pilot
condition. The local collector is corrected to select native BF16 only on
compute capability 8.0 or newer for future runs. The as-run source remains in
`outputs/phase5_transfer/phase5_v5_linux.zip` with SHA256
`462ea16e5d10e38e6633dd908fb3dad840e5eca605389a12c282adcaecd18d3f`
for its collector file.

The original checkpoint was `Kxck/Self_Correction_v1`. The run generated one
natural answer per candidate with the frozen Phase 1 math prompt and no hints,
using batch size one and up to 768 new tokens. The append-only audit records
raw outputs, seeds, model revision, precision, versions, manifest hash and
verifier outcomes. It stopped at the first ordered prefix containing at least
240 correct and 240 wrong answers. The final audit hash is
`7d7d9b56ccfbe2bd730fe52507871f01317dc106cd2938a89557b07777478357`;
the derived 815-row rollout hash is
`04263288354cbfacd7f018a65e7b57394f4a21da234719b29b65cf8cccecb024`.
Both files and the summary are backed up under `outputs/phase5_remote_v100/`.
Next freeze source-disjoint splits before hinted review or probe activation.

The reference-step locator is a number-match heuristic, not verified ground
truth for a model error. Location/type hints require independent accuracy
checks before use.
