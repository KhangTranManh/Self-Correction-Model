# Phase 16 P16-E1 — Colab T4 A0 engineering smoke

Frozen 2026-11-03 before any Phase 16 model generation. This is a nine-task
engineering check of model load, code extraction, sandbox execution, test
transcripts and per-item audit. It is **not** the protocol's feasibility
screen or 40-episode pilot. All nine original synthetic tasks are permanently
excluded from future teaching, probe, control and confirmation pools. No
protected split exists in P16-E1.

## Frozen inputs

- Task pack: `../data/engineering_smoke_v1/tasks.json`, SHA-256
  `cabac7e679b2de020e9de2cefa9c8d01d0f4b3c4f6f364600508aaafb7154cde`.
  Nine distinct engineering tasks: three count, three sum and three select.
  The pack contains each exact prompt, entry point, ten locked test inputs,
  oracle values and two upper-boundary pairs. Only task prompts are sent to
  the model; test inputs and oracle values remain in the verifier.
- Task builder: `../scripts/build_engineering_smoke.py`, SHA-256
  `7cce91a6f4c90a4d32616f578f66a41e75bb24c91fd35a2d645206394eaed52a`.
- Scorer: `../scripts/score_boundary.py`, SHA-256
  `4cabc19a099497ca5b9d5f388fb3d8f7be6102c71cbd82aaa6f9f5e86cf52b21`.
- Isolated verifier: `../scripts/verify_code.py`, SHA-256
  `bd897486d4a331b114820dbd4a9b655da8eed3eac3383beb1b994c6496512ef7`.
- Sandbox preflight: `../scripts/check_colab_sandbox.py`, SHA-256
  `1839a7c9aa2a6866ebd0ed8d5090934a400d1f465c84c27be081d2ea19108d18`.
- Trusted fixture helper: `../scripts/verify_runner_smoke.py`, SHA-256
  `279fd5cfd614f8efb766451a832880b6fc28e5358fb42a03473af234a32da03e`.
- GPU run script: `../scripts/colab_a0_engineering.py`, SHA-256
  `e268a6a58869750b72949e5e7a0d086064f0cf47f60538dac5bf351fc8cfb35c`.

## Execution

Use one dedicated Colab T4 session. Install `bubblewrap` in that disposable
runtime. Check that a sandboxed process has UID 65534, cannot see `/content`
or `/root`, and cannot connect to the network. Verify all nine trusted correct
and exclusive-upper fixture programs inside the sandbox before loading the
model. If any preflight fails, stop before model generation.

Load public `Qwen/Qwen2.5-7B-Instruct` in NF4 double quantization with FP16
compute, using `transformers==4.46.3`, `accelerate==1.0.1` and
`bitsandbytes==0.50.2`. Generate A0 only with a fixed system prompt, greedy
decoding, 1,024 input-token cap, 512 new-token cap, and a recorded per-task
seed. Each generated completion is checked inside a networkless bubblewrap
process with restricted filesystem, UID, CPU, address space, file size, open
files and process count. The scorer reads trusted oracle outputs only in the
parent. The run has a one-hour wall and writes append-only metadata, stage and
completion rows to `/content/phase16/gpu_smoke_v1/audit.jsonl`; it refuses to
overwrite any existing audit. Download the audit and summary before stopping
the Colab session.

During the run, poll the append-only audit and report completed tasks and
labels without treating a partial run as a final score. Afterward verify all
input hashes, nine unique completion IDs, every scorer label, and the final
audit hash locally. A failure or early stop is an infrastructure outcome.

## Interpretation

Report observed labels by task and whether the end-to-end path worked. Do not
estimate the population prevalence of C-upper-inclusive from these nine
tasks, compare interventions, or use them to tune a scientific claim. The
subsequent feasibility screen still requires a separately frozen source pool
with 40 natural teaching failures and disjoint probe/control capacity.

## Result

The T4 run completed all nine tasks. Stages were
`sandbox_and_oracle_ready`, `packages_ready`, `model_loading`, `model_ready`
and `complete`. Every generated solution passed all ten unit tests and was
labelled `correct`: 3/3 count, 3/3 sum, 3/3 select. No
`upper_boundary_signature`, syntax error, runtime error or timeout occurred.
The downloaded audit contains nine unique completion IDs; its SHA-256 is
`11b5f13417fe5af11b353512f1df227c1ff30d7d725417a2c32979c29433e420`
and matches the summary. A separate saved-output rerun inside the isolated
verifier reported zero disagreements (`../results/p16_e1_rescore.json`).
The T4 session `phase16-e1` was stopped after artifact recovery.

The frozen plan mentioned a local rescore. Executing arbitrary model code on
the local Windows host would not have the Colab sandbox, so the saved-output
re-execution ran in the Colab sandbox instead. Local checks verified hashes,
IDs and label totals. This execution-location deviation is recorded; the run
remains engineering-only and cannot support an efficacy claim.

These nine easy engineering tasks do not establish that the error type is
absent in a broader population. They show that the pipeline works and that
this pack itself supplies no natural teaching failures. The protocol's
feasibility screen requires a new, diverse, separately frozen source pool.
