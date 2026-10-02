# Serving

OpenAI-compatible vLLM serving for the project's checkpoints, plus a client
that runs the answer strategies tested in Phases 7–10.

## Models

All checkpoints descend from **Qwen/Qwen2.5-7B-Instruct**. Every adapter is
LoRA rank 16 (alpha 32) on all attention and MLP projections.

| Served name | What it is | Base | Status and evidence |
|---|---|---|---|
| `original-solver` | `Kxck/Self_Correction_v1` (rev `6437f94…`), Phase 1 SFT for verified critique and correction | Qwen2.5-7B-Instruct | Canonical solver. Phases 5–10 all use its first answers. |
| `warmstart-v2` | Phase 4 exploration warm-start LoRA | `original-solver` | Archived pilot. Experiments used it **merged** into FP16; served here as an unmerged LoRA, so outputs can differ slightly. |
| `warmstart-v2-merged` | V2 merged into the original solver (FP16) | — | Base for V3; built by `phase7/scripts/materialize_v2_local.py`. |
| `correction-sft-v3` | Phase 4 correction SFT LoRA | `warmstart-v2-merged` | Archived pilot; best blind re-solver in Phase 8, but over-revises when shown an answer. |
| `phase10-judge` | **Phase 10 judge LoRA** — trained to choose between two conflicting solutions and explain the error | `original-solver` | Trained in Phase 10 on the model's own verifier-correct judgments. **Not promoted:** on the holdout it picked the right solution at chance (50.7% vs 52.1% untrained) and trailed voting. Served for inspection only. |

Phase 3 routers (Decision-Only V1 and DPO variants) are served by
`phase3/scripts/serving/serve_phase3.sh`.

## What the methods mean (from the experiments)

| Method | How | Evidence |
|---|---|---|
| `single` | One answer | Phase 9 protected: 75.0% |
| `vote` | Plurality over five independent attempts; later attempts never see earlier ones | Phase 9: 83.25–85.5%, +8 to +10.5 points (Holm p < 0.001) |
| `self_check` | Keep if two independent attempts agree; otherwise a judge compares both | No gain: Phase 9 untrained judge 72.25–74.0%; Phase 10 trained `phase10-judge` 73.0% (same as untrained). |

Never show the model an earlier answer and ask it to re-solve: Phases 7–8
showed visible answers cut repair from ~45% to ~15% and break ~25% of
correct answers.

## Run

On a GPU host with the project's `.venv-vllm` (vLLM 0.7.0), the adapters under
`outputs/`, and — for `v3` — `models/phase7_v2_merged_fp16`:

```bash
bash serving/serve.sh original      # original-solver, warmstart-v2, phase10-judge
bash serving/serve.sh v3            # warmstart-v2-merged, correction-sft-v3
```

One 7B profile uses about 28 GB, so run one profile at a time on a 32 GB GPU.
The server binds to `127.0.0.1:8000`; set `SERVE_API_KEY` before binding to
any other interface.

```bash
python serving/client.py --method vote --question "..."
python serving/client.py --method self_check --judge-model phase10-judge --question "..."
python serving/client.py --method single --model correction-sft-v3 --question "..."
```

The client prints every attempt, whether the attempts agreed, and the final
solution. It does not verify answers — without a reference answer, `vote` is
the best-supported choice.
