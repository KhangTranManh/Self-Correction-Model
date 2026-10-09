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
| `phase13-verdict-judge` | **Phase 13 constrained-verdict DPO judge LoRA** — ends with "Verdict: Solution A/B" | `original-solver` | On GSM8K test: invented answers 2.1%, coverage 63%, consistent accuracy 74.0%. Use the constrained prompt from `phase13/scripts/constrained.py`. Not a replacement for vote@5. |
| `phase12-dpo-judge` | **Phase 12 order-swapped DPO judge LoRA** — highest consistent accuracy (79.3% on GSM8K test) | `original-solver` | On SVAMP its consistent both-orders verdicts were right 74.9% (untrained 68.7%); its self-check was non-inferior to compute-matched vote@3. Not a replacement for vote@5. Note: `serving/client.py` `self_check` judges in one order; the both-orders rule is in `phase12/scripts/analyze.py`. |
| `phase11-dpo-judge` | **Phase 11 DPO judge LoRA** — prefers correct over incorrect judgments of the same pair | `original-solver` | **Not promoted:** 67% validation preference, but on fresh problems it picks the right solution 49.4% (vs 46.6% untrained, n.s.) and favors the second solution shown. Served for inspection only. |

Phase 3 routers (Decision-Only V1 and DPO variants) are served by
`phase3/scripts/serving/serve_phase3.sh`.

## What the methods mean (from the experiments)

| Method | How | Evidence |
|---|---|---|
| `single` | One answer | Phase 9 protected: 75.0% |
| `vote` | Plurality over five independent attempts; later attempts never see earlier ones | Phase 9: 83.25–85.5%, +8 to +10.5 points (Holm p < 0.001); Phase 10: +9.0; Phase 11: +12.8; Phase 12: +8.0; Phase 13: +11.4 — still the most accurate method |
| `self_check` | Keep if two independent attempts agree; otherwise a judge compares both | No gain: Phase 9 untrained judge 72.25–74.0%; Phase 10 `phase10-judge` 73.0% (same as untrained); Phase 11 `phase11-dpo-judge` 72.2% vs 77.4% for agreement-gated voting. |

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
