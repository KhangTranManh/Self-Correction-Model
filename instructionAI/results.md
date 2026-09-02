# Current verified result

This is the canonical record of the latest completed training and evaluation run.
The detailed P0 analysis is in [`outputs/p0_benchmark_report.md`](../outputs/p0_benchmark_report.md).
Older percentages in commit history and `note.txt` are historical pilot results.

## Run identity (2026-09-02)

| Item | Value |
|---|---|
| Base | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuned model | [`Kxck/Self_Correction_v1`](https://huggingface.co/Kxck/Self_Correction_v1) |
| Training | Unsloth QLoRA, rank 32, alpha 64, dropout 0 |
| Training data | 513 objectively verified correction rows |
| Loss masking | Failed attempt and verifier feedback are context; loss only on the verified correction |
| Sequence length | 1,536 |
| Batch | 2 per device × 8 gradient accumulation = 16 effective |
| Training duration | 3 epochs, 99 optimizer steps, 800.4 seconds |
| Final train loss | 0.450565 |
| Published format | Standalone merged BF16, four shards, 14.19 GiB |

The merged checkpoint does not require Unsloth, PEFT, or the external vLLM
BitsAndBytes plugin at serving time.

## Canonical P0 benchmark

`src/evaluate_p0_vllm.py` compares the untouched base and merged fine-tuned model
through the same vLLM API protocol. It covers:

- B1 — verifier-guided correction;
- B2 — false feedback / sycophancy;
- B3 — autonomous neutral review;
- B7 — correct-answer preservation; and
- B8 — cross-domain generalization to SVAMP and HumanEval.

The evaluator uses deterministic math-answer and executable-unit-test verification,
not an LLM judge. Generation used temperature 0.7, top-p 0.9, seed 314159,
1,024 initial tokens, and 1,536 review/correction tokens.

### Candidate pools and denominators

| Pool | Size |
|---|---:|
| Held-out GSM8K | 600, starting at test offset 200 |
| Held-out MBPP | 300, starting at test offset 200 |
| SVAMP OOD | 200 test examples |
| HumanEval OOD | all 164 examples |

B1 and B3 branch from 100 objectively wrong initial answers. B2, B3, and B7
branch from the same 100 objectively correct initial answers within each model.
The selected groups are outcome-conditioned and therefore not identical across
models; all reports retain per-domain counts and shared-problem diagnostics.

Selected in-domain mixes:

| Model | Correct group | Wrong group |
|---|---:|---:|
| Base | 90 math / 10 code | 36 math / 64 code |
| Fine-tuned | 90 math / 10 code | 24 math / 76 code |

### P0 headline results

| Benchmark | Base | Fine-tuned | Interpretation |
|---|---:|---:|---|
| B1 guided correction, objective | 15/100 (15%) | **56/100 (56%)** | strong tuned gain |
| B1 strict correction protocol | 0/100 (0%) | **56/100 (56%)** | tuned learned the protocol |
| B2 preservation after false feedback | **88/100 (88%)** | 78/100 (78%) | tuned worse |
| B2 false-flip rate | **12/100 (12%)** | 22/100 (22%) | tuned worse |
| B3 autonomous correction | 6/100 (6%) | 8/100 (8%) | weak/no clear gain |
| B7 neutral-review preservation | 74/100 (74%) | **86/100 (86%)** | tuned better overall |
| B8 conditional OOD correction | 26/76 (34.2%) | 44/100 (44%) | different groups; not directly comparable |

The B8 fixed pool produced only 76 initially wrong base answers, so the evaluator
correctly used 76 rather than inventing 24 additional failures. On the 31 selected
OOD problems that were initially wrong for both models, base corrected 12 and tuned
corrected 11. This provides no evidence of an OOD correction gain.

### Initial capability

| Dataset | Base | Fine-tuned | Difference |
|---|---:|---:|---:|
| GSM8K held-out | 441/600 (73.5%) | 514/600 (85.7%) | +12.2 pp |
| MBPP held-out | 12/300 (4.0%) | 20/300 (6.7%) | +2.7 pp |
| SVAMP OOD | 158/200 (79.0%) | 169/200 (84.5%) | +5.5 pp |
| HumanEval OOD, strict | 130/164 (79.3%) | 14/164 (8.5%) | -70.7 pp |

The HumanEval failure is partly an output-format transfer problem. The tuned model
often emits a correction report on a fresh code prompt instead of code-only output.
It used a correction heading on 144/164 HumanEval answers; the base used one on 0/164.
A lenient audit that extracts the final correction section raises tuned HumanEval to
77/164 (47.0%), still below the base's 130/164 (79.3%). Therefore formatting explains
part, but not all, of the regression.

### What P0 demonstrates

The fine-tuned model learned externally grounded, verifier-guided repair. Its largest
gain is B1 code correction: 41/76 versus 0/64 for base. It also follows the learned
correction protocol consistently.

The model did **not** learn reliable error discrimination:

- false flips increased from 12% to 22%;
- autonomous correction is only 8%;
- under neutral review, it claimed to find an error on all 100 wrong cases and all
  100 correct cases; and
- matched OOD correction did not improve.

The phrase-based “self-reported error” field must not be reported as genuine detection.
Claiming an error on every answer is evidence of an indiscriminate correction template.

### P0 decision

| Benchmark | Decision |
|---|---|
| B1 | Pass |
| B2 | Fail |
| B3 | Fail |
| B7 | Partial pass; math strong, code only 5/10 |
| B8 | Fail/mixed |

Overall, P0 does not support a claim that the checkpoint learned generalizable
autonomous self-correction. P1/P2 should be treated as optional diagnostics until a
new training mixture passes P0.

## Visible reasoning retention

The P0 JSONL files retain the exact prompt/problem, complete initial output, exact
feedback or neutral-review intervention, complete final output, extracted candidate,
and verifier result. `initial_visible_reasoning_output` and
`visible_reasoning_output` explicitly preserve all text returned by the model.

Neither model emitted `<thinking>...</thinking>` tags in this run, and vLLM did not
return a separate `reasoning_content` field. Therefore `visible_thinking` and
`separate_reasoning_content` are null. Hidden activations or unreturned private
chain-of-thought cannot be captured through the API and must never be implied to exist
in the saved data.

## Canonical artifacts

- `outputs/p0_benchmark_report.md`
- `outputs/p0_trained_100_summary.json`
- `outputs/p0_base_100_summary.json`
- `outputs/p0_trained_100_log.jsonl`
- `outputs/p0_base_100_log.jsonl`
- `outputs/p0_code_format_audit.json`
- `src/evaluate_p0_vllm.py`
- `outputs/AGI_future_redeploy_bundle_20260902.tar.gz`

The earlier 20-problem evaluation is retained as historical evidence in:

- `outputs/eval_merged_vllm_fixed_summary.json`
- `outputs/eval_merged_vllm_fixed_log.jsonl`
- `outputs/eval_base_vllm_summary.json`
- `outputs/eval_base_vllm_log.jsonl`

It reported 7/12 strict guided corrections for tuned and 0/12 for base. The P0 B1
result supersedes that small sample as the current guided-correction estimate.

## Serving result

The merged checkpoint was validated with vLLM 0.28 on an RTX 3090/3090 Ti class GPU
at a 4,096-token serving window. One active model uses roughly 19–22.5 GiB depending
on the vLLM/PyTorch build and cache state.

CUDA-runtime-only hosts without `/usr/local/cuda` or `nvcc` must disable the FlashInfer
sampler JIT path:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Kxck/Self_Correction_v1 \
  --served-model-name Kxck/Self_Correction_v1 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --dtype bfloat16
```

Use an API key if binding to a public interface. Rental IPs, ports, and credentials
are ephemeral and must never be written into project documentation.

## Required next training evidence

Before claiming success, add a balanced verified SFT mixture containing:

1. correct answer + false feedback → defend and preserve;
2. correct answer + neutral review → preserve without inventing an error;
3. wrong answer + neutral review → detect and repair without checker detail;
4. fresh code request → code-only output, without correction scaffolding; and
5. OOD-style code prompts with diverse signatures and docstrings.

Keep the successful verifier-guided examples, but mix them with solve-only and
no-change examples so the correction protocol is used only when appropriate. Retrain
and rerun the same P0 evaluator before proceeding to P1/P2 as confirmatory tests.
