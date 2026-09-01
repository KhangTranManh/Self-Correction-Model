# Current verified result

This file is the canonical record of the latest completed training/evaluation run.
Older percentages in commit history or `note.txt` describe earlier pilot adapters and
must not be substituted for this run.

## Run identity (2026-09-02)

| Item | Value |
|---|---|
| Base | `Qwen/Qwen2.5-7B-Instruct` |
| Training method | Unsloth QLoRA; rank 32, alpha 64, dropout 0 |
| Training data | 513 objectively verified correction rows |
| Loss masking | Failed attempt + verifier feedback are context; loss only on the final verified correction |
| Sequence length | 1536 |
| Batch | 2 per device × 8 gradient accumulation = 16 effective |
| Training | 3 epochs, 99 optimizer steps, 800.4 seconds |
| Final train loss | 0.450565 |
| Adapter | `outputs/phase1_lora` |
| Published model | [`Kxck/Self_Correction_v1`](https://huggingface.co/Kxck/Self_Correction_v1) |
| Published format | Standalone merged BF16, four shards, 14.19 GiB |

The published model was created by applying the trained LoRA to the original BF16
Qwen base and saving merged safetensors. It does not require Unsloth, PEFT, or the
external vLLM BitsAndBytes plugin at serving time.

## Held-out evaluation

`src/evaluate_self_correction_vllm.py` evaluated 10 GSM8K and 10 MBPP problems starting
at offset 150. The same 20 prompts and seed were used for the merged trained model and
the untouched Qwen base. Each initial answer was checked objectively. Only an initially
wrong answer received a second turn containing the real verifier error under role
`tool`; the second answer was then checked by the same objective verifier.

Headline metric:

```text
self_correction_rate = objectively corrected second attempts / initially wrong attempts
```

### Canonical results

| Model | Initial correct | Initially wrong | Corrected | Rate | Missing required format |
|---|---:|---:|---:|---:|---:|
| Untouched Qwen | 8/20 | 12 | 0/12 | 0.0% | 12/12 |
| **Self_Correction_v1** | **8/20** | **12** | **7/12** | **58.3%** | **0/12** |

Trained-model breakdown:

| Domain | Initial correct | Initially wrong | Corrected | Rate |
|---|---:|---:|---:|---:|
| Math | 6/10 | 4 | 3/4 | 75.0% |
| Code | 2/10 | 8 | 4/8 | 50.0% |

Untouched Qwen did often produce another answer, but it never followed the trained
`### Phát hiện lỗi` / `### Nguyên nhân` / `### Sửa lại` protocol. A separate lenient
diagnostic that ignored the protocol still found only 2/12 objectively repaired base
answers (16.7%). That lenient number is diagnostic, not the canonical protocol metric.

The two models had the same count of initially wrong answers but not necessarily the
same wrong problem IDs, so this small comparison is not a perfectly paired estimate.
Keep the raw counts and expand the held-out set before making a publication-level claim.

## Interpretation

The model **did learn a useful correction behavior**: after an objective checker marks
an answer wrong and supplies a concrete error, it follows the learned critique format
and repairs the answer more often than the untouched base. The correct statement is:

> Self_Correction_v1 learned externally grounded, one-round error repair, but that
> behavior is not yet reliable.

Do not overstate this as autonomous introspection. The run did not test whether the
model can notice its own error without feedback. Five of twelve wrong attempts remained
wrong after feedback, initial accuracy stayed 8/20, and the held-out sample is only 20
problems. The project target of at least 70% self-correction is not yet met.

## Evaluation bug that produced a false 0%

The first vLLM evaluation incorrectly reported 0/12 because `_CORRECTION_RE` contained
a mojibake version of `Sửa lại`. All 12 trained-model corrections actually emitted the
valid Unicode heading, but the evaluator classified every one as `format_incomplete`.

The fix in `evaluate_self_correction_vllm.py`:

- represents the Vietnamese heading with Unicode escapes;
- accepts `Sửa lại`, `Corrected answer`, or `Correction`;
- accepts corrected content on the same line as the heading;
- still requires a completed correction section before objective verification.

The run was regenerated after the fix. The canonical local artifacts are:

- `outputs/eval_merged_vllm_fixed_summary.json`
- `outputs/eval_merged_vllm_fixed_log.jsonl`
- `outputs/eval_base_vllm_summary.json`
- `outputs/eval_base_vllm_log.jsonl`

Any older summary showing `self_correction_rate: 0.0` together with
`format_incomplete: 12` for `Self_Correction_v1` is the invalid pre-fix run.

## Serving result

The merged checkpoint was validated with vLLM 0.28 on one RTX 3090 24 GB at a
4096-token serving window. Observed VRAM use was approximately 22.5/24.6 GiB.
CUDA-runtime-only containers need the native sampler switch because FlashInfer otherwise
tries to JIT-compile with an unavailable `nvcc`:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Kxck/Self_Correction_v1 \
  --served-model-name Kxck/Self_Correction_v1 \
  --host 0.0.0.0 --port 8999 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --dtype bfloat16 --api-key "$VLLM_API_KEY"
```

Do not rely on a recorded rental-server IP or port: rented containers and exposed-port
mappings are ephemeral. Verify the current platform mapping and `/v1/models` health
response each session.

## Required next evidence

1. Expand the held-out comparison to at least 100–200 problems and report raw counts.
2. Add a false-feedback/sycophancy test: tell the model a correct answer is wrong and
   measure whether it improperly changes it.
3. Test autonomous detection separately, without a verifier declaring `SAI`.
4. Test two or three repair rounds and measure repeated-error rate.
5. Inspect the five remaining failures and add verified examples matching those error
   types rather than merely duplicating already-solved patterns.
