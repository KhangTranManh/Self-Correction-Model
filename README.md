# Phase 1 — Verified Self-Correction

Research pipeline teaching a 7B LLM the foundational skill of **recognizing when
it is wrong and correcting itself**, on domains where correctness can be checked
programmatically (math, code) — before any domain-specific knowledge training.
Right/wrong labels come from an **objective verifier** (sympy / real unit-test
execution), never from an LLM judging another LLM's output.

This is explicitly a *behavior* pipeline, not a *knowledge* pipeline: the goal
is the critique → correct loop itself, not skill at GSM8K/MBPP. See
[`note.txt`](note.txt) for the full research log (risks considered, dead ends,
decisions and why).

---

## Status (2026-09-02)

| | |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct, QLoRA (4-bit) via Unsloth |
| Training data | 513 verified correction rows; failed attempts and verifier feedback are prompt context, and loss is computed only on the verified correction |
| Training run | 3 epochs, 99 optimizer steps, rank 32 / alpha 64 LoRA, final train loss 0.4506 |
| Published model | Merged BF16 checkpoint: [`Kxck/Self_Correction_v1`](https://huggingface.co/Kxck/Self_Correction_v1) (14.19 GiB, four shards, directly loadable by vLLM) |
| Canonical P0 pool | 600 held-out GSM8K + 300 held-out MBPP; OOD: 200 SVAMP + all 164 HumanEval |

The canonical evaluation is now the 100-case P0 suite in
[`outputs/p0_benchmark_report.md`](outputs/p0_benchmark_report.md). It compares the
untouched base and fine-tuned model on verifier-guided correction, false feedback,
autonomous review, correct-answer preservation, and cross-domain generalization.

| P0 metric | Base | Self_Correction_v1 |
|---|---:|---:|
| B1 guided correction | 15/100 (15%) | **56/100 (56%)** |
| B2 false-flip rate | **12/100 (12%)** | 22/100 (22%) |
| B3 autonomous correction | 6/100 (6%) | 8/100 (8%) |
| B7 correct preservation | 74/100 (74%) | **86/100 (86%)** |
| B8 OOD correction | 26/76 (34.2%) | 44/100 (44%) |

The model learned a strong **externally guided correction** behavior, especially for
code, but P0 does not support reliable autonomous error discrimination. False flips
increased, autonomous repair remained weak, and HumanEval strict initial accuracy fell
from 79.3% to 8.5%. A lenient correction-section extraction raises tuned HumanEval to
47.0%, so output-format over-transfer explains part—but not all—of that regression.

The earlier 20-problem run (7/12 strict guided corrections for tuned, 0/12 for base)
is retained as historical evidence, but P0 supersedes it as the current estimate. See
[`instructionAI/results.md`](instructionAI/results.md) for the canonical interpretation.

---

## Key Findings

1. **Role-labeling the reflect turn matters, independent of model capability.**
   Per ["The Self-Correction Illusion" (arXiv 2606.05976)](2606.05976v1.pdf), LLMs
   fail to self-correct errors sitting in their own reasoning not from a
   reasoning deficit but from a lack of *addressability* — no learned habit of
   treating a claim inside their own output as a nameable, rejectable object.
   Re-presenting the identical error under an external chat-template role
   (`user`/`tool`/`memory`) instead of leaving it in `<thought>` measurably lifts
   correction rates, with **no retraining required** to see an initial effect.
   Earlier prompt-only and 135-row pilot runs established `tool` as the correct
   role because `verifier_detail` is genuinely checker output. In the current
   513-row run, the trained model followed the correction protocol for all 12
   wrong attempts and repaired 7; the untouched base followed it for none and
   repaired only 2 under a lenient non-protocol check. See
   [Status](#status-2026-09-02) for the canonical comparison.

2. **Revealing the correct answer in the error message is not sufficient for
   self-correction, and doesn't explain the math/code gap.** The original
   hypothesis was that math's verifier message (which may state the correct
   final answer) would make correction trivial. It did not: one of four wrong
   math attempts still failed after feedback, and four of eight code attempts
   failed despite precise tracebacks. The current evidence supports
   addressability—a concrete, locatable error—as useful, not sufficient.

3. **Two real measurement bugs were found and fixed.** In an early run, ~1/3 of "successful"
   self-corrections were generation cut off mid-`<thinking>`
   (never reaching `### Sửa lại`), with the verifier accidentally matching a
   stray number/code fragment from the unfinished text. Fixed by (a) doubling
   the reflect-turn token budget and (b) treating incomplete generations as
   failures instead of falling back to the raw text. In the current vLLM run,
   a mojibake-corrupted regex caused the opposite error: all 12 valid Unicode
   `### Sửa lại` headings were rejected, producing a false 0%. The extractor
   now uses Unicode escapes, accepts same-line content, and the run was repeated.

4. **The larger P0 benchmark separates correction sensitivity from error
   discrimination.** The tuned model improves guided correction from 15% to 56%, but
   false flips worsen from 12% to 22% and autonomous correction remains only 8%.
   It also claims an error under neutral review on every correct and wrong case. This
   is evidence of a learned correction trigger/template, not reliable introspection.

Full detail, including risks considered and rejected approaches, in
[`note.txt`](note.txt).

---

## Project Structure

```
configs/phase1.yaml       hyperparams + paths (no secrets)
data/problems/            source math/code problems
data/processed/           generated data (gitignored)
outputs/                  trained LoRA adapter (gitignored)
src/config.py             loads .env + yaml into one Config object
src/data/                 Problem schema + dataset loaders
src/verifier/             objective verifiers (math_verifier, code_verifier)
src/deepseek_client.py    calls DeepSeek to generate critique + correction
src/model_loading.py      shared Unsloth model/tokenizer loading (all 3 GPU scripts)
src/generate_attempts.py  small model solves problems -> attempts.jsonl (real attempts, not synthetic)
src/build_dataset.py      verify -> DeepSeek critique -> re-verify -> phase1_sft.jsonl
src/prepare_training_dataset.py  masks/compacts context -> phase1_sft_train.jsonl
src/train_sft.py          QLoRA SFT (Unsloth + trl)
src/evaluate_self_correction.py  held-out eval: does the model actually self-correct?
src/evaluate_self_correction_vllm.py  same objective eval through a vLLM HTTP server
src/evaluate_p0_vllm.py  100-case B1/B2/B3/B7 plus SVAMP/HumanEval B8 benchmark
src/export_merged_for_vllm.py  merge adapter into standalone BF16 weights + publish
src/push_to_hub.py        push a saved adapter directory to HF Hub manually
```

## Setup

1. Create `.env` (never commit this file):

   ```
   DEEPSEEK_MODEL=...
   DEEPSEEK_API_KEY=sk-...
   DEEPSEEK_BASE_URL=https://api.deepseek.com   # optional, has a default
   HF_TOKEN=hf_...                              # optional, only for pushing adapters
   ```

2. Install:

   ```
   pip install unsloth
   pip install -r requirements.txt
   ```

   **Hardware requirement:** GPU compute capability ≥ 7.5 (T4, L40S, A100, 4090,
   H100, ...). Check with `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
   V100/Volta (CC 7.0) is **not supported** by current Unsloth/Axolotl releases
   — see [Hardware notes](#hardware-notes).

3. Set `small_model.name_or_path` in `configs/phase1.yaml` to the model you're using.

## Pipeline (fixed order — later steps read earlier steps' output)

| Step | Command | Needs GPU? |
|---|---|---|
| 1 | `python -m src.prepare_public_datasets` | No |
| 2 | `python -m src.generate_attempts` | Yes |
| 3 | `python -m src.build_dataset` | No (needs correction API) |
| 4 | `python -m src.prepare_training_dataset` | No |
| 5 | `python -m src.train_sft` | Yes |
| 6 | `python -m src.evaluate_self_correction --adapter <repo_or_path>` | Yes |
| 7 (served model) | `python -m src.evaluate_self_correction_vllm --model Kxck/Self_Correction_v1` | vLLM server needs GPU |
| 8 (P0 benchmark) | `python -m src.evaluate_p0_vllm --model Kxck/Self_Correction_v1` | vLLM server needs GPU |

Steps 6–8 produce the metrics that matter — training loss alone does not tell you
whether the model learned to self-correct (see [Key Findings](#key-findings)).

**Before running full-scale on rented GPU hours:** run steps 1–3 on the small
sample problem set first, check the stats printed at the end of step 3
(`total / already_correct / kept / discarded`), and only then scale to the
full problem set.

### Publish and serve the current model

`export_merged_for_vllm.py` reads `HF_TOKEN` from the environment, merges the
adapter into the original BF16 base, writes a standalone checkpoint, and uploads it:

```bash
python -m src.export_merged_for_vllm \
  --adapter-dir outputs/phase1_lora \
  --output-dir outputs/Self_Correction_v1_merged \
  --repo-id Kxck/Self_Correction_v1
```

Serve the published model with native vLLM (no Unsloth/BitsAndBytes loader needed):

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Kxck/Self_Correction_v1 \
  --served-model-name Kxck/Self_Correction_v1 \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --dtype bfloat16
```

The sampler environment variable is required on CUDA-runtime-only containers that do
not include `nvcc`; otherwise FlashInfer tries to JIT-compile during warm-up and vLLM
exits. The verified RTX 3090 deployment used approximately 22.5/24.6 GiB VRAM.
Use `--host 0.0.0.0 --api-key "$VLLM_API_KEY"` only when a remote client must reach
the endpoint; keep the unauthenticated benchmark endpoint on localhost.

With the server available at `http://127.0.0.1:8000/v1`, run the canonical P0 suite:

```bash
python -m src.evaluate_p0_vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Kxck/Self_Correction_v1 \
  --log-file outputs/p0_trained_100_log.jsonl \
  --summary-file outputs/p0_trained_100_summary.json
```

Serve the untouched base separately and repeat with model id
`Qwen/Qwen2.5-7B-Instruct`. One 24 GB GPU cannot safely host both BF16 models at once.

## Hardware Notes

- Code uses Unsloth's `FastLanguageModel` directly — requires GPU CC ≥ 7.5
  (Turing/Ampere/Ada/Hopper). Verified in practice: current Unsloth/Axolotl
  releases require torch ≥ 2.11/2.3, and torch has dropped CC 7.0 (Volta/V100)
  kernel support since ~2.3 — `torch.cuda.is_available()` still returns `True`
  on V100 but compute silently errors or produces wrong results.
- This project has run across several GPU rental platforms (root-SSH VPS,
  Kaggle T4, JupyterHub L40S, Google Colab T4) — see `instructionAI/environment.md`
  for platform-specific connection notes.

## Sandbox / Safety

`CodeVerifier` executes model-generated code in a subprocess with a timeout and
memory limit — adequate for self-generated/trusted problem sources (current
scope). If extended to untrusted problem sources (e.g. crawled from the
internet), this needs proper containerization (docker/firejail) first —
subprocess isolation alone is not sufficient for adversarial code.

## Known Issues / Limitations

See [`note.txt`](note.txt) section 3 for the full list. Headline items:

- Training set is 513 correction-only samples versus the planned 800–1500 balanced
  examples. The missing no-change, false-feedback-defense, autonomous-review, and
  ordinary code-only examples are now measured failure modes, not hypothetical gaps.
- Math verifier's answer extraction can mis-parse stray formatting characters
  (e.g. a stray backtick) leaked from the model's own markdown-style output.
- P0 now tests sycophancy and autonomous review. The tuned false-flip rate is 22%,
  autonomous correction is 8%, and neutral-review error claims are indiscriminate.
- P0 does not include multi-round correction, feedback-specificity ablations, role
  ablations, prompt robustness, or calibration. These remain P1/P2 work and should be
  diagnostic until a retrained model passes P0.
- HumanEval shows correction-format over-transfer: strict accuracy is 8.5% and a
  lenient correction-section audit reaches 47.0%, still below the 79.3% base result.

## References

- Chen, K-Y., Su, F-Y., Chiang, J-H. (2026). *The Self-Correction Illusion:
  LLMs Correct Others but Not Themselves.* [arXiv:2606.05976](2606.05976v1.pdf)
