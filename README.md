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

## Status (2026-07-21)

| | |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct, QLoRA (4-bit) via Unsloth |
| Problem set | 1000 GSM8K + 624 MBPP = 1624 (was 300) |
| Training data | **631 samples** (was 135) |
| Adapter | `Kxck/AGI_v2`; `Kxck/AGI_v1` kept as the control |
| Held-out eval | **600 GSM8K + 150 MBPP** (offset 150, disjoint from training) |

**The headline metric is a product of two factors, and only one of them is
failing:**

```
self-correction rate  =  format completion  ×  fix rate given format
      37.8%           =        58.2%        ×          64.8%
```

| | Value | vs. target |
|---|---|---|
| Self-correction rate (headline) | **37.8%** (94/249) | target ≥70% — far off |
| ├ produced the required format | 58.2% | **this is where the entire gap is** |
| └ fixed correctly once it did | **64.8%** | close to target |

Reporting only the headline hides which factor is broken. See
[Key Finding 4](#key-findings).

**Two interventions were tested and neither moved the second factor:**

| Change | Format completion | Fix rate given format |
|---|---|---|
| 135 → 631 training samples (4.7×) | no change | no change (p = 0.856 overall) |
| reflect role `tool` → `memory` (matched) | 58.2% → 66.4% (p = 0.072) | 64.8% → 67.6% (**p = 0.62**) |
| train/eval framing mismatch | 58.2% → 21.1% (**p = 2e-16**) | 64.8% → 69.6% (**p = 0.60**) |

Historical numbers, kept for context — note the sample sizes:

| Run | Math | Code | Overall |
|---|---|---|---|
| Baseline (train + eval, role `user`) | 0.0% | 28.6% | 22.9% |
| Old adapter + role `tool` at eval only | 33.3% | 28.6% | 29.7% |
| Retrained role `tool`, 135 samples | 22.2% (2/9) | 46.4% (13/28) | 40.5% (15/37) |
| **631 samples, properly sized eval** | **31.7%** (33/104) | **42.1%** (61/145) | **37.8%** (94/249) |

40.5% → 37.8% is **not** a regression: Fisher exact p = 0.856. The 40.5% rested on
37 wrong cases and was never a firm baseline. Every historical *math* figure rested
on 7–9 cases — see [Key Finding 5](#key-findings).

Success criteria from the project plan: self-critique accuracy ≥75–80% (**never
measured** — that is the *identification* metric, not the correction metric),
self-correction ≥70%, no repeated errors after 2–3 rounds (**never tested** — eval
runs one round). Full reasoning in [`note.txt`](note.txt).

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
   Switching the reflect turn's role from `user` to `tool` (since
   `verifier_detail` is genuinely a tool/checker output) took math
   self-correction from 0% → 33.3% as a prompt-only probe on the old adapter,
   and after adopting `tool` as the pipeline default and retraining, the
   overall rate (matched train/eval role) reached 40.5%, nearly double the
   original baseline — see [Status](#status-2026-07-21) for the full table.

   **Superseded in part by Findings 4 and 5.** The framing effect is real and
   reproduced under a properly sized eval, but it does *not* work through improved
   diagnosis as this finding originally assumed — it works by making the model
   finish writing. And the specific math figures quoted here (0% → 33.3%) rest on
   7–9 cases and do not support a claim about math.

2. **Revealing the correct answer in the error message is not sufficient for
   self-correction, and doesn't explain the math/code gap.** The original
   hypothesis was that math's verifier message (which states the correct
   final answer) would inflate self-correction rates artificially. Held-out
   data showed the opposite: math self-correction was 0% even when the exact
   correct answer was handed to the model — it could not locate *which step*
   in its own derivation was wrong. Code's verifier message never reveals the
   answer but pinpoints an exact line/function via traceback, and self-corrects
   far more often. Conclusion: addressability (a concrete, locatable claim) —
   not answer-leakage — is what drives self-correction success.

3. **A real measurement bug was found and fixed**: ~1/3 of "successful"
   self-corrections in an early run were generation cut off mid-`<thinking>`
   (never reaching `### Sửa lại`), with the verifier accidentally matching a
   stray number/code fragment from the unfinished text. Fixed by (a) doubling
   the reflect-turn token budget and (b) treating incomplete generations as
   failures instead of falling back to the raw text.

4. **The bottleneck is format adherence, not self-correction skill.** A 2×2
   experiment (adapter trained on `tool` or `memory` framing × evaluated under
   either) decomposed the metric into its two factors:

   | Cell | Format completion | Fix rate given format | Headline |
   |---|---|---|---|
   | A: tool / tool | 58.2% | **64.8%** | 37.8% |
   | B: tool / memory | 63.6% | **64.3%** | 40.9% |
   | C: memory / memory | 66.4% | **67.6%** | 44.8% |
   | D: memory / tool | 21.1% | **69.6%** | 14.7% |

   The middle column barely moves across two adapters, two framings, matched and
   mismatched — and it did not move when training data grew 4.7×. Cell D is the
   sharpest evidence: it has the *worst* headline (14.7%) and the *best* fix rate
   (69.6%). A model whose framing it was never trained on still diagnoses just as
   well; it simply rarely produces output at all.

   **The self-correction skill sits at ~65–70% and has not responded to any
   intervention tried.** The plan's target is ≥70%. That target is essentially
   already met on the skill itself — nobody had measured it separately.

   This also reinterprets Finding 1: the `memory` framing does help, but **not by
   the mechanism the paper describes**. The paper attributes the lift to improved
   addressability, i.e. better diagnosis. Here diagnosis does not change
   (p = 0.62); what changes is whether the model finishes writing.

5. **Every historical math figure was measured on 7–9 cases.** The same adapter on
   the same problems measured **15.8% math at n=19 and 31.7% at n=104**. Because the
   base model already solves ~84% of GSM8K correctly, the denominator
   (`initial_wrong`) is tiny unless the problem count is large. Rule of thumb: **600
   math problems** are needed for a denominator near 100. The finding that "role
   `tool` lifted math from 0% to 33.3%" — which motivated changing the whole
   pipeline — was 0/7 versus 3/9.

   Relatedly, the metric has a **selection effect**: the denominator is the model's
   own failures, so a better model is left with a smaller, harder denominator. On
   the exact 30 math problems used for the 40.5% measurement, the newer adapter gets
   **all 30 right on the first try**. Always report `initial_correct` alongside.

6. **Balancing the problem set does not balance the training set.** The problem
   set was deliberately kept near 61/38 math/code to avoid skewing the model.
   Measured against the verifier on 906 fresh attempts, the result is the
   opposite of what that implies:

   | Domain | Problems | Solved correctly first try | Needs correction |
   |---|---|---|---|
   | Math (GSM8K) | 432 | **364 (84.3%)** | 68 |
   | Code (MBPP) | 474 | 35 (7.4%) | **439** |

   Training data is built only from *failures*, so it comes out ~87% code. What
   determines the mix is not how many problems you supply but **where the model
   fails** — Qwen2.5-7B finds GSM8K easy and MBPP hard. Adding math problems is
   nearly useless here: ~84% would be discarded as already-correct. Getting more
   math signal requires harder problems (the MATH competition set), not more of
   them — which needs a new loader and a verifier that handles symbolic answers,
   not just numbers. This also reframes finding 2: the 22.2% math vs 46.4% code
   gap was measured on an adapter trained from data that was *already* code-heavy.

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
src/prompts.py            SINGLE source for every shared prompt template
src/eval_common.py        held-out loader + log format shared by both eval scripts
src/run_lock.py           PID lock: refuses a second concurrent instance
src/deepseek_client.py    calls the API model to generate critique + correction
src/model_loading.py      shared Unsloth model/tokenizer loading
src/prepare_public_datasets.py  GSM8K/MBPP -> data/problems/*.jsonl (split-aware)
src/generate_attempts.py  small model solves problems -> attempts.jsonl (vLLM, batched)
src/build_dataset.py      verify -> API critique -> re-verify -> phase1_sft.jsonl (threaded)
src/train_sft.py          QLoRA SFT (Unsloth + trl)
src/evaluate_self_correction.py       held-out eval, Unsloth engine
src/evaluate_self_correction_vllm.py  same measurement, vLLM engine (2 batched passes)
src/push_to_hub.py        push a saved adapter directory to HF Hub manually
src/notify.py             pipeline progress -> Telegram (stdlib only, runs outside any venv)
src/tg_inbox.py           collects Telegram messages -> outputs/tg_inbox.jsonl
```

**`src/prompts.py` exists to enforce an invariant, not to save typing.**
`_REFLECT_PROMPT_TEMPLATE` used to be duplicated in `build_dataset.py` and
`evaluate_self_correction.py` with a comment saying the two copies must stay
byte-identical. A drift between them is exactly the train/inference mismatch that
the `verifier_detail` fix was introduced to remove. All three call sites now read
one constant, so they cannot diverge.

## Setup

1. Create `.env` (never commit this file):

   ```
   DEEPSEEK_MODEL=...
   DEEPSEEK_API_KEY=sk-...
   DEEPSEEK_BASE_URL=https://api.deepseek.com   # optional, has a default
   HF_TOKEN=hf_...                              # optional, only for pushing adapters
   ```

2. Install — **use two separate virtualenvs**:

   ```
   # vLLM: steps 1 and 4
   uv venv ~/venv-vllm  --python 3.12
   uv pip install --python ~/venv-vllm/bin/python vllm python-dotenv pyyaml sympy datasets huggingface_hub

   # Unsloth: steps 2 and 3
   uv venv ~/venv-unsloth --python 3.12
   uv pip install --python ~/venv-unsloth/bin/python unsloth
   uv pip install --python ~/venv-unsloth/bin/python -r requirements.txt
   ```

   This is not fastidiousness. Measured on one machine, the two stacks resolve to
   *different CUDA majors*: vLLM pulled `torch 2.11.0+cu130`, Unsloth pulled
   `torch 2.10.0+cu128`. Installed into one environment, the second overwrites the
   first, and the failure surfaces at **training** time — after the GPU hours for
   step 1 are already spent.

   `uv` is used rather than `pip` because rented boxes frequently lack both `pip`
   and `python3-venv` while `sudo` needs a password. `uv` is a static binary that
   creates venvs without `ensurepip` and installs without `pip`, so neither
   missing piece matters. Plain `pip install unsloth` works fine where pip exists.

   **Also required for step 1 (vLLM's `torch.compile` builds CUDA kernels at
   runtime):** `python3-dev` for `Python.h`, plus `nvcc` and `ninja` on the path.
   `_ensure_build_toolchain()` in `generate_attempts.py` handles the last two
   automatically — it points `CUDA_HOME` at the pip-provided
   `nvidia-cuda-nvcc` package and puts the venv's `bin/` on `PATH`, so no system
   CUDA toolkit and no root are needed. Only `python3-dev` needs a package
   install. If the compile path still fails, set `VLLM_ENFORCE_EAGER=1` **and**
   `VLLM_USE_FLASHINFER_SAMPLER=0` — see [Hardware notes](#hardware-notes).

   **Hardware requirement:** GPU compute capability ≥ 7.5 (T4, L40S, A100, 4090,
   H100, ...). Check with `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
   V100/Volta (CC 7.0) and anything older is **not supported** — see
   [Hardware notes](#hardware-notes) for the full filter before renting a box.

3. Set `small_model.name_or_path` in `configs/phase1.yaml` to the model you're using.

## Pipeline (fixed order — later steps read earlier steps' output)

| Step | Command | Needs GPU? |
|---|---|---|
| 1 | `python -m src.generate_attempts` | Yes (vLLM) |
| 2 | `python -m src.build_dataset` | No — API only |
| 3 | `python -m src.train_sft` | Yes (Unsloth) |
| 4 | `python -m src.evaluate_self_correction_vllm --adapter <repo_or_path>` | Yes (vLLM) |

Step 4 is the metric that matters — training loss alone does not tell you
whether the model learned to self-correct (see [Key Findings](#key-findings)).

Steps 1 and 2 are both **resumable**: they skip `problem_id`s already recorded in
`attempts.jsonl` / `build_dataset_seen_ids.txt`. Re-running after adding problems
only processes the new ones, so scaling up never re-spends GPU or API budget on
work already done.

**Steps 1 and 2 refuse to run twice concurrently** (`src/run_lock.py`). Two
instances appending to the same output file produce duplicate records and, for
step 2, pay the API twice for the same problem. This was a documented rule that
got violated in practice; it is now enforced by a PID lock rather than a comment.

**Step 2 is the long pole and does not use the GPU.** It is one API call per
failed attempt against a reasoning model with an 8000-token budget — sequentially
that is 9–15 hours for ~1300 problems, with a rented GPU sitting idle throughout.
It is now threaded (`deepseek.max_workers`, default 16); the step is bound by
network latency, not CPU. Measured: 16 workers, no rate limiting, ~500 API calls
in roughly 25 minutes.

**Two engines for step 4.** `evaluate_self_correction_vllm.py` exploits the
measurement's natural two-phase shape — solve everything, verify, then reflect
only on the failures — so it is two batched `generate()` calls instead of two
sequential calls per problem. The Unsloth version is kept as a cross-check.
Numbers from the two engines will never match exactly (different kernels, different
RNG); at `temperature=0.7` even two runs of the same engine differ. Tighten the
comparison by raising the problem count, not by picking an engine.

**Before running full-scale on rented GPU hours:** run steps 1–2 on the small
sample problem set first, check the stats printed at the end of step 2
(`total / already_correct / kept / discarded`), and only then scale to the
full problem set.

## Hardware Notes

- Code uses Unsloth's `FastLanguageModel` directly — requires GPU CC ≥ 7.5
  (Turing/Ampere/Ada/Hopper). Verified in practice: current Unsloth/Axolotl
  releases require torch ≥ 2.11/2.3, and torch has dropped CC 7.0 (Volta/V100)
  kernel support since ~2.3 — `torch.cuda.is_available()` still returns `True`
  on V100 but compute silently errors or produces wrong results.
- This project has run across several GPU rental platforms (root-SSH VPS,
  Kaggle T4, JupyterHub L40S, Google Colab T4) — see `instructionAI/environment.md`
  for platform-specific connection notes.

### Choosing a rental GPU — filter by compute capability, not by VRAM

The question to ask a provider before paying is **"what compute capability?"**,
never "how many GB?". VRAM does not compensate for a missing CC level: a 24GB
card below the threshold runs nothing in this pipeline, while a 16GB T4 runs all
of it.

| Tier | Cards | Verdict |
|---|---|---|
| Ampere+ (CC ≥ 8.0) | A10, A40, A100, L4, L40S, 3090, 4090, 5090, H100 | **Target this.** bf16 + FlashAttention-2 + tensor cores |
| Turing (CC 7.5) | T4, RTX 2080 | Absolute floor. Works, but no bf16, no FA2 |
| Volta (CC 7.0) | V100 | **Reject** — silent wrong compute, see above |
| Pascal and older (CC ≤ 6.1) | P40, P100, P4, M60, K80 | **Reject** — see below |

**Pascal (P40/P100) is a trap worth naming explicitly**, because rental listings
advertise the P40 as "24GB VRAM · 12 TFLOPS" at an attractive hourly price:

- CC 6.1 — below the 7.5 floor. Unsloth will not run (`train_sft.py`,
  `evaluate_self_correction.py`), and vLLM requires SM ≥ 7.0 so
  `generate_attempts.py` will not run either. That is the entire GPU pipeline.
- No tensor cores, no bfloat16.
- The advertised 12 TFLOPS is **FP32**. FP16 throughput on GP102 is capped at
  1/64 of FP32 (~0.18 TFLOPS) — a hardware characteristic, not a driver issue.
  Even after rewriting all three scripts to plain `transformers + peft`, it would
  be far slower than a T4 for this workload.

**Sizing, once CC ≥ 8.0 is satisfied:** 24GB (A10 / 3090) is the sweet spot —
enough to run vLLM on a 7B in fp16 for `generate_attempts.py` with **no
`load_in_4bit`**, which removes `bitsandbytes` from that stage entirely. That
matters: the observed failure on a 16GB T4 was not VRAM per se but the forced
4-bit path pulling in a `bitsandbytes` build that needed a CUDA 13.x runtime the
box did not have (`libnvJitLink.so.13: cannot open shared object file`). More
VRAM fixes that failure by avoiding the quantization stack, not by holding more
weights.

**Keep `load_in_4bit: true` for `train_sft.py` even on a large card.** The
current 40.5% self-correction number was measured on a QLoRA adapter; switching
to bf16 LoRA changes a variable and makes new results non-comparable to that
baseline. Raising `training.max_seq_length` back from 2048 to 4096 *is* safe and
expected — it was lowered only to fit a 16GB card, not for any design reason.

### vLLM's runtime compile toolchain

`torch.compile` builds CUDA kernels **when the engine starts**, so step 1 depends
on a working compiler toolchain — and each missing piece only surfaces after
~5 minutes of model loading. Encountered in order on one box:

| Error | Cause | Fix |
|---|---|---|
| `fatal error: Python.h` | no `python3-dev` | install it (needs root) |
| `Could not find nvcc` | no system CUDA toolkit | point `CUDA_HOME` at the pip `nvidia-cuda-nvcc` package — no 3GB install, no root |
| `FileNotFoundError: 'ninja'` | calling `<venv>/bin/python` without activating, so `<venv>/bin` is off `PATH` | prepend the interpreter's directory to `PATH` |
| `CUDA compiler and toolkit headers are incompatible` | `nvcc` / `nvrtc` / `crt` pip packages at different versions | align versions, or bypass (below) |

**`enforce_eager` alone is not a general escape hatch.** `flashinfer` JIT-compiles
its sampling kernels regardless of eager mode — it sits in the sampling step, not
the forward pass — so the last row above survives `VLLM_ENFORCE_EAGER=1`. The
working fallback disables both:

```
VLLM_ENFORCE_EAGER=1 VLLM_USE_FLASHINFER_SAMPLER=0 python -m src.generate_attempts
```

Eager mode costs less than expected on this workload: 1324 prompts completed in
1m36s (13.8 req/s, ~2450 output tok/s). Large batches saturate the GPU, which
hides the per-kernel launch overhead that makes eager slow at batch size 1.

## Scaling the problem set

`prepare_public_datasets.py` appends from splits that are **disjoint from the
held-out eval range**. MBPP partitions by `task_id` into fixed blocks, and the
boundaries are load-bearing:

| Split | task_id | Count | Use |
|---|---|---|---|
| prompt | 1–10 | 10 | training |
| test | 11–510 | 500 | 11–160 training; **161+ is held-out eval** |
| validation | 511–600 | 90 | training |
| train | 601–974 | 374 | training (fully consumed) |

`prepare_code_extra()` hard-rejects `split="test"` for this reason.

**Pre-merged MBPP dumps are a trap.** Redistributions that ship all 974 tasks as
one flat `.jsonl` with no split labels (e.g. the Kaggle `mpwolke/mbppjsonl`
dataset) look like ~1000 new code problems. They are the same MBPP with the only
thing protecting eval integrity stripped out. Training on them means training on
the eval set, it cannot be undone, and the symptom is simply that **the numbers
get better** — nothing errors. Always pull from a source with explicit splits.

**Any new code source must ship runnable tests.** `CodeVerifier` needs an
`entry_point` and a list of `assert`s. A question/solution corpus cannot be used:
generating tests for it would mean an LLM writing the tests, reintroducing the
LLM-judging-LLM dependency the whole design rejects. Rejected on this basis:
Kaggle `bhaveshmittal/python-programming-questions-dataset` (13k rows, columns
`Instruction`/`Input`/`Output` where `Output` is solution code, not tests).

Full reasoning and the candidate-source comparison in
[`instructionAI/data_pipeline.md`](instructionAI/data_pipeline.md).

## Sandbox / Safety

`CodeVerifier` executes model-generated code in a subprocess with a timeout and
memory limit — adequate for self-generated/trusted problem sources (current
scope). If extended to untrusted problem sources (e.g. crawled from the
internet), this needs proper containerization (docker/firejail) first —
subprocess isolation alone is not sufficient for adversarial code.

## Known Issues / Limitations

See [`note.txt`](note.txt) section 3 for the full list. Headline items:

- The published 40.5% was measured on 135 samples vs. the planned 800–1500 — a
  pipeline-validation number, not a final one. The in-progress scale-up run
  targets ~640.
- **The training mix is ~87% code and cannot be balanced by adding math
  problems** — see Key Finding 4. Fixing it requires harder math (MATH), which
  needs a symbolic-answer verifier that does not exist yet.
- MBPP `full/train` is exhausted (374/374). Only ~100 further code problems are
  available without touching the eval reserve.
- Math verifier's answer extraction can mis-parse stray formatting characters
  (e.g. a stray backtick) leaked from the model's own markdown-style output.
- No test yet for sycophancy (model changing a *correct* answer when falsely
  told it's wrong) or multi-round correction (note.txt requires no repeated
  errors after 2–3 rounds; current eval only tests one round).
- `build_dataset` discards attempts whose reasoning exceeds the 8000-token
  budget (`finish_reason=length`). Observed ~6 of the first 500 — a logged,
  accepted loss rate, not a bug to chase to zero.

## References

- Chen, K-Y., Su, F-Y., Chiang, J-H. (2026). *The Self-Correction Illusion:
  LLMs Correct Others but Not Themselves.* [arXiv:2606.05976](2606.05976v1.pdf)
