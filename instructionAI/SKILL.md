---
name: agi-phase1-self-correction
description: Phase 1 pipeline that teaches a small 7B model to recognize its own mistakes and self-correct, using verifiable domains (math/code) before any open-domain knowledge training.
---

# What this project is

A fine-tuning pipeline for a small (~7B) LLM, teaching the foundational skill "know
you're wrong → self-critique → self-correct" before any domain-specific knowledge is
trained in. The skill is taught on VERIFIABLE domains (math, code) where correctness is
checked by a program (sympy / unit tests), not by an LLM's subjective judgment — this is
the core design principle (see conventions.md).

A larger reasoning model (currently GLM-5.2 via an OpenAI-compatible proxy; originally
DeepSeek — env var names still say `DEEPSEEK_*` for historical reasons, see
environment.md) plays "grader": it only critiques attempts the small model has already
produced and that an objective verifier has already confirmed are wrong.

## File index

| File | Covers |
|---|---|
| [architecture.md](architecture.md) | File/folder tree, one-line role per file, dependency graph, the seven-stage runtime pipeline plus serving path |
| [conventions.md](conventions.md) | Code conventions, safe-vs-dangerous-to-change list, design rationale |
| [environment.md](environment.md) | GPU/hardware constraints, exact dependency pins, remote-box setup/reconnect procedure |
| [data_pipeline.md](data_pipeline.md) | Verifier design, the verifier_detail grounding fix, thinking-trace capture, evaluation methodology |
| [results.md](results.md) | Canonical latest run, trained-vs-base metrics, interpretation, serving validation, and next evidence needed |

`note.txt` (repo root, NOT part of this folder) is the user's own rolling success-criteria
/ status doc — read it for current targets and progress, but it is not maintained as part
of this documentation set (see conventions.md's exclude-list rationale).

## Critical rules (violating these breaks something, not just "suboptimal")

1. **Unsloth requires GPU compute capability >= 7.5 — check `nvidia-smi --query-gpu=
   compute_cap` before assuming the current code works.** `train_sft.py`,
   `generate_attempts.py`, and `evaluate_self_correction.py` currently import and use
   `unsloth.FastLanguageModel` directly (no plain-transformers fallback in the code).
   This is fine on Turing/Ampere/Ada/Hopper (T4, L40S, A100, 4090, H100...) but **breaks
   on V100 (Volta, CC 7.0)** — empirically verified: Unsloth's and Axolotl's own
   dependency pins force `torch>=2.3`/`torch>=2.11`, and torch dropped CC 7.0 kernel
   support around that line (`torch.cuda.is_available()` still returns `True` but real
   computation errors or silently misbehaves). If the GPU changes back to a V100-class
   card, these three files need to be reverted to the plain `transformers + peft + trl`
   pattern (AutoModelForCausalLM + BitsAndBytesConfig + peft.get_peft_model, no Unsloth
   import) with `torch==2.2.1+cu121` pinned — see git history for that version, or
   environment.md for the full rationale.

2. **Pipeline stage order is fixed by data dependency, not a preference:**
   `prepare_public_datasets.py` → `generate_attempts.py` (GPU) → `build_dataset.py`
   (API only, no GPU) → `prepare_training_dataset.py` → `train_sft.py` (GPU) →
   `evaluate_self_correction.py` (GPU), or `evaluate_self_correction_vllm.py` when the
   merged model is already served.
   Each stage reads the previous stage's output file. Running out of order fails with
   a missing-file error, not silent wrong behavior.

3. **Never run two instances of `generate_attempts.py` or `build_dataset.py`
   concurrently** (including against the same or different problem sets). Both open
   their output file in write mode; two processes racing on the same path corrupts it
   (this happened once — produced a truncated, partially-interleaved `attempts.jsonl`).
   Always check `ps aux | grep <script>` before launching one of these.

4. **The self-correction "reflect" prompt MUST embed the real verifier error message**
   (`verifier_detail`), in `build_dataset.py` (training data generation),
   `evaluate_self_correction.py`, and `evaluate_self_correction_vllm.py`, using the
   identical template. Without this,
   the small model learns to hallucinate a plausible-sounding but ungrounded diagnosis
   instead of reading and reacting to the actual error — confirmed empirically (0%
   self-correction success on held-out code problems before this fix was added).

5. **`math_verifier.py`'s number normalization is order-sensitive**: strip currency
   prefix → decide thousands-separator (`70,000`) vs Vietnamese-decimal-comma (`84,5`)
   BEFORE any blanket comma→dot replacement → then strip trailing units (`cm²`, `kg`).
   Three real false-positive bugs were found and fixed here (unit suffix, currency
   prefix, thousands separator) — each one made the verifier reject a numerically
   correct answer as wrong. Any future edit to this function must re-run the regression
   cases listed in conventions.md.

6. **`.env` is per-machine and never synced.** The local (`d:\AGI\.env`) and remote
   (`~/AGI/.env` on the GPU box) files are two separate, independent files — `.env` is
   deliberately excluded from every `scp`/`rsync` of the project. Each machine's `.env`
   must be created/edited directly on that machine.

7. **`DEEPSEEK_MODEL` / `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` are provider-agnostic
   env var names**, not a hard dependency on DeepSeek the company. `deepseek_client.py`
   just needs an OpenAI-compatible endpoint. Currently configured for GLM-5.2 through a
   proxy (`https://api.vilao.ai/v1`, model id `op/z-ai/glm-5.2`). Don't assume the
   variable name tells you the actual provider — check `.env` (or ask) if it matters.

8. **The rented GPU box may be a brand-new container on reconnect**, even at the same
   IP:port. Symptom: SSH reports "REMOTE HOST IDENTIFICATION HAS CHANGED". If so, the
   disk is fresh — code, installed packages, `.env`, and all `data/processed/`
   /`outputs/` artifacts are gone and must be recreated (see environment.md for the
   exact reconnect/setup procedure).

9. **`CodeVerifier` is not a security sandbox.** It runs model-generated code via
   `subprocess` with a timeout and memory limit — adequate for trusted problem sources
   (GSM8K/MBPP-style) but not for untrusted/adversarial code.

10. **`train_sft.py`'s `push_to_hub` path has no local-save fallback on failure.** If
    `configs/phase1.yaml`'s `huggingface.repo_id` isn't a real repo the token can write
    to, the push raises AFTER training completes, and the trained adapter is lost
    (never written to disk). Verify `repo_id` and `HF_TOKEN` before a real training run,
    or set `training.push_to_hub: false` to save locally instead.

11. **Never trust a self-correction score without checking `format_incomplete` and raw
    headings.** A mojibake-corrupted `### Sửa lại` regex once turned 12 valid correction
    sections into a false 0/12 result. Keep the HTTP evaluator's heading matcher encoded
    with Unicode escapes, allow same-line corrected content, and inspect raw generations
    whenever all corrections fail for the same formatting reason. The canonical current
    result and invalid-run signature are recorded in `results.md`.
