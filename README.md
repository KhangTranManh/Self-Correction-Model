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

## Status (2026-07-19)

| | |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct, QLoRA (4-bit) via Unsloth |
| Training data | 143 samples (target per plan: 800–1500 — current run is a pipeline-validation pilot, not final scale) |
| Trained adapter | [`Kxck/AGI_v1`](https://huggingface.co/Kxck/AGI_v1) (private) on HF Hub |
| Held-out eval set | 30 GSM8K + 30 MBPP problems (offset 150, disjoint from training data) |

**Latest measured self-correction rate** (`self_corrected / initial_wrong`, held-out set,
counting only responses that completed the `### Sửa lại` format — see
[Known issues](#known-issues--limitations)):

| Reflect-turn role | Math | Code | Overall |
|---|---|---|---|
| `user` (baseline, matches current trained adapter) | 0.0% | 28.6% | 22.9% |
| `tool` (prompt-only change, no retrain — see [Findings](#key-findings)) | 33.3% | 28.6% | 29.7% |

Success criteria from the project plan (not yet met — pilot scale):
self-critique accuracy ≥75–80%, self-correction success rate ≥70%, no repeated
errors after 2–3 correction rounds.

**In progress:** regenerating training data with the reflect turn re-labeled to
role `tool` (informed by the finding below) for the next training round.

---

## Key Findings

1. **Role-labeling the reflect turn matters, independent of model capability.**
   Per ["The Self-Correction Illusion" (arXiv 2606.05976)](2606.05976v1.pdf), LLMs
   fail to self-correct errors sitting in their own reasoning not from a
   reasoning deficit but from a lack of *addressability* — no learned habit of
   treating a claim inside their own output as a nameable, rejectable object.
   Re-presenting the identical error under an external chat-template role
   (`user`/`tool`/`memory`) instead of leaving it in `<thought>` measurably lifts
   correction rates, with **no retraining required**. Switching the reflect
   turn's role from `user` to `tool` (since `verifier_detail` is genuinely a
   tool/checker output) took math self-correction from 0% → 33.3% on the
   currently trained adapter, with no change to code (code's traceback is
   already addressable regardless of role).

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
src/train_sft.py          QLoRA SFT (Unsloth + trl)
src/evaluate_self_correction.py  held-out eval: does the model actually self-correct?
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
| 1 | `python -m src.generate_attempts` | Yes |
| 2 | `python -m src.build_dataset` | No (needs DeepSeek API) |
| 3 | `python -m src.train_sft` | Yes |
| 4 | `python -m src.evaluate_self_correction --adapter <repo_or_path>` | Yes |

Step 4 is the metric that matters — training loss alone does not tell you
whether the model learned to self-correct (see [Key Findings](#key-findings)).

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

## Sandbox / Safety

`CodeVerifier` executes model-generated code in a subprocess with a timeout and
memory limit — adequate for self-generated/trusted problem sources (current
scope). If extended to untrusted problem sources (e.g. crawled from the
internet), this needs proper containerization (docker/firejail) first —
subprocess isolation alone is not sufficient for adversarial code.

## Known Issues / Limitations

See [`note.txt`](note.txt) section 3 for the full list. Headline items:

- Training set is 143 samples vs. the planned 800–1500 — current results are
  pipeline-validation, not final numbers.
- Math verifier's answer extraction can mis-parse stray formatting characters
  (e.g. a stray backtick) leaked from the model's own markdown-style output.
- No test yet for sycophancy (model changing a *correct* answer when falsely
  told it's wrong) or multi-round correction (note.txt requires no repeated
  errors after 2–3 rounds; current eval only tests one round).

## References

- Chen, K-Y., Su, F-Y., Chiang, J-H. (2026). *The Self-Correction Illusion:
  LLMs Correct Others but Not Themselves.* [arXiv:2606.05976](2606.05976v1.pdf)
