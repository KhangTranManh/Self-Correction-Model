# Phase 2 — From imitation to discrimination (KTO on the critique step)

Phase 1 hit a ceiling that is **a wrong training objective, not a tuning gap**:
SFT on corrections teaches the *format* of correcting, not the *discriminative
skill* of knowing when one is actually wrong. The model "corrects" largely by being
handed the answer and working backward; remove the answer and it hallucinates an
error or flips a correct answer (sycophancy). See the "Phase 2" section of
[`../phase1/note.txt`](../phase1/note.txt) for the full diagnosis and method survey.

Phase 2 changes the objective from *imitate a good correction* to **prefer a real
critique over a fabricated one**, using preference learning where the **verifier
supplies the labels for free**.

## Why KTO (not DPO)

DPO needs a *pair* (chosen vs rejected) for the same prompt. Phase 1 data has no
such pairs: SFT holds only correct corrections; eval logs hold one correction per
problem. Building clean pairs needs multi-sampling (GPU).

**KTO (Kahneman-Tversky Optimization)** takes *single* examples with a binary
`{desirable, undesirable}` label — which is exactly the shape of the data we already
have. That unpaired-data fit is the *only* reason we pick it. (KTO still uses a
reference model like DPO — `reward = β·log(π/π_ref)`; it is **not** reference-free.
With LoRA, both KTO and DPO avoid loading a separate reference by toggling the
adapter off, so there's no VRAM edge unique to KTO.)

## The three inviolable principles still hold

1. **Correctness decided by a program, not an LLM.** KTO labels come directly from
   the verifier result already recorded (`second_passed`) — no LLM re-judges here.
2. **Mistakes are the model's own.** Undesirable examples come from *this* 7B model's
   real failures. **Coupling to watch:** if the base model changes (e.g. to 9B), the
   7B failure logs can no longer serve as undesirable — they must be regenerated.
3. **Train prompt == infer prompt, byte for byte.** `build_preference.py` reuses
   Phase 1's own `build_reflect_message()` and held-out loader to reconstruct
   prompts, with a self-check that asserts the tool framing matches the SFT data.

## Pipeline

| Step | Script | GPU? | Status |
|---|---|---|---|
| 0. Build KTO seed from Phase 1 outputs (bootstrap, see caveat) | `build_preference.py` | No | **done, tested** |
| 1a. Generate sycophancy negatives (+ measure the risk) | `gen_sycophancy.py` | Yes | defined, needs GPU-time impl+verify |
| 1b. Generate multi-sample failed corrections | `gen_corrections.py` | Yes | **to add** at GPU time |
| 2. KTO-tune the critique behavior (from AGI_v3) | `train_kto.py` | Yes | written to pattern, **not yet run** |
| 3. Evaluate vs AGI_v3 baseline | reuse `../phase1/.../evaluate_vllm.py` | Yes | reuse Phase 1 |

### What "undesirable" we need, and where it comes from

The discrimination skill has two facets; each yields a desirable/undesirable pair,
**labelled by the real verifier** (never an LLM):

| Setup | desirable | undesirable |
|---|---|---|
| Model right, told *falsely* it's wrong (`gen_sycophancy`) | holds firm / pushes back | **flips a correct answer** |
| Model wrong, given *real* feedback, sampled N× (`gen_corrections`) | genuinely fixes it | **fabricates a fix, still wrong** |

Generation quality rules: generous `max_new_tokens` (so negatives are *reasoned
wrong*, not *cut off* — 83/108 of the seed's negatives are the weak cut-off kind and
should be kept separate); use *harder* problems (MATH level 2–3 + code — GSM8K is too
easy to fail); target a few hundred of each so the class ratio lands within KTO's
recommended `[1, 1.33]` weight band against the 675 desirable.

> **CONTAMINATION CAVEAT (important):** the 152 eval-derived examples in the current
> seed come from the **held-out** set (probe logs, `eval_gsm8k_0150+`). They were a
> convenience for *testing the builder only*. They must **not** enter the real KTO
> training run — training on them then evaluating on the same held-out set would be
> training on the test. Real generation (steps 1a/1b) targets the **training split**
> (`gsm8k_train_*`, `mathhard_*`, `mbpp_train_*`), keeping the 600+150 held-out clean
> for measuring Phase 2's effect.

### Step 0 — `build_preference.py` (done, no GPU)

Builds a KTO dataset from Phase 1 outputs. Verified on real data:

```
python build_preference.py \
    --eval-log <path>/probe_baseline.jsonl \
    --eval-log <path>/probe_localize.jsonl \
    --out data/preference/kto_seed.jsonl
# → 783 examples: 675 desirable / 108 undesirable
#   sft_deepseek 631 · eval_self_correct_pass 44 · eval_self_correct_fail 25 · eval_format_incomplete 83
```

- **desirable** = the large model's correct corrections (SFT, 631) + the 7B model's
  own *passing* self-corrections from eval logs.
- **undesirable** = the 7B model's own *failed* / *format-incomplete* corrections.

### Steps 1–3 — need GPU (rent when ready)

Run when a box is rented (sync `phase1/` + `phase2/`, work from inside `phase2/`).
`gen_sycophancy.py` and `train_kto.py` follow Phase 1's vLLM / Unsloth patterns but
are **not yet executed** — they must be verified by a real run, per the project's
"trust only what ran" discipline.

## Known risks to watch

- **Voice vs. correctness confound.** Desirable-from-SFT are the *large model's*
  corrections (different voice); undesirable are the *7B's* own failures. KTO could
  learn "sound like the large model" instead of "be correct." Mitigation: the 7B's
  own *passing* corrections are included as desirable in the model's own voice —
  weight them up, and watch eval for style-mimicry without accuracy gain.
- **Class imbalance.** Seed data is ~6:1 desirable:undesirable — too skewed for
  weights alone to fix. Step 1 (sycophancy) is the main source of real undesirable
  examples and must run before serious training.
- **KTO may not beat the ceiling if it is pure model capacity.** Run a scale control
  (bigger base on the same eval) to size how much of the ceiling is method vs scale
  before over-investing — see `../phase1/note.txt` Phase 2 "Phương án E".

## Layout

```
phase2/
├── README.md               ← this file
├── build_preference.py     ← step 0 (done, no GPU)
├── gen_sycophancy.py       ← step 1 (GPU)
├── train_kto.py            ← step 2 (GPU)
├── configs/phase2.yaml     ← KTO / LoRA hyperparams
└── data/preference/        ← built datasets
```

Base model and secrets are inherited from Phase 1 (`../phase1/.env`,
`../phase1/configs/phase1.yaml`) via Phase 1's `load_config()` — not duplicated here.
