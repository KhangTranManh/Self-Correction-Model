# Conventions

## Core design principle — don't violate without flagging it

The "wrong attempt" that gets corrected must always come from the actual small model
doing real inference on a problem — never fabricated or imagined by the larger
API model. This is why `generate_attempts.py` runs before `build_dataset.py`
(architecture.md's stage order) and why it never calls the external API. The reasoning:
training data must match the small model's *real* error distribution, or the learned
self-correction skill won't transfer to how the model actually fails in practice. If a
future change proposes having the API model "imagine" a plausible wrong attempt to save
GPU time, that's a deliberate regression of this principle — flag it to the user rather
than just doing it.

Similarly: verification of both the original attempt AND the API model's proposed
correction is always done by the objective `Verifier` (sympy / unit tests), never by
trusting the API model's own claim that its fix is correct. `build_dataset.py` discards
corrections that don't pass re-verification — this is intentional data-quality gating,
not a bug to "optimize away" for a higher keep-rate.

## Code style

- Comments, docstrings, and print/log messages in the codebase are Vietnamese without
  diacritics (easier to type/grep, avoids encoding issues in some terminals). User-facing
  prompt templates and system prompts embedded in strings (what gets sent to the LLM or
  shown to an end user) use full Vietnamese diacritics.
- Config split: all hyperparameters and file paths live in `configs/phase1.yaml`; all
  secrets live in `.env`. Both are loaded together once, in `src/config.py`, into a
  single `Config` dataclass — no script reads `.env` or the yaml file directly itself.
- Dataclasses over dicts for structured data crossing function boundaries
  (`Problem`, `Attempt`, `VerifierResult` in `src/data/schema.py`) — dicts are only used
  for the final JSON-serializable record shapes (attempts.jsonl rows, ChatML messages).

## Adding a new problem domain (e.g. "logic")

Touches all of: `src/data/schema.py` (`Problem.domain` Literal type), a new
`load_*_problems()` in `problem_sources.py`, a new `Verifier` implementation in
`src/verifier/`, a new prompt template in `generate_attempts.py`'s
`_PROMPT_TEMPLATES` dict, and a new entry in `build_dataset.py`'s `verifiers` dict
(and `evaluate_self_correction.py`'s, if held-out eval should cover it too). There is no
single central registry — missing one of these locations silently breaks that domain
rather than raising an error naming the gap.

## Known gotchas (each cost real debugging time — don't reintroduce)

- **`math_verifier.py` number parsing order matters.** Currency prefix (`$`, `€`, ...)
  must be stripped BEFORE deciding whether a comma is a thousands separator
  (`"70,000"` → `70000`, English-style, common in GSM8K model outputs) or a decimal
  separator (`"84,5"` → `84.5`, Vietnamese-style) — a blanket `.replace(",", ".")` before
  that decision silently turns `"$70,000"` into `70.000` (i.e. 70), which then compares
  unequal to a genuinely correct `70000` reference. Regression cases that must keep
  passing (see the test block that was run inline during development, not committed as
  a formal test suite — consider adding one if this file is touched again):
  `"$18"` vs ref `"18"` → pass; `"84 cm²"` vs ref `"84"` → pass; `"$70,000"` vs ref
  `"70000"` → pass; `"1,234,567"` vs ref `"1234567"` → pass; `"84,5"` vs ref `"84.5"`
  → pass; a genuinely wrong value must still fail (don't over-correct into always
  passing).
- **`SyntaxError` must be in the caught-exception tuple** in `MathVerifier.verify()`'s
  parse try/except — `sympy.parse_expr` can raise it (not just `SympifyError`/
  `ValueError`), and an uncaught `SyntaxError` crashes the whole `build_dataset.py` run
  on a single malformed model answer instead of recording it as a failed verification.
- **`tokenizer.pad_token_id` must be set** (fallback to `eos_token`) and
  `attention_mask` passed explicitly to `model.generate()` in any inference script —
  otherwise transformers emits a correctness warning ("pad token is same as eos token")
  and generation behavior for edge cases is unreliable.
- **DeepSeek/GLM `max_tokens` must budget for `reasoning_content`, not just the final
  JSON.** The reasoning trace can run past 1000 tokens before the model even starts the
  JSON answer; a `max_tokens` too small truncates mid-reasoning (`finish_reason=length`,
  empty or cut-off `content`). 8000 was chosen after measuring real traces; a small
  fraction of genuinely hard problems still hit this ceiling and get discarded — that's
  an acceptable, logged loss rate, not a bug to chase to zero.
- **Never launch a second instance of a script that opens the same output file in write
  mode** while one is already running (see SKILL.md rule 3) — always
  `ps aux | grep <script_name>` before starting `generate_attempts.py` or
  `build_dataset.py` on the remote box.

## Safe vs. dangerous to change

**Safe to change freely:** prompt wording in `generate_attempts.py`'s
`_PROMPT_TEMPLATES`; hyperparameters in `configs/phase1.yaml` (epochs, batch size, LoRA
rank, etc.); how many problems `prepare_public_datasets.py` pulls (CLI args); which
public dataset(s) feed a given domain.

**Dangerous — verify the specific reasoning in data_pipeline.md before changing:**
the `_REFLECT_PROMPT_TEMPLATE` wiring between `build_dataset.py` and
`evaluate_self_correction.py` (must stay byte-for-byte identical, and must keep
embedding real `verifier_detail` — see SKILL.md rule 4); the `torch` version pin
(environment.md); anything that would let the "attempt" step and the "critique" step
run on the same underlying reasoning source (violates the core design principle above).
