# Architecture

## File/folder tree

```
d:\AGI/
├── .env                          secrets (per-machine, never synced — see SKILL.md rule 6)
├── .gitignore                    excludes .env, data/processed/, outputs/, __pycache__
├── README.txt                    setup + run instructions for a human operator
├── note.txt                      user's own rolling success-criteria/status notes (not part of this doc set)
├── requirements.txt               pinned deps + WHY comments (torch pin rationale lives here)
├── configs/
│   └── phase1.yaml               hyperparams, file paths, non-secret provider config
├── data/
│   ├── problems/
│   │   ├── math.jsonl            GSM8K subset — {id, question, reference_answer}
│   │   └── code.jsonl            MBPP subset — {id, question, entry_point, tests}
│   └── processed/                 generated at runtime, gitignored
│       ├── attempts.jsonl         generate_attempts.py output — {problem_id, attempt_idx, text}
│       └── phase1_sft.jsonl       build_dataset.py output — final ChatML SFT training data
├── outputs/
│   └── phase1_lora/               trained LoRA adapter (gitignored; may be pushed to HF Hub instead)
└── src/                          grouped by concern — see "Package layout" below
    ├── config.py                  loads .env + phase1.yaml -> Config dataclass (imported by ~everything)
    ├── core/                      shared building blocks, no single stage owns them
    │   ├── schema.py               Problem / Attempt / VerifierResult / CorrectionRecord dataclasses
    │   ├── prompts.py              SINGLE source for every shared prompt template + build_reflect_message()
    │   ├── model_loading.py        shared Unsloth model/tokenizer loader
    │   └── run_lock.py             PID lock — refuses a second concurrent instance of the same script
    ├── data/                      what a problem is, and what "truth" is
    │   ├── problem_sources.py      load_math_problems(), load_code_problems() — JSONL -> list[Problem]
    │   └── verifiers/
    │       ├── base.py             Verifier protocol (verify(problem, candidate_text) -> VerifierResult)
    │       ├── math.py             sympy-based objective answer checking
    │       └── code.py             subprocess-based unit test execution
    ├── llm/
    │   └── api_client.py           OpenAI-compatible chat client — critique_and_correct(), captures reasoning_content
    ├── pipeline/                  the 5 ordered stages + eval support (run as python -m src.pipeline.<name>)
    │   ├── prepare_datasets.py     [stage 1] GSM8K/MBPP -> data/problems/*.jsonl, split-aware
    │   ├── generate_attempts.py    [stage 2] small model self-attempts (vLLM, no API) -> attempts.jsonl
    │   ├── build_dataset.py        [stage 3] verify -> API critique -> re-verify -> phase1_sft.jsonl (threaded)
    │   ├── train_sft.py            [stage 4] QLoRA SFT via Unsloth+trl -> outputs/phase1_lora/ or HF Hub
    │   ├── evaluate.py             [stage 5] held-out eval, Unsloth engine (kept as cross-check)
    │   ├── evaluate_vllm.py        [stage 5] same measurement, vLLM engine — 2 batched passes
    │   ├── eval_common.py          held-out loader + log format + report, shared by BOTH eval scripts
    │   └── reframe_dataset.py      rewrite the reflect turn's framing in a dataset — NO API cost
    └── ops/                       utilities outside the pipeline data flow
        ├── notify.py               pipeline progress -> Telegram; stdlib only, runs outside any venv
        ├── tg_inbox.py             collect Telegram messages -> outputs/tg_inbox.jsonl (does NOT execute them)
        ├── push_to_hub.py          push an existing local adapter dir to HF Hub
        └── check_connection.py     cheap API config/connectivity sanity check — no GPU needed
```

### Package layout — grouping by concern

| Package | Holds | Rule of thumb |
|---|---|---|
| `src/config.py` | the one Config object | stays at root; imported by nearly every module |
| `src/core/` | schema, prompts, model loading, run-lock | shared, stage-agnostic; imports nothing heavy so **both venvs** load it |
| `src/data/` | problem sources + objective verifiers | "what is a problem, what is truth" |
| `src/llm/` | the OpenAI-compatible API client | the only module that talks to the external model |
| `src/pipeline/` | the 5 ordered stages + eval helpers | run as `python -m src.pipeline.<name>` |
| `src/ops/` | notify, inbox, push, connectivity check | side utilities, not part of the data flow |

Stage modules keep their previous names (`generate_attempts`, `train_sft`, …); only
the import path gained a `pipeline.` segment. So the invocation is now
`python -m src.pipeline.generate_attempts`.

### Why `prompts.py` and `eval_common.py` exist

Not to save typing — to make two invariants structural rather than remembered.

`prompts.py` holds `REFLECT_PROMPT_TEMPLATE` and `build_reflect_message()`. That
template used to be duplicated in `build_dataset.py` and `evaluate_self_correction.py`
with a comment saying the copies must stay byte-identical; a drift between them is
exactly the train/inference mismatch the `verifier_detail` fix was built to remove.
All three call sites (both eval scripts plus dataset construction) now read one
constant, so they cannot diverge. `build_reflect_message()` additionally encodes the
`memory` → `system` + `<memory>` mapping — see SKILL.md rule 13 for why constructing
that turn inline is dangerous.

`eval_common.py` holds the held-out loader, log format and report. If each eval script
loaded its own problems, the two engines' numbers would not be comparable — and being
comparable is the entire reason both are kept.

Both modules deliberately import nothing heavy (no vllm, no unsloth), because they are
imported from **both** virtualenvs. `generate_attempts.py` imports vllm at module
level, so `evaluate_self_correction.py` importing `_build_prompt` from it used to drag
vLLM into the Unsloth environment.

## Dependency graph

- `src/config.py` is imported by nearly every module — the only place `.env` and
  `configs/phase1.yaml` are read.
- `core/schema.py` + `data/problem_sources.py` are used by the pipeline stages
  `generate_attempts`, `build_dataset`, and both `evaluate*`.
- `data/verifiers/math.py` and `data/verifiers/code.py` are used by `build_dataset`
  and both `evaluate*` (never by `generate_attempts`, which only produces attempts,
  doesn't judge them).
- `llm/api_client.py` is used only by `build_dataset` and `ops/check_connection`.
  `train_sft`, `generate_attempts`, and both `evaluate*` never call the external API —
  they only touch the local GPU model.

## Runtime flow (the 5 stages — see data_pipeline.md for the reasoning behind each)

```
src.pipeline.prepare_datasets
    reads: HF Hub datasets (gsm8k, mbpp)
    writes: data/problems/math.jsonl, data/problems/code.jsonl
        │
        ▼  (GPU — vLLM)
src.pipeline.generate_attempts
    reads: data/problems/*.jsonl
    writes: data/processed/attempts.jsonl
        │
        ▼  (no GPU — API calls to whatever provider .env points at)
src.pipeline.build_dataset
    reads: data/problems/*.jsonl, data/processed/attempts.jsonl
    writes: data/processed/phase1_sft.jsonl
        │
        ▼  (GPU — Unsloth)
src.pipeline.train_sft
    reads: data/processed/phase1_sft.jsonl
    writes: outputs/phase1_lora/  (or pushes to HF Hub if training.push_to_hub: true)
        │
        ▼  (GPU — vLLM)
src.pipeline.evaluate_vllm   (or .evaluate for the Unsloth engine)
    reads: outputs/phase1_lora/, fresh held-out problems (offset past what training used)
    writes: an eval log jsonl + prints the decomposed metric to stdout
```

Every script is invoked as `python -m src.<package>.<module>` from the project root
(relative imports like `from src.config import ...` require this — running a file directly from
inside `src/`, e.g. `cd src && python foo.py`, fails with `ModuleNotFoundError`).
