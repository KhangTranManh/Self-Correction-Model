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
└── src/
    ├── config.py                  loads .env + phase1.yaml -> Config dataclass (single source of truth for all scripts)
    ├── data/
    │   ├── schema.py               Problem / Attempt / VerifierResult / CorrectionRecord dataclasses
    │   └── problem_sources.py      load_math_problems(), load_code_problems() — JSONL -> list[Problem]
    ├── verifier/
    │   ├── base.py                 Verifier protocol (verify(problem, candidate_text) -> VerifierResult)
    │   ├── math_verifier.py        sympy-based objective answer checking
    │   └── code_verifier.py        subprocess-based unit test execution
    ├── deepseek_client.py          OpenAI-compatible chat client — critique_and_correct(), captures reasoning_content
    ├── prepare_public_datasets.py  GSM8K/MBPP (HF datasets) -> data/problems/*.jsonl
    ├── generate_attempts.py        small model self-attempts (no external API) -> attempts.jsonl
    ├── build_dataset.py            verify -> API critique/correction -> re-verify -> phase1_sft.jsonl
    ├── train_sft.py                QLoRA SFT via transformers+peft+trl -> outputs/phase1_lora/ or HF Hub
    ├── evaluate_self_correction.py held-out eval: does the trained adapter actually self-correct?
    ├── push_to_hub.py              standalone: push an existing local adapter dir to HF Hub
    └── test_deepseek_connection.py cheap API config/connectivity sanity check — no GPU needed
```

## Dependency graph

- `config.py` is imported by every other module in `src/` — it is the only place `.env`
  and `configs/phase1.yaml` are read.
- `data/schema.py` + `data/problem_sources.py` are used by `generate_attempts.py`,
  `build_dataset.py`, and `evaluate_self_correction.py`.
- `verifier/math_verifier.py` and `verifier/code_verifier.py` are used by
  `build_dataset.py` and `evaluate_self_correction.py` (never by `generate_attempts.py`,
  which only produces attempts, doesn't judge them).
- `deepseek_client.py` is used only by `build_dataset.py` and
  `test_deepseek_connection.py`. `train_sft.py`, `generate_attempts.py`, and
  `evaluate_self_correction.py` never call the external API — they only touch the local
  GPU model.

## Runtime flow (the 5 stages — see data_pipeline.md for the reasoning behind each)

```
prepare_public_datasets.py
    reads: HF Hub datasets (gsm8k, mbpp)
    writes: data/problems/math.jsonl, data/problems/code.jsonl
        │
        ▼  (GPU)
generate_attempts.py
    reads: data/problems/*.jsonl
    writes: data/processed/attempts.jsonl
        │
        ▼  (no GPU — API calls to whatever provider .env points at)
build_dataset.py
    reads: data/problems/*.jsonl, data/processed/attempts.jsonl
    writes: data/processed/phase1_sft.jsonl
        │
        ▼  (GPU)
train_sft.py
    reads: data/processed/phase1_sft.jsonl
    writes: outputs/phase1_lora/  (or pushes to HF Hub if training.push_to_hub: true)
        │
        ▼  (GPU)
evaluate_self_correction.py
    reads: outputs/phase1_lora/, fresh held-out problems (offset past what training used)
    writes: nothing — prints a stats dict + self-correction success rate to stdout
```

Every script is invoked as `python -m src.<module_name>` from the project root (relative
imports like `from src.config import ...` require this — running a file directly from
inside `src/`, e.g. `cd src && python foo.py`, fails with `ModuleNotFoundError`).
