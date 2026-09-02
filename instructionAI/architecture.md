# Architecture

## File/folder tree

```
d:\AGI/
├── .env                          secrets (per-machine, never synced — see SKILL.md rule 6)
├── .gitignore                    excludes .env, data/processed/, outputs/, __pycache__
├── README.md                     setup, current result, and run instructions
├── benchmark_testcase.txt        benchmark backlog and P0/P1/P2 priority definitions
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
│       ├── phase1_sft.jsonl       build_dataset.py output — verified ChatML correction data
│       └── phase1_sft_train.jsonl prepare_training_dataset.py output — loss-ready SFT rows
├── outputs/
│   ├── phase1_lora/               trained LoRA adapter
│   ├── Self_Correction_v1_merged/ standalone merged BF16 checkpoint for vLLM
│   ├── eval_*_vllm_*.{json,jsonl} historical small HTTP evaluations
│   ├── p0_*_100_summary.json      canonical base/trained P0 summaries
│   ├── p0_*_100_log.jsonl         complete P0 prompts, outputs, reasoning text, scores
│   ├── p0_benchmark_report.md     canonical human-readable P0 comparison
│   └── p0_code_format_audit.json  strict-vs-lenient code-format diagnostic
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
    ├── prepare_training_dataset.py compact/mask context -> phase1_sft_train.jsonl
    ├── train_sft.py                QLoRA SFT via Unsloth+TRL -> outputs/phase1_lora/
    ├── evaluate_self_correction.py held-out eval: does the trained adapter actually self-correct?
    ├── evaluate_self_correction_vllm.py same eval through a running vLLM HTTP API
    ├── evaluate_p0_vllm.py        B1/B2/B3/B7/B8 base-vs-tuned benchmark via vLLM
    ├── export_merged_for_vllm.py   merge adapter into BF16 base and upload a standalone model
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
- `prepare_training_dataset.py` converts verified builder output into prompt/completion
  training rows while preserving every verified correction target.
- `evaluate_self_correction_vllm.py` uses the same `Problem` schema and objective
  verifiers as the local evaluator, but generations come from `/v1/chat/completions`.
- `evaluate_p0_vllm.py` extends the HTTP evaluation into paired behavioral branches:
  B1 and B3 share selected wrong attempts; B2, B3, and B7 share selected correct
  attempts; B8 loads SVAMP and HumanEval as OOD sources. It writes full raw JSONL and
  a compact JSON summary. It does not call the correction-data API.
- `export_merged_for_vllm.py` loads the original BF16 base, applies the saved PEFT
  adapter, saves merged safetensors, and uploads the folder using `HF_TOKEN`.

## Runtime flow (seven stages plus optional merged serving)

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
        ▼  (no GPU)
prepare_training_dataset.py
    reads: data/processed/phase1_sft.jsonl
    writes: data/processed/phase1_sft_train.jsonl
        │
        ▼  (GPU)
train_sft.py
    reads: data/processed/phase1_sft_train.jsonl
    writes: outputs/phase1_lora/
        │
        ├──────────────────────────────┐
        ▼  (GPU, direct adapter)       ▼  (GPU merge, then vLLM)
evaluate_self_correction.py
    reads: outputs/phase1_lora/, fresh held-out problems (offset past what training used)
    writes: evaluation logs/summary
                                export_merged_for_vllm.py
                                    reads: BF16 base + outputs/phase1_lora/
                                    writes: merged checkpoint + HF model repository
                                        │
                                        ▼
                                vLLM OpenAI-compatible server
                                        │
                                        ▼
                                evaluate_self_correction_vllm.py
                                    reads: held-out datasets + vLLM responses
                                    writes: JSONL records + JSON summary
                                        │
                                        ▼
                                evaluate_p0_vllm.py
                                    reads: GSM8K/MBPP + SVAMP/HumanEval + vLLM API
                                    branches: guided / false feedback / neutral review
                                    writes: p0_*_100_log.jsonl + summary.json
```

The P0 base and fine-tuned runs are sequential because one 24 GB GPU can host only
one merged BF16 7B model with the configured vLLM cache. Switch weights, keep every
evaluation argument identical, then restore the fine-tuned service after comparison.

Every script is invoked as `python -m src.<module_name>` from the project root (relative
imports like `from src.config import ...` require this — running a file directly from
inside `src/`, e.g. `cd src && python foo.py`, fails with `ModuleNotFoundError`).
