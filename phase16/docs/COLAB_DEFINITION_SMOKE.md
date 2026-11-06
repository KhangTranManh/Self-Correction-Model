# Phase 16 Colab definition smoke

Completed 2026-11-03 in a separate **CPU** Colab CLI session named
`phase16-definition`. This checked the mechanical definition of
C-upper-inclusive. It did not generate model output, run LoRA, screen a task
pool, or open confirmation data. The session was stopped after downloading
the audit.

## Inputs and method

- Local scorer: `../scripts/score_boundary.py`, SHA-256
  `4cabc19a099497ca5b9d5f388fb3d8f7be6102c71cbd82aaa6f9f5e86cf52b21`.
- Smoke script: `../scripts/colab_definition_smoke.py`, SHA-256
  `02255070a59f4acec245ba712146f5cdd5bef11d0262bd79f2b23ce8e6fff2d0`.
- The script evaluates trusted, hand-written correct and `< R` versions of
  count, sum and select tasks. Each version gets ten oracle cases, including
  two independent upper-boundary pairs. The scorer receives the resulting
  transcript. No model-written Python is executed.
- Colab reported Python 3.13.18. The session was CPU-only.

## Verified result

| Variant | Correct implementation | Exclusive-upper implementation |
|---|---|---|
| count | `correct`, 10/10 | `upper_boundary_signature`, 6/10, two pairs |
| sum | `correct`, 10/10 | `upper_boundary_signature`, 6/10, two pairs |
| select | `correct`, 10/10 | `upper_boundary_signature`, 6/10, two pairs |

Downloaded artifact: `../results/colab_definition_smoke.json`, SHA-256
`a43a237fe527f1d735d96d60c9488198619b90c842076b61d60536de8e20e2bc`.
The artifact's scorer hash matched the local source, and the three labels,
case counts, pair counts and no-model/no-GPU flags were checked locally.

## Reproduction outline

Create a fresh CPU Colab CLI session, create `/content/phase16`, upload
`score_boundary.py` there, then execute `colab_definition_smoke.py` with
`colab exec --file`. Download `/content/phase16/definition_smoke.json` and
stop the session. The smoke script refuses to overwrite an existing remote
audit; use a fresh session for repetition.

This is an engineering result only. The source pool, isolated runner for
model-written code and baseline error prevalence are still undefined.
