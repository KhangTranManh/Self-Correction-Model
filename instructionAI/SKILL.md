---
name: instructionAI
description: Persistent project memory for this repo. Read this file first, then the files it lists that match the task.
---

# instructionAI

Self-correction research on small language models. A deterministic verifier
(SymPy or executed tests) is the correctness oracle. An LLM is never the
oracle.

## Files in this folder

| File | Role |
|---|---|
| `SKILL.md` | Entry point |
| `handoff.md` | Current continuation snapshot for a new AI session |
| `architecture.md` | Phase ownership, serving, security |
| `data_pipeline.md` | Frozen data rules through Phase 13 |

`handoff.md` is the living snapshot. Prefer it over older `note.txt` text that
still describes Phase 15 as ungenerated.

## Rules that break work if violated

1. Never train on Phase 5–14 protected or holdout sources, or on Phase 15's
   reserved final set.
2. Never treat an LLM as the correctness oracle.
3. Do not start Phase 16 GPU / A0 screening until the CPU contract, oracle,
   history, and sandbox gates in `phase16/docs/` pass.
4. Do not `git add` HuggingFace parquet/jsonl dumps under `phase16/data/`.
   They exceed GitHub's 100 MB file limit. Raw sources go to Drive, not Git.
5. Local `main` can sit one commit behind `origin/main`. Pull before push.
