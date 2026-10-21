# Phase 15 — learning to update from one or two corrections

**Status: draft design (2026-10-22). Nothing has been generated.**

Phases 10–13 showed that fine-tuning on hundreds of examples changes the form of the model's output, not the
skill. A harness (re-solving, voting, putting a correction in context) is what has worked. Phase 15 asks whether a
7B model can **learn the update procedure itself**: after one or two verifier-corrected mistakes, the model writes
its own update data ("self-edit"), a single LoRA update is applied, and the model stops making that kind of mistake
on new problems **without the correction in context**. The key comparison is against the harness.

Five arms: no update, harness (correction in context), naive LoRA update on the corrected examples, untrained
self-edit, and a self-edit writer meta-trained with RL (as in SEAL, arXiv:2506.10943). Error types are behavior
errors with deterministic detection rules, taken from Phase 14 (judge format errors, loops, re-solving, third
answers) and from solving (answer format, loops). Domain knowledge is not taught.

| Path | Contents |
|---|---|
| `docs/PREREGISTRATION.md` | Question, error types, arms, endpoints P1–P3, interpretation rules, pilot, costs, open decisions |
| `RUN_STATUS.md` | Decision log and, later, the run record |
