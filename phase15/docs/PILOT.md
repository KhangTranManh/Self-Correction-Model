# Phase 15 dev pilot, amendment P15-A1

**Written 2026-10-27, before any pilot generation.** The user directed this run
to stay on Kaggle. This amends the preregistration's 48–80 GB FP16 pilot. It
does not open an evaluation split and it does not run arm A4.

The machine constants are in `kaggle/pilot_v1/pilot_spec.json`.

## What changed

- GPU: one private Kaggle script, `khangkxcp/phase-15-dev-pilot-v1`, on the
  T4 image that completed the engineering smoke.
- Execution: Kaggle runs only the named code file. The helpers, spec, screen,
  and reserve are packed into `phase15_kaggle_pilot_standalone.py`, which
  unpacks them and then runs the same pilot. Version 1 stopped in about one
  second because those helpers were not importable. No generation happened.
- Precision: NF4 QLoRA, double quantization, FP16 compute. Packages are
  transformers 4.46.3, peft 0.13.2, accelerate 1.0.1, bitsandbytes 0.50.2.
- Learner: `Qwen/Qwen2.5-7B-Instruct` only. Llama-3.1-8B-Instruct is not in
  this run.
- Solutions: temperature 0.7, top-p 1, 768 new tokens, one seeded sample for
  the solve base rate, and a second seeded sample only to build a one-right
  judge pair.
- Judgments: greedy, 768 new tokens, unconstrained Phase 9 judge prompt.
  No required `Verdict:` line, so a format error can still be class 2.
- Inner update: rank 8, alpha 16, 3 steps, learning rate 5e-4, the same
  attention and MLP modules as the smoke. These are the predeclared starting
  settings, not a finished hyperparameter search.
- Self-edits pass the local gate. One retry is allowed. A rejected lesson is
  not trained.

Results from this run are evidence for this Kaggle NF4 protocol. They are not
an FP16 result.

## Data

Source: MATH training split, levels 1–3, kept only when the answer is one
constant and the Phase 1 verifier accepts it, and the question key is absent
from the Phase 1–14 scan. Rank: SHA-256 of `phase15_pilot_v1|20261027` and the
question key.

- Screen: the first 100 ranked problems. Base rates are measured here.
- Reserve: the next 300. Used only to fill episodes for a kept type.
- The remainder is unallocated. It is not the final evaluation set.

A type is kept when its rate is at least 10% and at least 20 eligible items
were scored. Solve rates use sample 1. `S-integer` is eligible only on count
questions. Judge rates use one-right pairs only.

## Episodes

For each kept type, up to 40 disjoint episodes: one teaching item, two probes,
one control. The teaching text must not contain a probe or control gold answer
or an 8-word span of that problem. Arms A0, A1, A2, and A3. The wall clock is
7 hours. The kernel saves every completion and stops cleanly at the wall.

The harness go/no-go is decided only when every kept type has its 40 episodes.
A shorter run is a partial pilot. If no type is kept, that base-rate result is
complete: those types do not enter the episode stage.

## What this run does not do

It does not meta-train, does not freeze the main protocol, and does not choose
the reserved final dataset. MultiArith remains rejected.

## Partial outcome (2026-11-02)

Kernel v2 reached the seven-hour wall after 31 S-format episodes and before
the J-format episode stage. The audit SHA-256 is
`ebfe072f22630b0aeff01b0242e674dd1d86c6cde9e7fd9614c3b61a931263ec` and
matches the downloaded file. Local rescoring reproduced every one of the 473
solution classifications/correctness labels and all 38 judgment labels.
All 31 started episodes completed and reset to the frozen initialization.

The screened base rates were S-format 36/100 and J-format 10/38; those were
the only kept types. Partial S-format joint probe scores (correct and free of
the target error) were A0 24/62, A1 29/62, and A2 28/62. A3 ran on only 21
episodes because ten self-edits remained rejected after retry; it scored
19/42 versus matched A0 at 17/42. Its matched control score was 14/21 versus
A0 at 15/21. These are descriptive scores without a completed pilot or a
go/no-go decision.

The downloaded pool exposes a protocol-capacity contradiction. The available
S-format records can form 40 leakage-safe disjoint episodes, but the 38 judge
rows can form only one J-format episode under the frozen disjointness and
leakage rules. The implementation requires 40 episodes for every kept type,
so J-format cannot complete under P15-A1. No continuation kernel was launched.
The next GPU run requires a recorded amendment that either expands judge
screening substantially or changes/drops the J-format episode requirement.

Amendment P15-A2 is now recorded separately in `PILOT_A2.md`. It closes
J-format as infeasible under this pool and freezes a coherent 40-episode
S-format completion run. No P15-A1 outcome is discarded or pooled into that
new run.
