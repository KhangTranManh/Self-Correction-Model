# AGI — Behavior-First

A long-horizon research project toward general capability, built on one bet:
**teach an LLM the right *behaviors* before loading it with *knowledge*.** A model
that cannot tell when it is wrong, cannot say "I'm not sure," and cannot revise
itself will only fail more confidently as you make it more knowledgeable. So we
build the behavioral foundation first — on domains where correctness is checkable
by a program, not by another model's opinion.

The project advances in **phases**. Each phase isolates one foundational behavior,
proves (or disproves) that it can be taught to a small model with objective
supervision, and hands a clear, honestly-scoped result to the next phase.

---

## The core bet

Most capability work adds knowledge and hopes good behavior emerges. We invert it:

- **Behavior before knowledge.** Self-correction, calibration, knowing-when-to-stop
  — these are prerequisites, not emergent bonuses.
- **Objective supervision only.** Right/wrong is decided by a program (a symbolic
  math check, a real unit-test run), never by an LLM judging an LLM. If a behavior
  can't be checked mechanically, we don't train on it — we find a domain where it
  can.
- **Small model, honest ceilings.** We work at 7B so that when a behavior *can't*
  be taught, we find out cheaply and say so — a clean negative result is worth more
  than an inflated positive one.

### Three inviolable principles (project-wide)

These carry across every phase. Breaking one voids the result.

1. **Correctness is decided by a program, never by an LLM judging an LLM.**
2. **Mistakes must be produced by the model itself** — never a large model
   imagining a "plausible" wrong answer. The training distribution must match how
   the model actually fails.
3. **The train-time prompt must equal the inference-time prompt, byte for byte.**

---

## Roadmap

### Phase 1 — Verified Self-Correction  ·  `phase1/`  ·  *complete, honestly scoped*

Teach a 7B model to recognize it is wrong and correct itself, on math (GSM8K,
MATH) and code (MBPP), with correctness from sympy / unit tests.

**What it proved:** the mechanism works end to end — the verified critique→correct
loop runs, and self-correction is *real and strong on code*, where a traceback
points at the exact failing line.

**Where it hit a ceiling:** self-correction on *math* stalls at ~32–36% and could
not be lifted — two independent interventions (reflect-turn framing; step-localized
error messages) both came back **negative**. The diagnosis is now evidence-backed:
SFT on corrections teaches the *format* of correcting, not the *discriminative
skill* of knowing when one is actually wrong. The model "corrects" largely by being
handed the answer and working backward; remove the answer and it hallucinates or
flips a correct answer (sycophancy).

**Verdict:** original numeric targets were not met, and we have clean evidence they
are **not reachable with 7B + SFT** — that requires changing something fundamental,
which is Phase 2. Full research log, risks, dead ends and statistics in
[`phase1/note.txt`](phase1/note.txt); operational detail in
[`phase1/README.md`](phase1/README.md).

### Phase 2 — From imitation to discrimination  ·  *in progress*

The Phase 1 ceiling is a wrong *objective*, not a tuning gap. Base models are
optimized to *produce* a good answer; the assistant turn is always the target,
never the object under evaluation — so the model never learned the second-order
stance of judging its own prior output, and hallucinates when forced into it.

Phase 2 changes the training objective from *imitate a good correction* to *prefer
a real critique over a fabricated one*, using **KTO** (chosen over DPO because our
data is unpaired) where the **verifier supplies the labels for free**: the model's
own hallucinated fixes and its sycophantic flips become the *undesirable* signal.

**Progress (2026-07-23):** generated 1741 preference examples from the model's own
behavior on the training split — and measured a **79% sycophancy rate** (told
falsely it was wrong, AGI_v3 caves ~4 out of 5 times). The KTO training pipeline is
verified end-to-end (smoke-passed); a full tuning run is pending a practical config
(the first attempt ran at ~32 s/step — too slow to finish in one rental). Details in
[`phase2/README.md`](phase2/README.md); method survey in the *Phase 2* section of
[`phase1/note.txt`](phase1/note.txt).

### Phase 3 — Selective correction / error discrimination  ·  `phase3/`  ·  *pilot in progress*

Phase 3 turns correction into a decision problem: keep a correct answer under
neutral or misleading feedback, revise a wrong answer under neutral or true
feedback, and preserve normal solve behavior on fresh tasks. Correctness is still
decided only by symbolic checks or executable tests.

**Progress (2026-09-05):** Base and `Self_Correction_v1` attempts were collected
and bucketed on 1,774 GSM8K/MBPP/APPS sources. A validated 860-row behavior
selection exists. Two smaller QLoRA pilots were run directly from the merged
`Kxck/Self_Correction_v1` checkpoint against the same frozen 30-row micro-eval.

The latest 200-row, two-epoch pilot learned the exact KEEP/REVISE contract on
20/20 review cases, cut false-feedback flips from 5/5 to 1/5, and preserved the
final answer on 5/5 neutral-correct cases. It still repaired 0/5 neutral-wrong
cases and regressed to 0/5 behavioral success on both normal solve and regression
recovery. Result: **4/8 gates; do not scale to 860 yet.** Details and exact
artifact status are in [`phase3/README.md`](phase3/README.md).

### Later phases

Knowledge integration, multi-step tool use, and longer-horizon reasoning are
downstream of a model that can reliably self-assess. They wait until the behavioral
base is solid.

---

## Repository layout

```
.
├── README.md            ← this file (project vision, roadmap)
├── phase1/              ← Phase 1: Verified Self-Correction (self-contained)
│   ├── README.md         ← Phase 1 operational readme
│   ├── note.txt          ← full research log (the authoritative record)
│   ├── src/              ← pipeline code (core / data / llm / pipeline / ops)
│   ├── configs/ data/ instructionAI/ outputs/ ...
├── phase2/              ← Phase 2: KTO on the critique step
│   ├── README.md         ← Phase 2 plan + run results
│   ├── gen_preference.py ← generate preference data from the model's own behavior
│   ├── render_kto.py     ← conversational → standard format for trl
│   ├── train_kto.py      ← KTO tuning from AGI_v3
│   └── build_preference.py, configs/, data/preference/
└── phase3/              ← Phase 3: selective correction / error discrimination
    ├── README.md         ← current experiment record and next gate
    ├── build_behavior_selection.py, construct_behavior_pilot.py
    ├── build_behavior_pilot_v2.py, train_behavior_pilot.py
    ├── evaluate_behavior_micro.py, configs/, data/, outputs/
```

Each phase is a **self-contained folder**: its code resolves all paths relative to
its own root, so phases don't interfere. Later phases reuse stable Phase 1
primitives (prompts, verifiers) via explicit path imports rather than duplicating
them. Run each phase from inside its own folder (see its README).

---

## Status

Phase 1 closed with an honest negative on its headline target and a clear,
evidence-backed reason why. Phase 2 produced preference data and a verified KTO
pipeline, but its full run remains unfinished. Phase 3 now has a complete source
inventory and two controlled SFT pilots. The latest pilot proves that the exact
decision contract and resistance to false feedback can be learned, but autonomous
repair and fresh-task retention do not yet coexist. The immediate next step is a
checkpointed one-versus-two-epoch ablation with more fresh-task anchors—not the
full 860-row run.
