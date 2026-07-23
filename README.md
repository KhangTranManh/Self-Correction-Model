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

### Phase 2 — From imitation to discrimination  ·  *next*

The Phase 1 ceiling is a wrong *objective*, not a tuning gap. Base models are
optimized to *produce* a good answer; the assistant turn is always the target,
never the object under evaluation — so the model never learned the second-order
stance of judging its own prior output, and hallucinates when forced into it.

Phase 2 changes the training objective from *imitate a good correction* to *prefer
a real critique over a fabricated one*, using preference learning (DPO/ORPO) where
the **verifier supplies the labels for free**: a model's own hallucinated critique
and its sycophantic flips become explicit *rejected* examples. Full method
comparison (preference learning, verifier-stance training, process rewards,
calibration, scale control) in the *Phase 2* section of
[`phase1/note.txt`](phase1/note.txt).

### Later phases — *not yet designed*

Knowledge integration, multi-step tool use, and longer-horizon reasoning are
downstream of a model that can reliably self-assess. They wait until the behavioral
base is solid.

---

## Repository layout

```
.
├── README.md            ← this file (project vision, roadmap)
└── phase1/              ← everything for Phase 1 (self-contained)
    ├── README.md         ← Phase 1 operational readme
    ├── note.txt          ← full research log (the authoritative record)
    ├── src/              ← pipeline code (core / data / llm / pipeline / ops)
    ├── configs/          ← phase1.yaml (hyperparams, paths)
    ├── data/             ← problem sets + processed training data
    ├── instructionAI/    ← architecture / conventions / environment docs
    └── ...               ← eval logs, reports, reference paper
```

Each phase is a **self-contained folder**: its code resolves all paths relative to
its own root, so phases don't interfere and any one can be run in isolation. To run
Phase 1, work from inside `phase1/` (see its README).

---

## Status

Phase 1 closed with an honest negative on its headline target and a clear,
evidence-backed reason why — which is exactly the input Phase 2 needs. The next
concrete step is assembling preference data for Phase 2 from the existing Phase 1
eval logs (no GPU required), before committing further compute.
