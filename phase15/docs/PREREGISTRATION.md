# Phase 15 preregistration (draft)

**Status: draft written on 2026-10-22; nothing has been generated. Open decisions are listed at the end and must be
settled, recorded in `RUN_STATUS.md`, and the file frozen before the first GPU run.** After freezing, any change is
an amendment recorded in `RUN_STATUS.md` before the generation it affects.

## Question

Can a 7B model turn **one or two verifier-corrected mistakes** into a **weight update** that changes its behavior on
**new problems with the same kind of mistake**, with the correction **no longer in context**, and does this do at
least as well as simply putting the correction in context (a harness)?

In short: is fast self-updating in the weights possible at this scale, or is a harness all one gets?

## Motivation

- Phases 1, 10–13: training on hundreds of examples mostly changed the *format* of the model's output, not the
  underlying skill. DPO changed how the judge expresses a choice, and the hidden-state probe showed the internal
  signal itself was unchanged (Phase 13). So a naive update on one or two examples is not expected to work.
- Phase 14: a model judging its own solutions is close to guessing in a single order. Any update signal must
  therefore come from the deterministic verifier (data rule 1), never from the model's own judgment.
- Phase 14 also gives a ready, deterministic taxonomy of **behavior errors** in judging, where most of the lost
  accuracy is: format errors, cut-off loops, re-solving instead of choosing, invented third answers.
- The only approach with a reason to expect success is to **learn the update procedure itself** (meta-learning).
  SEAL (Zweiger et al., 2025, arXiv:2506.10943) trains a model with RL to write its own fine-tuning data
  ("self-edits"), rewarded by the updated model's downstream accuracy. Test-time training (Akyürek et al., ICML
  2025, arXiv:2411.07279) shows that temporary parameter updates at inference can beat in-context learning on
  few-shot tasks. Neither compares a learned in-weights update against a harness on behavior errors with
  deterministic labels, and neither checks whether the update transfers to new problems once the context is removed.

## Scope (Phase 0 rule)

The phase teaches **behavior** (how to avoid a type of mistake), never domain knowledge. An update targets an
error *type*, not a fact. The problems used to teach and the problems used to test are always different, so the
answers to the test problems are never in the update data. Self-edits are checked for leakage (see Integrity).

## Terms

| Term | Definition |
|---|---|
| Error type τ | A behavior mistake with a deterministic detection rule (table below) |
| Teaching item | A problem on which the base model made a mistake of type τ, with the verifier's feedback and the verified correction |
| Episode | One error type τ; k ∈ {1, 2} teaching items; a set of **probe items** (new problems of the same task, where the base model makes τ at a known rate); a set of **control items** (other types and other tasks) |
| Update | Any change applied before the probe and control items are answered. Every episode starts from the base weights; updates are discarded after the episode |
| Probe score | Share of probe items solved without making τ and with a verifier-correct answer (both reported separately) |

## Error types (candidates; final list fixed by the pilot)

Each rule is deterministic and reuses Phase 14 code where possible.

| Task | Type | Detection rule | Phase 14 evidence |
|---|---|---|---|
| Judge | J-format | no valid A/B choice, but the rescued reading matches one solution (Phase 14 class 2) | up to 22.8 accuracy points lost for Qwen |
| Judge | J-loop | hit the token cap and the truncation labeller says "looping" (class 3) | Llama: 56–91% of capped judgments |
| Judge | J-resolve | not capped, re-solves the problem instead of choosing (class 5) | Mistral, Llama |
| Judge | J-third | not capped, gives a number matching neither solution (class 6) | all models |
| Solve | S-format | the final line is not `Đáp số: <number>` (lenient fails, rescued succeeds) | Qwen, Mistral |
| Solve | S-loop | solution hits the 768-token cap while looping | Llama MATH 15.7% capped |
| Solve | S-integer | the problem asks for a count and the answer is non-integer | to be measured |

A type enters the main run only if, in the pilot, the base model makes it on ≥ 10% of eligible items and the pool
has enough items for the planned episodes. Types are split into **meta-training types** and **held-out types**
before any meta-training (see Endpoints).

## Model and data

- **Learner:** one model for the main run, chosen by the pilot from Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct
  (open decision). Development on Qwen2.5-1.5B-Instruct on the local RTX 3060 (pipeline only; no result from it is
  reported as evidence).
- **Pools.** Teaching items, probe items and control items are drawn from problem pools that were never used in
  Phases 1–14. Judge tasks use pairs built from the learner's own conflicting solutions, as in Phase 14 (both
  orders). Candidate sources (open decision, checked for overlap with every earlier phase): the unused part of
  ASDiv, the MATH training split levels 1–3, and one new dataset reserved only for the final evaluation.
- Three disjoint splits by source problem: **meta-train** (for arm A4 only), **dev** (pilot and hyperparameters),
  **evaluation** (opened once).

## Arms

| Arm | What happens before the probe items | Correction in context at probe time? |
|---|---|---|
| A0 | Nothing (base model) | no |
| A1 Harness | The teaching item(s), feedback and correction are put in the prompt | **yes** |
| A2 Naive update | One LoRA update trained directly on the teaching item(s) with the verified correction as target | no |
| A3 Self-edit, untrained | The model reads the teaching item(s) and writes a self-edit: the lesson in general form plus rewritten examples of the correct behavior; one LoRA update on the self-edit | no |
| A4 Self-edit, meta-trained | As A3, after the self-edit writer has been trained with RL on the meta-train split | no |

Secondary combinations, reported but not tested: A1 + A4 (update and context together).

**Inner update (A2–A4), identical for all arms:** LoRA on the attention and MLP projections, rank and steps fixed by
the dev split before the main run, the same learning rate and token budget for every arm. A2 and A3 get the same
number of optimizer steps as A4.

**Meta-training (A4):** ReST-EM-style, as in SEAL: for each meta-train episode, sample several self-edits, apply each
as an inner update, score each on that episode's meta-train probe items, keep the self-edits that improve the probe
score over A0, and fine-tune the self-edit writer on them. Repeat for a fixed number of rounds (open decision). The
reward uses only verifier labels. Meta-train probe items never appear in the evaluation split.

## Endpoints (evaluation split, opened once)

Unit: episode. Confidence intervals by bootstrap clustered by episode and source problem (10,000 resamples). Holm
correction over P1 and P2.

| Endpoint | Comparison | Pass rule |
|---|---|---|
| **P1 — learning in the weights exists** | A4 − A0, probe score on **meta-training types** (new problems) | difference ≥ 5 points and CI above 0 |
| **P2 — not just a harness** | A4 (no context) − A1 (correction in context), same types | non-inferior: CI lower bound above −2 points |
| **P3 — no damage** | A4 − A0 on control items | CI lower bound above −2 points |

Secondary, descriptive:

- **Generalization to held-out types:** P1 and P2 repeated on types never seen during meta-training. This is the
  strongest version of the question ("the model has learned how to learn from a correction").
- A3 − A2 (does writing a self-edit help without training?) and A4 − A3 (what meta-training adds).
- k = 1 versus k = 2 teaching items.
- Verifier-correct accuracy on probe items, not only the absence of τ: an update could remove the behavior while
  making answers worse.
- Comparison at matched compute with sampling: vote@3 and vote@5 on the probe items. Phase 14 showed voting is
  strong; the cost of an inner update is reported in model calls and GPU-seconds.

## Interpretation rules (written in advance)

- **P1 and P2 pass:** "A 7B model can learn to turn one or two corrections into an in-weights update that matches
  a harness on new problems of the same kind." Claimed only for the meta-training types unless the held-out
  analysis also passes.
- **P1 passes, P2 fails:** "In-weights updates help, but less than a harness": at this scale the harness is the
  better design.
- **P1 fails:** "No reliable in-weights learning from one or two corrections at 7B." This is a valid, reportable
  result and closes the direction for this model size.
- **P3 fails:** any gain in P1 is reported together with the damage; no claim of a usable method.
- A2 passing P1 without meta-training would contradict Phases 10–13 and must be checked for leakage before it is
  believed.

## Pilot (dev split, before freezing)

1. Measure the base rate of every candidate type for the candidate learners; keep types with rate ≥ 10%.
2. Run A0–A3 on about 40 dev episodes per kept type. No meta-training.
3. **Go / no-go:** if A1 (harness) gives no gain on a type, that type cannot be corrected from one or two examples
   even in context and is dropped. If A1 gives no gain on any type, the phase stops here and reports that.
4. Fix inner-update hyperparameters (rank, steps, learning rate) on the dev split only.

## Limits to report in advance

- Episodes are independent: every update starts from the base weights. Accumulating many updates over time
  (continual learning, forgetting across a sequence) is not tested here; it would be the next phase.
- One learner model; the result may not hold for other sizes or families.
- Error types are those with deterministic rules; harder behavior errors (misreading the question) are out of
  scope.
- RL meta-training can learn to win the reward on the meta-train types without generalizing; the held-out-type
  analysis and the separate evaluation split are the checks for this.

## Cost estimate (rough, to be revised after the pilot)

- Pipeline development: local RTX 3060 with the 1.5B model.
- Pilot: about 5–10 GPU-hours on a 48–80 GB GPU.
- Meta-training: each self-edit sample costs one LoRA update and one probe evaluation (vLLM with dynamic LoRA
  loading). With about 300 episodes, 4 samples, 2 rounds: about 2,400 inner updates, roughly 40–80 GPU-hours.
- Evaluation: 5 arms × about 200 episodes, roughly 10–20 GPU-hours.
- Total roughly 60–110 GPU-hours, several times Phase 14. Results are copied to `D:\AGI` as each stage finishes.

## Integrity

- Labels come only from the deterministic verifier and the Phase 14 detection rules.
- Splits by source problem; overlap with Phases 1–14 checked and recorded before the pilot.
- Leakage check: no self-edit or teaching item may contain the source text or the gold answer of any probe or
  control item (n-gram and number overlap, rule fixed before the pilot).
- Execution locks as in Phase 14 (`verify_lock.py`: pilot and main), evaluation opened once.
- Every update, self-edit and score is saved so that every number can be recomputed on CPU.

## Open decisions (to settle before freezing)

1. Learner model for the main run (Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct).
2. Problem sources for the three splits, and the dataset reserved for evaluation.
3. Final list of error types and the meta-training / held-out split of types.
4. Inner-update hyperparameters (from the dev split).
5. Number of ReST-EM rounds and self-edit samples per episode.
6. Number of probe and control items per episode, and the number of evaluation episodes (power analysis from the
   pilot's variance).
7. GPU and budget.

## References

- Zweiger, Pari, Guo, Akyürek, Kim, Agrawal. *Self-Adapting Language Models.* arXiv:2506.10943 (2025).
- Akyürek et al. *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning.* ICML 2025,
  arXiv:2411.07279.
- Phase 14 results: `paper/summary/SUMMARY_ALL.md`; error classification: `paper/scripts/phase14_paper_tables.py`
  (`classify`), truncation labels: `paper/scripts/phase14_truncation.py`.
