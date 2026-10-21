# AGI Behavior-First Research

Can a small language model notice, judge, and repair its own mistakes? Every answer is checked by a deterministic
verifier (SymPy for math, test execution for code); an LLM is never the correctness oracle. The goal is behavior
(self-critique and self-correction), not domain knowledge.

## Main result: the paper (Phases 9–14)

**Published preprint (Phase 14):** Manh Khang Tran, *Why Self-Arbitration Looks Like Guessing: A Diagnostic Study
of Small Language Models*, Zenodo, 2026. [doi:10.5281/zenodo.23541462](https://doi.org/10.5281/zenodo.23541462) ·
[Zenodo record](https://zenodo.org/records/23541462)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23541462.svg)](https://doi.org/10.5281/zenodo.23541462)

LaTeX source of the Zenodo version: [paper/zenodo/main.tex](paper/zenodo/main.tex) (PDF:
[paper/Self_Arbitration_Zenodo.pdf](paper/Self_Arbitration_Zenodo.pdf)); working draft [paper/manuscript/main.tex](paper/manuscript/main.tex);
arXiv package in `paper/arxiv/`; remaining author tasks in [paper/PAPER_TODO.md](paper/PAPER_TODO.md); all numbers in
[paper/summary/SUMMARY_ALL.md](paper/summary/SUMMARY_ALL.md).

A 7B model solves a problem twice, gets two different answers, and is asked which of its own solutions is right.
Four models (Llama-3.1-8B, Mistral-7B, Self_Correction_v1, Qwen2.5-7B-Instruct) × four datasets (ASDiv, MATH
levels 1–3, GSM8K, SVAMP), both presentation orders, deterministic labels.

| Finding | Evidence | Status |
|---|---|---|
| Judged in one order, the choice looks like guessing | 45–64% right in the 16 cells | descriptive |
| **But the skill is real and general** | skill index above 0 in 16/16 cells; E1 passed in 4/4 preregistered endpoint cells (+0.32 to +0.36) | **preregistered** |
| Position bias costs little | 0–2.5 accuracy points (≤ 1.9 with a probit link) | exploratory |
| **The loss is the judge not making a valid choice** | only 51–90% of judgments pick A or B; the rest are format errors (up to 22.8 points for Qwen), loops cut at the token cap (up to 13.9), re-solving or a third answer (2.6–16.4) | exploratory |
| Part of it is an answer-format artefact | re-reading the final line lifts Qwen from 45–64% to 72–88% | exploratory |
| Forcing a "Verdict: A/B" line removes most invalid choices | valid 79–99%; gap 2–14 instead of 6–43 points | exploratory |
| Averaging over both orders adds little | E2 passed in 1/4 endpoint cells (Llama MATH +5.0 [0.9, 9.3]); close to Ren et al. 2023 | **preregistered** |
| Hidden states know which solution is right | probe AUC 0.66–0.90 across datasets; beats surface features consistently only for Self_Correction | exploratory |
| **Self-arbitration does not beat voting** | judge loses to vote@3 in 15/16 cells (−1.7 to −9.6 points); self-check ties vote@3; vote@5 beats every two-candidate strategy | exploratory |
| Probe-weighted vote@3 helps for some models | above vote@3 out of sample for Llama and Mistral (+1.7 to +4.4), not for Self_Correction or Qwen; still below vote@5 | exploratory |

Open question, not a claim: the untrained Qwen parent has a 2–3× larger skill index than its self-correction
fine-tune, but each model judged its own pairs. The fine-tune's SFT data carry a "later answer is right" prior in
513/513 training rows, which is *consistent with* its second-position preference.

## The self-arbitration line (Phases 9–14)

| Phase | Question | Result |
|---|---|---|
| 9 | Can the model check itself with independent attempts? | vote@5 +8 to +10.5 points (Holm p < 0.001); disagreement catches 82–85% of wrong answers; picking between two solutions at chance (44–52%) |
| 10 | Does SFT on its own correct judgments teach the choice? | No: 52.1% → 50.7%; trails voting |
| 11 | Does DPO on correct vs incorrect judgments teach it? | Likelihood moves (67%), choices do not (+2.8, n.s.); strong order bias |
| 12 | Does judging both orders reveal a skill? (SVAMP) | **Yes, preregistered:** consistent verdicts 74.9% right after order-swapped DPO (68.7% untrained); self-check non-inferior to vote@3; vote@5 still best |
| 13 | Does it transfer to GSM8K, and can invented answers be removed? | Transfers (79.3% consistent); forced verdict cuts invented answers to 2.1%; no judge strategy beats vote@5; the answer is readable in hidden states (~70%) and DPO does not change that |
| 14 | Is the skill general across model families and datasets? | **E1 4/4 (general), E2 1/4 (conditional)**; full 4 × 4 grid, bridge, regression and probes. **Produced the paper** ([doi:10.5281/zenodo.23541462](https://doi.org/10.5281/zenodo.23541462)) |

Reports: [Phase 9](phase9/docs/FINAL_REPORT.md), [10](phase10/docs/FINAL_REPORT.md),
[11](phase11/docs/FINAL_REPORT.md), [12](phase12/docs/FINAL_REPORT.md), [13](phase13/docs/FINAL_REPORT.md),
[14](phase14/README.md).

**What the evidence supports overall.** Every accuracy gain in this project comes from a harness around an
unchanged model: blind re-solving, probe routing, voting. At the model level the skill to judge exists, but it is
hidden by invalid choices, and training (SFT, DPO) changed how the choice is expressed, not the internal signal.
The Phase 0 goal of a model that reliably recognizes and repairs its own errors is only partly reached.

## Earlier phases (1–8)

| Phase | Question | Result |
|---|---|---|
| 1 | Can verified SFT teach critique and correction? | Produced `Kxck/Self_Correction_v1`; SFT mainly teaches the correction format; autonomous math correction weak |
| 2 | Verifier-labeled preferences (KTO)? | Pipeline validated, no finished adapter |
| 3 | Router KEEP vs REVISE? | Decision-Only V1 65.0% balanced accuracy, REVISE recall 49.0%; preference tuning did not improve detection |
| 4 | On-policy / selective-correction training? | GRPO, SFT, DPO, blind preferences, repeated review: no reliable gain |
| 5 | Is there a correctness signal before feedback? | Frozen probes generalize (71–74% balanced); neutral review does not use it safely |
| 6 | Can readout or a harness bridge signal to repair? | Detection exists, repair only 5–10% even when told the answer is wrong |
| 7 | Does seeing the earlier answer block repair? | Blind re-solving fixed 25–32 of 80 wrong answers vs 4–8 when the answer was visible |
| 8 | Resampling, distraction, or anchoring? | A second blind attempt alone gains ~6 points; probe → blind re-solve beats keep-all, not blind-all |

Each phase folder has its own README and final report.

## Next: Phase 15 (draft)

Can a 7B model learn to turn one or two verifier-corrected mistakes into a weight update that changes its
behavior on new problems, with the correction no longer in context, at least as well as a harness? Draft design,
nothing run: [phase15/docs/PREREGISTRATION.md](phase15/docs/PREREGISTRATION.md).

## Data rules

1. Correctness comes from a deterministic verifier, not an LLM judge.
2. Wrong answers are natural model failures; no synthetic wrong answers in canonical training.
3. Frozen evaluation sources never enter training or dataset selection.
4. KEEP and REVISE share the same neutral-review template distribution.
5. Model-visible rationales may be stored for audit but are not called hidden chain-of-thought.
6. Every promoted adapter has a reproducible config, dataset hash, frozen evaluation, and local backup.
7. Endpoints are preregistered and opened once; every change is an amendment recorded before the run it affects.

## Repository layout

```text
AGI/
├── paper/          # Paper: manuscript/, results/, scripts/, figures/, summary/SUMMARY_ALL.md, PAPER_TODO.md, arxiv/
├── phase1/ … phase13/   # One folder per closed phase (README, docs/FINAL_REPORT.md, scripts, data)
├── phase14/        # Generalization study: 4 models × 4 datasets, hidden-state probes, DATA_CATALOG.md, RUN_STATUS.md
├── phase15/        # Draft: learning to update from one or two corrections
├── instructionAI/  # Cross-phase architecture and data rules
├── serving/        # vLLM serving profiles and strategy client
└── outputs/        # Generated outputs and adapters (gitignored; Phase 14 hidden states on F:\AGI_data)
```

Reproducing the paper (CPU only, from saved outputs): [paper/README.md](paper/README.md). Phase 3 structure and
model registry: [phase3/docs/CODEBASE.md](phase3/docs/CODEBASE.md),
[phase3/configs/experiments.yaml](phase3/configs/experiments.yaml). Local audits:
`python phase3/scripts/validate_local_state.py [--hash-adapters]` and
`.venv-phase4-verify/Scripts/python.exe phase4/scripts/validate_local_state.py --hash-adapters`.

## Serving

[`serving/serve.sh`](serving/README.md) serves the solver checkpoints and the Phase 10/11 judge adapters
(profiles `original`, `v3`); `serving/client.py` runs the `single`, `vote` and `self_check` strategies with the
experimental prompts. Phase 3 routers use `phase3/scripts/serving/serve_phase3.sh`. Both bind to localhost; only set
`PHASE3_HOST=0.0.0.0` on a network you intend to expose (no authentication is added).

## Secrets and large artifacts

- `.env` is local-only; never read it for documentation work and never commit it.
- `outputs/` is gitignored; adapters and hidden states are backed up separately.
- Raw attempts and frozen result files are research evidence: never rewrite or delete them.
