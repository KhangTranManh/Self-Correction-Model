# Phase 15 — learning to update from one or two corrections

**Status: closed after development diagnostics P15-A1 through P15-A5. The S-format branch ended because deterministic output recovery solved its format objective, while LoRA's small remaining rescued-answer probe gain came with control regression. The protected evaluation split and preregistered main endpoints were never opened. See [final report](docs/FINAL_REPORT.md).**

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
| `docs/FINAL_REPORT.md` | Closed development result, decision, and limits |
| `RUN_STATUS.md` | Decision log and, later, the run record |
| `kaggle/smoke_v1/` | Private Kaggle QLoRA engineering smoke; synthetic data only, not evidence |
| `scripts/` | Self-edit gate, Phase 14 error-type wrappers, and the source-overlap audit |
| `docs/PREPARATION.md` | Local preparation results. Not a pilot and not a frozen protocol |
| `docs/PILOT.md` | Amendment P15-A1: the Kaggle NF4 dev pilot |
| `docs/PILOT_A2.md` | Amendment P15-A2: coherent 40-episode S-format completion run |
| `docs/PILOT_A3.md` | Amendment P15-A3: matched-seed A0/A1/A2 development diagnostic |
| `docs/PILOT_A4.md` | Amendment P15-A4: deterministic suffix-normalizer diagnostic |
| `docs/PILOT_A5.md` | Amendment P15-A5: strict versus rescued-answer decomposition and GPU gate |
| `results/` | Recovered P15-A2/P15-A3 audits, summaries, and independent rescores |
| `data/preparation_v1/` | Counts from those checks. No problem text and no split |

The Kaggle smoke established that Qwen2.5-7B-Instruct can load in NF4 on a
Tesla T4, receive isolated three-step LoRA updates, generate under A0–A3, and
return exactly to one frozen adapter initialization after every episode. It
does not establish that updating improves behavior: on 16 synthetic judge
probes, A0/A1/A2/A3 scored 11/9/8/9, while every arm preserved all eight
unrelated controls.

The dev pilot specified by amendment P15-A1 is the private kernel
`khangkxcp/phase-15-dev-pilot-v1`. Version 2 completed at the wall and its
audit was downloaded and rescored locally with zero detector or correctness
disagreements. On 31 partial S-format episodes, joint probe accuracy was
A0/A1/A2 = 24/62, 29/62, and 28/62. A3 ran on 21 episodes and scored 19/42,
versus matched A0 at 17/42; ten self-edits failed the gate. These are
descriptive partial-pilot scores, not a go/no-go result. The screened J-format
pool can form only one leakage-safe disjoint episode, not the required 40, so
the protocol must be amended before more GPU work. See `docs/PILOT.md` and the
rescore artifact under `outputs/phase15_kaggle_pilot_v1/v2/`.

ASDiv has no eligible problem absent
from Phases 1–14. The MATH training split, levels 1–3, has 2,900 verifier-accepted
problems and no exact overlap with the scanned inventories. MultiArith was
checked as a possible reserved set and was not reserved.

P15-A2 completed all 40 frozen S-format episodes, but a post-run review found
that A0, A1, and A2 used different decoding seeds. P15-A3 therefore repeated
the same development comparison using one shared seed per item. Its probe
joint scores were 31/80 for A0, 37/80 for the context harness, and 40/80 for
the direct LoRA update. Relative to A0 on the same sampled items, A1 produced
14 fixes and 8 harms; A2 produced 21 fixes and 12 harms. Control joint scores
were 30/40, 34/40, and 29/40. These are development diagnostics, not protected
evaluation evidence; meta-trained self-editing was not tested in P15-A3.

P15-A4 then applied a frozen deterministic suffix normalizer to the unchanged
P15-A3 A0 outputs, without consulting reference answers or generating new
tokens. It raised probe joint correctness from 31/80 to 45/80 and format-clean
from 38/80 to 80/80, with zero paired correctness harms; controls rose from
30/40 to 36/40. Twenty-nine of 49 normalized outputs nevertheless remained
wrong. This shows that the strongest S-format gain is available as safe output
recovery and further weakens a LoRA-specific interpretation. Hard token
blocking was not run because the postprocessor already reached 100% format
cleanliness without perturbing the model's token distribution.

P15-A5 rescored all P15-A3 outputs to separate strict parser acceptance from
whether the frozen rescued-answer extractor could recover the correct numeric
answer. Probe rescued correctness was A0/A1/A2 = 45/44/49 of 80; paired A2
versus A0 changes were 13 fixes and 9 harms. Controls were 36/38/34 of 40;
A2 made 3 fixes and 5 harms. The frozen development gate required the
A2 control loss to be at most 1/40, but it was 2/40, so no IA3 GPU run is
authorized. The residual LoRA signal is small and trades against unrelated
controls; it is not clean evidence of improved general reasoning.
