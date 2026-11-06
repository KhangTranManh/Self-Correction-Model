# Phase 15 dev diagnostic, amendment P15-A5

Recorded 2026-11-03 before running the P15-A5 rescore. This is a frozen,
CPU-only decomposition of the existing complete P15-A3 audit. It generates no
new model output, changes no model or adapter weight, and does not open a
protected evaluation split.

## Question

P15-A3 scored an answer with the official lenient parser. P15-A4 showed that a
deterministic suffix normalizer can recover numeric answers already present in
some outputs and raise unchanged A0 probe joint score from 31/80 to 45/80.
Therefore strict correctness mixes at least two effects: whether a usable
answer appears in the completion and whether the official parser accepts its
final-answer format.

P15-A5 measures those effects separately before authorizing another GPU
intervention. The format-insensitive measure below is called
`rescued_correct`, not pure reasoning or semantic correctness: it is still an
answer-extraction heuristic and cannot certify the quality of intermediate
reasoning.

## Frozen source and scoring

Use only `results/p15_a3_audit.jsonl`, whose expected SHA-256 is
`ef2a05b5e4c534451028a2b8cf97fa0787bb381f83171ce0e9b2d9542562bf79`,
and the exact P15-A2/A3 40-episode manifest, whose expected SHA-256 is
`1839aba51f447c4f0da8b1b95a8fea9775072a81736f50f1faf96e418ac1c2e9`.
Require 40 completed episodes, one completion for every episode/item/arm,
80 probes and 40 controls per arm, and one identical recorded sampling seed
across A0, A1, and A2 for each item.

Reapply the frozen Phase 15 parser and detector to every completion in all
three arms:

- `strict_correct`: the official lenient parse equals the reference answer.
- `rescued_correct`: the frozen rescued numeric parse equals the reference.
- `format_clean`: the frozen detector does not report `S-format`.
- `strict_joint`: `strict_correct` and `format_clean`.
- `rescued_joint`: `rescued_correct` and `format_clean`.
- `correct_but_strict_unparseable`: rescued correct but not strict correct.
- `wrong_but_clean`: format-clean but not rescued correct.

Reference answers are used only after generation for scoring. They do not
select, repair, or alter any output. Report arm totals separately for probes
and controls, plus paired fixes and harms for A1 versus A0 and A2 versus A0 on
all five binary outcomes. Also check the recomputed strict labels against the
labels saved in the audit.

## Frozen development gate

Authorize definition of a new matched-seed IA3 GPU diagnostic only if A2 LoRA
passes all three conditions on `rescued_correct`:

1. Probe gain over A0 is at least 4/80 (five percentage points).
2. Paired probe fixes are strictly greater than paired probe harms.
3. Control score is no worse than A0 by more than 1/40.

The 4/80 threshold is a pragmatic resource gate, not a significance claim. It
asks whether a non-format LoRA signal is large and directionally consistent
enough to justify another two-to-three-hour T4 run. Failure closes the planned
IA3 comparison for now; it does not prove that IA3 or weight updates can never
help. Passing the gate authorizes writing and validating an IA3 protocol, not
launching GPU generation without that protocol.

## Result and decision

The rescore covered 40/40 episodes and all 120 matched-seed items per arm.
The source hashes matched, the protected split remained closed, and there
were zero disagreements with the saved correctness or all-error joint labels.

On probes, rescued correctness was A0 45/80, A1 44/80, and A2 49/80. Thus the
context arm's apparent strict gain (31 to 37) disappeared after answer rescue:
it made 6 rescued-correctness fixes and 7 harms relative to A0. LoRA retained
a smaller +4/80 rescued gain, with 13 fixes and 9 harms. Its strict +9/80 gain
therefore decomposes into a +4/80 answer-content difference and a further
format-acceptance difference under this extractor.

On controls, rescued correctness was A0 36/40, A1 38/40, and A2 34/40. LoRA
made 3 fixes and 5 harms, a net -2/40. The IA3 definition gate therefore
failed its control condition even though both probe conditions passed. No IA3
GPU run is authorized from this branch. This is evidence of an unstable
tradeoff--a small probe gain accompanied by unrelated-control regression--not
proof that LoRA improved general reasoning or that IA3 cannot work elsewhere.

All 360 outputs had a numeric rescued parse. The saved detailed result is
`P15_A5_RESCORE.json`.
