# Phase 15 dev pilot, amendment P15-A2

**Written and corrected 2026-11-02, before any P15-A2 model generation.**
This amendment is limited to completing the S-format development pilot that
P15-A1 stopped at the seven-hour wall. It does not open an evaluation split,
change an outcome, or run arm A4.

## Reason for the amendment

P15-A1 screened 100 MATH development problems, kept S-format and J-format by
the base-rate rule, and reached 31 of 40 S-format episodes before the wall.
CPU reconstruction from the downloaded audit found capacity for the original
40 disjoint S-format episodes but only one disjoint J-format episode. The
preregistration also requires enough items for the planned episodes. J-format
therefore cannot enter the main run from this pool and is closed as
infeasible under P15-A1; its partial screen rate is retained as descriptive
information. The J-format requirement is not weakened after seeing outcomes.

## Frozen completion run

- Reconstruct the exact 40 S-format episode assignments from the P15-A1 v2
  audit and its original screen and reserve files.
- Require the 31 completed episode IDs to match the first 31 reconstructed
  IDs. Freeze all 40 assignments in a manifest before launching P15-A2.
- Rerun all 40 episodes in one dedicated process. P15-A1 saved the common
  initialization hash but not the LoRA initialization tensors, so a nine-item
  continuation could use a different random initialization and would not be
  strictly combinable with the first 31. The P15-A1 outcomes remain a
  reproducibility comparison and are not pooled into the P15-A2 score.
- Use a separate private Kaggle kernel and output directory. Do not append to
  or modify the P15-A1 audit.
- Keep Qwen2.5-7B-Instruct, NF4 QLoRA, package versions, prompts, decoding,
  seeds, LoRA rank and modules, three optimizer steps, learning rate, token
  caps, arms A0-A3, gate, retry rule, and adapter-reset checks unchanged.
- Set the original seven-hour wall. If it stops early, the result remains
  partial; do not splice it with P15-A1.
- Declare the S-format pilot complete only after local rescoring reproduces
  all labels, all 40 episodes are complete in P15-A2, every reset matches its
  one frozen initialization, and no episode ID or item is duplicated.

## Decision rule after completion

The S-format harness go/no-go uses all 40 frozen episodes and the original
P15-A1 rule: if A1 gives no gain over A0, S-format is dropped and Phase 15
stops because no feasible type remains. If A1 gives a gain, the result only
supports continuing design on the dev split; it is not evaluation evidence.
A2 and A3 remain descriptive feasibility comparisons. No claim is made for
J-format.
