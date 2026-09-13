# Five-stage router pipeline

**Status: completed historical experiment family; Phase 3 is closed.**

The canonical starting policy is Decision-Only V1. All follow-up artifacts are
grouped under `outputs/phase3_five_stage_pipeline/`; every stage still has an
independent directory so later results cannot overwrite earlier evidence.

| Stage | Experiment | Main output | Promotion gate |
|---|---|---|---|
| 01 | Frozen classifier | `outputs/phase3_five_stage_pipeline/stages/01_frozen_classifier/` | Frozen balanced accuracy and REVISE recall improve without KEEP recall below 65% |
| 02 | Decision-token QLoRA | `outputs/phase3_five_stage_pipeline/stages/02_decision_token/` | Frozen 200-row benchmark beats Decision-Only V1; KEEP recall stays at least 70% |
| 03 | Pair shortcut audit | `outputs/phase3_five_stage_pipeline/stages/03_shortcut_audit/` | No high-risk label cue; flagged pairs are excluded or corrected |
| 04 | Same-origin hard-negative tuning | `outputs/phase3_five_stage_pipeline/stages/04_same_origin/` | Fresh verifier accepts pairs; frozen code and REVISE metrics improve |
| 05 | Verifier-reward GRPO pilot | `outputs/phase3_five_stage_pipeline/stages/05_reward_grpo/` | Beats the best prior promoted stage without policy collapse |

## Required report layout

Each stage must contain:

```text
stage directory/
├── config.yaml                 exact immutable settings
├── run.log                     stdout/stderr
├── status.json                 running/completed/failed and timestamps
├── report.json                 machine-readable metrics and artifact hashes
├── report.md                   human-readable result and decision
├── predictions.jsonl           per-example decisions when applicable
└── final_adapter/              only for weight-updating stages
```

The pipeline-level report lives at
`outputs/phase3_five_stage_pipeline/pipeline_report.md` and compares every
stage to the same Decision-Only V1 frozen baseline. A stage is not called an
improvement from training loss alone.

Rejected adapters were intentionally not downloaded from the GPU. Their
duplicated tokenizer/config snapshots were removed from the active stage tree
and retained only in `outputs/archive/redundant_phase3_20260907/`; no rejected
stage is presented as a loadable checkpoint.

## Local migration map

| Previous local path | Canonical local path |
|---|---|
| `outputs/phase3_frozen_classifier_v1/` | `outputs/phase3_five_stage_pipeline/stages/01_frozen_classifier/` |
| `outputs/phase3_decision_token_v2/` | `outputs/phase3_five_stage_pipeline/stages/02_decision_token/` |
| `outputs/phase3_shortcut_audit_v1/` | `outputs/phase3_five_stage_pipeline/stages/03_shortcut_audit/` |
| `outputs/phase3_same_origin_router_v1/` | `outputs/phase3_five_stage_pipeline/stages/04_same_origin/` |
| `outputs/phase3_grpo_router_v1/` | `outputs/phase3_five_stage_pipeline/stages/05_reward_grpo/` |

The `config.yaml` and report files stored inside completed stages remain
immutable run snapshots, so their recorded remote `/root/agi/...` paths are
historical provenance rather than active local configuration. Reproduction
uses the YAML files under `phase3/configs/`, whose output paths follow the new
layout.

## Ordering

1. Train/evaluate the frozen classifier using already saved activations.
2. Train decision-token-only QLoRA from Decision-Only V1 and evaluate on the
   protected 200-row benchmark.
3. Run the automated audit and complete/review its flagged pair queue before
   accepting new preference data.
4. Generate same-model-origin candidates, fresh-verify them, construct the
   hard-negative dataset, train from the best eligible prior checkpoint, and
   rerun both behavioral evaluation and representation extraction.
5. Run only a small verifier-reward GRPO pilot. The reward is correctness of
   KEEP/REVISE against fresh verifier state, with an explicit penalty if KEEP
   recall falls below the gate.

If a stage fails its gate, its reports and adapter metadata are retained as a
negative result, but omitted local weights are not presented as a usable
checkpoint. Stage 04/05 therefore fall back to Decision-Only V1 unless Stage 02
is promoted.
