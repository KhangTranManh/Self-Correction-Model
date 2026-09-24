# Phase 5 final report

Phase 5 completed one protected opening after prompts, checkpoint lineage,
probe layer/C choices, and controls were locked. Reviews used vLLM 0.7.0 in
FP16; hidden-state probes used frozen Transformers FP16 forward passes. No
weights were trained and no checkpoint is promoted.

## Protected behavioral results

| Checkpoint | Condition | Contract | Final accuracy | Wrong→correct | Correct→wrong |
|---|---:|---:|---:|---:|---:|
| Original solver | neutral | 98.12% | 50.00% | 0.00% | 0.00% |
| Original solver | status | 71.25% | 50.00% | 0.00% | 0.00% |
| Warm-start V2 | neutral | 98.75% | 48.75% | 0.00% | 2.50% |
| Warm-start V2 | status | 98.12% | 53.12% | 6.25% | 0.00% |
| Correction SFT V3 | neutral | 98.75% | 32.50% | 5.00% | 40.00% |
| Correction SFT V3 | status | 99.38% | 54.37% | 8.75% | 0.00% |

The split starts at 50% accuracy by construction. V2 passes the preregistered
assisted-repair gate: status minus neutral wrong→correct is +6.25 percentage
points with paired 95% bootstrap CI `[1.25, 12.50]`, status contract validity
is above 95%, and harm does not increase. This is not autonomous detection,
because status explicitly reveals whether the initial answer is wrong.

V3's neutral behavior is unsafe: its four fixes are outweighed by 32 harmed
initially correct answers. The result reproduces on train, development, and
protected splits and cannot be dismissed as a small smoke artifact.

## Protected pre-hint probe

| Checkpoint | Layer / C | Balanced accuracy | ROC-AUC | Correct recall | Wrong recall |
|---|---:|---:|---:|---:|---:|
| Original solver | 14 / 0.01 | 71.25% | 78.30% | 78.75% | 63.75% |
| Warm-start V2 | 14 / 0.01 | 71.25% | 78.09% | 78.75% | 63.75% |
| Correction SFT V3 | 14 / 1.0 | 74.38% | 76.98% | 73.75% | 75.00% |

The surface-only development control reached 67.5% balanced accuracy. Across
100 shuffled-label fits per checkpoint, mean balanced accuracy stayed near
chance and none matched the selected hidden probes (`p=0.0099`). Correctness is
therefore genuinely decodable before feedback, although the generative policy
does not use that signal safely under neutral review.

## Interpretation boundary

The probe is an external diagnostic harness, not evidence that the standalone
checkpoint can self-correct. It reads a linearly accessible direction from
frozen hidden states and predicts correctness outside the model's generative
decision path. This establishes **signal availability**, but not **signal use**.

The end-to-end autonomous requirement is stricter: without a correctness hint,
the model must detect wrong answers, preserve correct answers, and turn a
REVISE decision into a verified repair. No tested checkpoint satisfies that
requirement. V2 demonstrates limited assisted repair only when status feedback
discloses correctness. V3 exposes the opposite failure mode: increased error
sensitivity coupled with severe over-revision of correct answers.

Consequently, a future composition of model + probe + repair policy should be
reported as a harness-controlled correction system unless autonomous model-only
behavior is separately demonstrated on a new protected set.

## Disposition

- No autonomous-review checkpoint is promoted.
- V2's status-conditioned gain is retained as evidence for assisted repair only.
- V3 shows a stronger error-sensitive boundary but unacceptable sycophantic
  over-revision without a correctness hint.
- The protected set is closed after one opening and cannot be reused for model,
  prompt, layer, regularization, or threshold selection.

Full raw prompts, raw outputs, parsed decisions, verifier results, activations,
probe models, bootstrap comparisons, and hashes are in
`outputs/phase5_gpu_vllm/phase5_final_report.json` and neighboring artifacts.
