# Phase 15 dev diagnostic, amendment P15-A4

Recorded 2026-11-03 before applying the P15-A4 transformation. This is a
deterministic CPU-only diagnostic on the existing P15-A3 A0 completions. It
does not generate new model output, update weights, open an evaluation split,
or reuse any probe answer as an input.

## Reason

P15-A3 found that the largest intervention signal was improved S-format
compliance. Hard token blocking is not the first follow-up because masking
renormalizes the token distribution and may trade one behavioral bias for
another, as Phase 3 activation steering did for KEEP versus REVISE. It can also
suppress useful reasoning, create tokenizer-dependent dead ends, and make an
incorrect answer look cleaner without making it correct.

P15-A4 measures the ceiling of a narrower system intervention that cannot
deflect generation: deterministic suffix normalization after generation.

## Frozen transformation

For each P15-A3 `a0_base` completion independently:

1. Reapply the frozen local `solve_errors` and parsers.
2. If and only if `S-format` is present, the lenient parse is nonnumeric, and
   the rescued parse is numeric, form a candidate by preserving the complete
   original output and appending exactly `Đáp số: <rescued value>` on a new
   final line.
3. Accept the candidate only if its lenient parse is numeric and equal to the
   pre-transformation rescued value. Otherwise leave the output unchanged and
   record a rejected normalization.
4. Never delete, replace, or mask a generated token. Do not consult the
   reference answer, teaching answer, another arm, or another item while
   transforming an output.

## Outcomes and risk audit

Report probe and control answer correctness, S-format cleanliness, and joint
correct-and-clean counts before and after normalization. Report paired
fixes/harms, number accepted/rejected, how many normalized outputs remain
incorrect, and any unexpected change to a completion that was already
format-clean. A transformation is mechanically safe only if there are zero
correctness harms, zero changes to clean outputs, and zero parse-preservation
failures.

This arm is a system postprocessor, not evidence that the model learned from a
correction. It is not pooled with P15-A2 or treated as a replacement for the
P15-A3 matched-seed weight-update diagnostic.

## Result and decision

The frozen rule was applied to all 120 P15-A3 A0 completions. It accepted 49
S-format normalizations and left 71 already-clean outputs byte-for-byte
unchanged. There were no rejected candidates, parse-preservation failures,
changes to already-clean outputs, or correctness harms.

On probes, correctness/joint rose from 31/80 to 45/80 (14 fixes, zero harms)
and S-format cleanliness rose from 38/80 to 80/80. On controls,
correctness/joint rose from 30/40 to 36/40 (six fixes, zero harms) and
cleanliness rose from 33/40 to 40/40. Of the 49 accepted normalizations, 29
were still incorrect after becoming format-clean. Thus the intervention
recovers answers that the model already placed in its text, while also showing
that clean formatting does not imply correct reasoning.

Do not launch a hard-token-mask follow-up for S-format: the non-generative
normalizer reached 100% format cleanliness with zero observed harm, so token
blocking would add distribution-shift and tokenizer risks without an unmet
format objective. This result weakens a LoRA-specific interpretation of P15-A3:
the unchanged A0 outputs reach 45/80 joint after deterministic normalization,
versus 40/80 for A2 LoRA. It is a system-reliability result, not model learning.
