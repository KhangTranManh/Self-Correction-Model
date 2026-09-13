# Phase 3 Two-Stage Selective Repair

## Conclusion

`selective_repair_promising`

No weights were merged or trained. Frozen Self_Correction_v1 initial answers
are routed by the unchanged Decision-Only adapter; the unchanged V1 checkpoint
performs neutral repair only for REVISE decisions.

## Evaluation set

- 200 rows: 100 initially correct and 100 initially wrong by construction.
- 100 math and 100 code; GSM8K 60, MBPP 40, APPS 40, SVAMP 40, HumanEval 20.
- No verifier result, expected decision, or ground truth was included in inference prompts.
- Because the set is outcome-balanced, 50% initial accuracy is fixed by design and is not
  an estimate of natural deployment prevalence.

## Main metrics

| Metric | Selective repair | Always repair |
|---|---:|---:|
| Initial V1 accuracy | 50.0% | 50.0% |
| Final accuracy | 55.5% | 49.5% |
| Accuracy delta | 5.5% | -0.5% |
| Correct preservation | 99.0% | 74.0% |
| Wrong-to-correct recovery | 12.0% | 25.0% |
| Correct-to-wrong degradation | 1.0% | 26.0% |
| Repair-call rate | 30.5% | 100.0% |

## Router

- Exact contract: 100.0%.
- Decision accuracy / balanced accuracy: 65.5% / 65.5%.
- KEEP recall: 85.0%; REVISE recall: 46.0%.
- False revisions: 15; missed errors: 54.
- Repair success over all REVISE calls: 42.6%; on initially-wrong REVISE calls: 26.1%.

## Transition table

| Initial state | Decision | Repair result | Count |
|---|---|---|---:|
| Correct | KEEP | n/a | 85 |
| Correct | REVISE | still correct | 14 |
| Correct | REVISE | became wrong | 1 |
| Wrong | KEEP | n/a | 54 |
| Wrong | REVISE | repaired correct | 12 |
| Wrong | REVISE | still wrong | 34 |
| Any | INVALID | not called | 0 |

## Domain results

| Domain | Initial | Selective final | Always final | Decision | Repair call |
|---|---:|---:|---:|---:|---:|
| code | 50.0% | 51.0% | 40.0% | 59.0% | 15.0% |
| math | 50.0% | 60.0% | 59.0% | 72.0% | 46.0% |

## Dataset results

| Dataset | Initial | Selective final | Always final | Decision | Repair call |
|---|---:|---:|---:|---:|---:|
| APPS | 50.0% | 52.5% | 40.0% | 57.5% | 12.5% |
| GSM8K | 50.0% | 58.3% | 58.3% | 68.3% | 35.0% |
| HUMANEVAL | 50.0% | 50.0% | 50.0% | 60.0% | 20.0% |
| MBPP | 50.0% | 50.0% | 35.0% | 60.0% | 15.0% |
| SVAMP | 50.0% | 62.5% | 60.0% | 77.5% | 62.5% |

## Token cost

Operational token counts include prompt plus completion tokens and exclude optional audit calls.

| System | Average tokens/request |
|---|---:|
| V1 only | 327.1 |
| Always repair | 963.7 |
| Selective repair | 892.2 |
| Selective saving vs always | 71.4 (7.4%) |

Generated-token reduction vs always repair: 42.2%. The much smaller total-token reduction occurs because the full 7B router must reread the
problem and initial answer for every request.

### Measured service latency

- Average decision request: 0.889s.
- Average repair request: 7.374s.
- Modeled additional latency, always/selective: 7.374s / 3.416s.

Initial generations were reused from the frozen project logs, so current-run initial-solve
latency is unavailable. Decision and repair request latencies are retained in raw caches.

## Controls and audit

- Unseen neutral-template decision accuracy: 53.5%.
- Substantive separate decision rationales: 200/200.
- Rationale generations are not used for routing, scoring, or token-cost comparison.
- Always and selective systems use the same deterministic repair generation per source;
  selective accounting includes it only when the router emitted REVISE.

## Interpretation

Selective repair produced three net additional correct answers over V1-only and four
more than always-repair while making 66% fewer repair calls. It strongly reduced damage
to initially-correct answers, but missed 51/100 errors, lost accuracy on code overall,
and generalized weakly to unseen review wording. The accuracy/compute tradeoff improved,
but the margin and total-token saving are too small for a strong claim.
Paired exact McNemar tests are not significant: V1-only vs selective p=0.003; selective vs always p=0.073.
