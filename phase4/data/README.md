# Phase 4 data

Phase 4 closed on 2026-09-15. Data and raw evidence are retained unchanged.

| Directory | Archived contents |
|---|---|
| `pilot_v1/` | Original sources and untouched confirmation candidates |
| `warmstart_v1/` | Small guided correction / KEEP split |
| `expansion_v1/` | 2,000 math sources and natural initial answers |
| `verified_corrections_v1/` | Guided attempts and balanced 904/100 SFT split |
| `preferences_v1/` | Constructed 904/100 DPO preferences |
| `blind_preferences_v2/` | 16,064 blind reviews and insufficient-coverage 28/4 balanced pairs |
| `fresh_diagnostic_v1/` | 344 candidates, natural initials and verifier audit |
| `fresh_diagnostic_v1/fixed/` | Frozen 80-source math/code × initial correct/wrong diagnostic |

The fixed diagnostic is development evidence, not confirmation. References and
verifier labels are never neutral-review prompt inputs. See `../docs/RESULTS.md`
and `../configs/experiments.yaml` for final results and evidence pointers.

`rollouts/` is generated and gitignored. Every cycle must retain its input
manifest, raw model responses, verifier outcomes, configuration, model/adaptor
identity, and SHA-256 manifest. The same cycle directory is copied to the local
workstation before the next expensive stage begins.

Phase 3 protected evaluation and probe datasets are prohibited inputs.
