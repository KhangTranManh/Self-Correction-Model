# Router Recovery V2 data

## Current state

Data preparation is closed and training was never started. The local verified
pool contains 108 same-problem, same-model-origin pairs (216 behavioral rows):
84 math pairs and 24 code pairs. Every correct member passed and every wrong
member failed a fresh verifier. There are no synthetic wrong answers, duplicate
sources, or source overlaps with probe-256, frozen-200, Decision-Only data, or
Router Calibration V1.

The canonical partial artifacts are in `current/`. They intentionally retain
the `.partial.jsonl` suffix because they never reached the 200-pair training
gate. Phase 3 keeps them for audit and possible import into a future phase, not
for training inside Phase 3.

## Files

| File | Purpose |
|---|---|
| `existing_verified_pairs.jsonl` | canonical paired correct/wrong records available without GPU |
| `development_all.partial.jsonl` | two neutral-review behavioral rows per existing pair |
| `existing_pair_audit.jsonl` | accepted/rejected source audit |
| `preparation_summary.json` | counts, overlap result, generation need |
| `generation/generation_manifest.jsonl` | completed historical generation queue |
| `generation/batch_summary.json` | historical small-batch definitions and hashes |
| `generation/batches/*.jsonl` | historical resumable GPU batches |
| `current/verified_pairs.partial.jsonl` | canonical 108-pair verified pool |
| `current/development_all.partial.jsonl` | balanced KEEP/REVISE rows for the partial pool |
| `current/current_summary.json` | latest machine-readable batch result |
| `current/dataset_status_report.json` | consolidated validation and closure status |
| `current/dataset_status_report.md` | human-readable closure status |
| `full_code_scan/scan_summary.json` | compact summary of the unexecuted APPS/MBPP scan |
| `full_code_scan/exclusions.jsonl` | excluded-source audit |
| `full_code_scan/README.md` | compaction and reproduction note |

## Historical GPU result

The sequential GPU pass generated 2,804 valid-prompt candidates across
corrected V1/base, MBPP, APPS, low-temperature, and adaptive high-temperature
batches. It admitted 43 new pairs on top of the 65-pair CPU pool. An earlier
480-candidate run used the wrong prompt construction and is excluded.

The final yield was 108/200 minimum pairs. Additional sampling from the same
sources/configurations was stopped after marginal yield fell to 2/184.

The later full-code scan identified a potential pool of 4,087 APPS and 339 MBPP
sources, but Phase 3 closed before its generation queue ran. These were only
structurally eligible candidates, never accepted training data. The large
unexecuted manifests and batch files were removed during closure; the compact
summary, exclusions, and reproducible builder remain. Any future generation
must occur in a new phase with new holdouts and a new experiment registry.

For dual-outcome tasks, at least one candidate must freshly pass and at least
one plausible candidate must freshly fail before a pair can be admitted.

## Unreached target

- Minimum 200 pairs; target 300.
- Exactly one KEEP and one REVISE row per source.
- Target domains: 60% code, 40% math.
- Source-grouped splits: 60% representation train, 20% threshold calibration,
  and 20% internal test.
- Same neutral-template distribution for both labels.
- Final split files are created only after validation passes.

Reproduce the original 65-pair CPU seed artifacts only in a new experiment:

```bash
python phase3/scripts/data/prepare_router_recovery_v2.py
```
