# P16-R0 A0 CPU admission inventory v1 (2026-11-08)

Status: **worklist frozen; zero corrections admitted**.

This step converts the completed P16-G0 run into a complete, fail-closed CPU
review inventory. It does not treat published TACO tests as an independent
oracle, certify a correction pool, authorize training, or open protected data.

## Integrity boundary

The builder verifies the downloaded run manifest, lock, four shard audits and
summaries, saved-output rescore, private candidate bank, and explicit
nondeterminism evidence before writing an inventory. The tracked A0 evidence
uses `eol=lf` in `.gitattributes` because the run manifest hashes exact bytes;
without that rule, a Windows checkout converts the files to CRLF and invalidates
all ten downloaded-file hashes.

The frozen inputs are:

- bank SHA-256 `1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352`;
- lock SHA-256 `f655a2f3de813bc1d420d37d105dff7d881156e37b05b34a44cd155e04ae3b4c`;
- run-manifest SHA-256 `d69c69d25f88364f842692f7bf5b5a28718fe167cd1b9cab600d3c6bfbb67d7e`;
- rescore-file SHA-256 `dcff5a6d4fb0774065074e55bced413cad687cfffcd3fb1a10f61bbb00572079`;
  and
- nondeterminism-evidence SHA-256
  `48f93c8b7a60227341fbe8ab8d357e6710cfa60ea046c8fe0695f1b291cb34dd`.

## Accounting

`../results/a0_admission_inventory_v1.json` retains all 244 dataset-side
failures in original A0 position order:

| Initial disposition | Count | Reason |
|---|---:|---|
| Pending independent-oracle review | 168 | Wrong-answer / `substantive_content` candidate with a deterministic saved-output rerun |
| Rejected | 75 | Frozen `execution_or_scaffold` class; not correction data |
| Rejected | 1 | `taco_train_05701` failed deterministic rerun |

Every record stores only identities, hashes, source metadata, published-test
counts, gate states and rejection reasons. It does not copy prompt text,
reference solutions, tests, expected values or model code from their frozen
source artifacts. Each record and the whole inventory have recomputed
canonical hashes. Inventory SHA-256:
`00743b1561a4c0e632ca2020c91e7a1c891e6d9e75af28b3ff04ad7edc6be377`.

## Independent-oracle boundary

The 168 queued records are not corrections. Each still needs a task-specific,
deterministic oracle constructed from the task contract rather than copied
published expected values or dataset reference solutions. It must then pass
contract, full-history, semantic-near-duplicate, sandbox, and human-review
gates. A model judgment is not an oracle. Rejection records remain in the
audit and cannot be relabelled to reach the 64-correction target.

Review should follow frozen A0 position order. If fewer than 64 independently
verified residuals survive, close the stable-onset experiment as infeasible.
Do not substitute execution/scaffold failures, lower a gate, run LoRA, or
open protected data.

## Reproduction

From the repository root:

```powershell
python -B phase16/scripts/build_a0_admission_inventory.py `
  --root phase16/data/a0_discovery_v1/gpu_bundle `
  --run phase16/results/a0_discovery_run_v2 `
  --manifest phase16/results/a0_discovery_run_v2_manifest.json `
  --rescore phase16/results/a0_discovery_run_v2/a0_rescore_v1.json `
  --nondeterminism phase16/results/a0_discovery_run_v2/a0_rescore_nondeterminism_v1.json `
  --output phase16/results/a0_admission_inventory_v1.json
```

The writer refuses to overwrite the existing inventory. Reproduce to a new
path and compare its canonical `inventory_sha256`.
