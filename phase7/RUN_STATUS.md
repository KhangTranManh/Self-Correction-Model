# Phase 7 run record — 2026-09-25

**Completed.** The background GPU pipeline ended with
`PHASE7_PIPELINE_COMPLETE`. Development generated 80 paired outputs per
checkpoint; protected generated 320 paired outputs per checkpoint. All six
audits and outputs are backed up under `outputs/phase7_paired_v1/` on this
workstation and match their remote-generated summary hashes. The protected
analysis was independently reproduced locally with the same SHA-256,
`95f9da44b66fbc359afc1f5f24c4db51cdaa6e64a9fd94bd82a07e06dd0cb2c6`.
Read the [final report](docs/FINAL_REPORT.md) for the findings.

The initial-answer collector stopped at the first quota prefix: 334 sources,
234 verifier-correct and 100 verifier-wrong. Its audit SHA-256 is
`f396a37e76da9281fb0f1120dcf8cb979b4ecac06d3827dd241a5a738b14812b`.
The balanced split is 40 development sources (20/20) and 160 protected
sources (80/80). The paired-generation protocol lock SHA-256 is
`f468394d73726595d913252a1ce897ae2f2bf7dcdfc7719c8ad726f271e4fa46`.

The run began on a V100 32 GB with vLLM 0.7.0 and FP16 weights. That host
became unavailable during V2 development, after 67 of 80 outputs had been
saved. A checked resume bundle moved the audit to an RTX 3090 24 GB host;
its SHA-256 is
`a4cc8341c5e99a47ac3068215186a44e5aee84930557f2c761ce68a8fb8f4114`.
The RTX 3090 FP16 smoke passed at 19,430 MiB after generation. Rebuilt V2
shards and all other 20 inputs matched the original paired lock exactly.
One V2 development pair straddled the host change; every protected pair ran
entirely on the RTX 3090. The remote GPU retains code and results at
`/root/AGI_phase7/`.

The frozen Phase 5 probe artifact was unavailable, so non-oracle routed
analysis was not run. No model was trained or promoted. The
`configs/experiments.yaml` status field is a hash-bound pre-smoke snapshot;
it was left unchanged to preserve both protocol locks.
