# P16-G0: inference-only residual-discovery handoff v1 (2026-11-07)

Status: **CPU candidate bank prepared; GPU generation remains conditional on
the remote source preflight lock**.

This dated amendment resolves a sequencing loop in the recovery-first design.
The final `64/80/20/40` release pool cannot be required before A0 because the
64 residual `substantive_content` failures can only be identified after raw
model output has passed through the frozen recovery wrapper. Conversely, A0
must not authorize training merely because it discovers failures.

The gates are therefore separate:

1. **G0 source gate:** CPU-built structural candidate bank plus a remote,
   source-only tokenizer and sandbox/reference preflight. Passing this gate
   authorizes only A0 inference on the frozen 400-item lock.
2. **R0 admission gate:** saved A0 output is recovered and classified, then
   candidate residuals receive independent oracle, contract, history,
   near-duplicate, family and sandbox review on CPU.
3. **U release gate:** only a real `p16_recovery_first_cpu_release_v1` bundle
   containing 64 admitted residual corrections and the fixed 80/20/40 held-out
   roles can precede a separately frozen micro-update execution protocol.

G0 never authorizes LoRA, optimizer state, a protected opening, or an update
effect claim.

## Frozen source bank

The builder `../scripts/prepare_a0_discovery.py` read only pinned TACO train
revision `d593ed0a2becbbc952230bb89be09189bf1056dc`; TACO test remained unopened.
All nine train Parquet shards are again present locally, and their byte sizes
and SHA-256 values match `../data/taco_inventory_v2/summary.json`.

The source funnel excludes non-call-based rows, pictures, exact Phase 1--15
history hits, later members of exact-text duplicate groups, invalid Python
callable names, fewer than five published tests, malformed call records, and
fewer than two distinct parseable reference solutions defining the requested
callable. Token 5-gram Jaccard connected components at threshold 0.80 define
near-duplicate families before the frozen hash order is assigned.

The reproducible result is:

- 1,323 structural candidates in 1,323 detected near-duplicate components;
- 1,322 CodeWars rows and one LeetCode row;
- private bank canonical SHA-256
  `1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352`;
- private bank file SHA-256
  `db3987682bc6b391deb0302bd48342c125fabd3165e52d6fd1ae437ca9ea5212`;
  and
- public summary `../results/a0_candidate_bank_summary_v1.json`, file SHA-256
  `f2e0f266716aa44e83b5c50ace4d5c0b00a8c7d86088cee26d62e8a72042da86`.

Raw third-party prompts, published tests and reference solutions remain in the
gitignored `../data/a0_discovery_v1/candidate_bank.json`. The public summary
contains IDs and hashes, not prompt text or solutions.

This bank is deliberately labelled **structural only**. Published tests and
dataset reference solutions are source-side discovery evidence, not an
independent oracle. Its strong CodeWars concentration also means G0 cannot
support a multi-source or population-generalization claim.

## Remote preflight and lock

Run `../scripts/colab_a0_discovery.py preflight` before loading model weights.
The preflight:

1. verifies the private bank against the public summary and fixed bank hash;
2. verifies a networkless bubblewrap process running as UID 65534;
3. loads the tokenizer from pinned learner revision
   `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`;
4. rejects prompts over 2,048 chat-template tokens;
5. executes dataset reference code only inside the sandbox;
6. requires two distinct reference-solution hashes to reproduce every
   published test for an item; and
7. selects the first 400 passing, distinct near-duplicate families in the
   already frozen bank order.

If fewer than 400 families pass, the preflight fails before model generation.
It may not lower the target or substitute tasks after viewing model output.
On success it writes `a0_lock.json` and an append-only preflight audit. The
lock explicitly says `a0_inference_authorized: true`,
`gpu_training_authorized: false`, and
`protected_data_opening_authorized: false`.

## Frozen A0 inference

The learner, decoding and recovery settings are:

- model: `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28`;
- NF4 double quantization with FP16 compute;
- `transformers==4.46.3`, `accelerate==1.0.1`,
  `bitsandbytes==0.50.2`;
- greedy decoding, 2,048 input-token cap and 768 new-token cap;
- per-item recorded seed from the first eight hexadecimal characters of
  SHA-256(item ID); and
- one reference-free recovery rule: strip exactly one complete outer Markdown
  code fence labelled empty, `python`, or `py`; otherwise preserve the bytes.

The lock is divided into four immutable 100-item shards. Each command refuses
to overwrite an existing audit. It retains raw and recovered output, hashes,
generation metadata, sandbox status, full-test label, and preliminary class.
No test input or expected output is sent to the model.

Preliminary classes are screening labels only:

- recovery-fixed fenced code is `presentation_only`;
- recovered code that runs but fails a published test is a
  `substantive_content` **candidate**;
- syntax, interface, runtime and timeout outcomes are
  `execution_or_scaffold`; and
- a fully correct unchanged output is not a failure.

The word “candidate” matters: dataset-test failure alone cannot enter the
64-correction prefix.

## Handoff commands

Copy these files into one remote directory:

- private `candidate_bank.json`;
- public `a0_candidate_bank_summary_v1.json`;
- `prepare_a0_discovery.py`; and
- `colab_a0_discovery.py`.

With that directory as `/content/phase16`:

```bash
python /content/phase16/scripts/colab_a0_discovery.py preflight
python /content/phase16/scripts/colab_a0_discovery.py generate --shard 0
python /content/phase16/scripts/colab_a0_discovery.py generate --shard 1
python /content/phase16/scripts/colab_a0_discovery.py generate --shard 2
python /content/phase16/scripts/colab_a0_discovery.py generate --shard 3
```

Download the lock, preflight audit, four shard audits and four summaries before
releasing the remote instance. A partial or failed shard is infrastructure
evidence only and must not be spliced with a changed lock or package version.

## CPU work after A0

CPU rescore must reproduce every saved completion in the same sandbox and
audit all candidate residuals before selection. Only independently verified
residuals may be admitted. Correction ordering and held-out role assignment
remain hidden from GPU outcomes until source/family separation and the full
CPU contract pass. If fewer than 64 admitted residuals remain, report the
stable-onset experiment as infeasible; do not relabel scaffold errors or relax
the oracle gate.
