# P16-F0-A initial TACO train audit

Recorded 2026-11-03. This is a source/metadata inventory, **not** an A0
model screen, a verified C-upper-inclusive pool, or a treatment result.

## Acquisition

- Source: `BAAI/TACO`, revision
  `d593ed0a2becbbc952230bb89be09189bf1056dc`.
- Downloaded: nine `ALL/train-*.parquet` shards and the dataset card to
  `../data/taco_raw/`, 2,174,060,481 bytes of train Parquet. Individual
  shard SHA-256 hashes are in `../data/taco_inventory_v2/summary.json`.
- TACO `test` shard was **not downloaded or opened**. The raw source folder
  is ignored by Git. The dataset card reports 25,443 train and 1,000 test
  rows and notes that some TACO material comes from APPS/CodeContests.
  [Source card](https://huggingface.co/datasets/BAAI/TACO/blob/main/README.md).

## Reproducible metadata pass

`../scripts/taco_inventory.py` streamed only train shards and wrote hashes
and metadata, not raw questions or solutions, to
`../data/taco_inventory_v2/train_metadata.jsonl` (SHA-256
`3bc4902460890c3d5cbb422a7ad20647c33e90c22cee20a3ef6747e09c731b48`).
The summary's SHA-256 is
`4e82db38ffeab74323318daa56611894f88e5e821988c926b60114e6b7f886a7`.

| Field | Observed train rows |
|---|---:|
| Total | 25,443 |
| Function-call tests | 2,893 |
| Standard-input/output tests | 21,809 |
| Empty or mismatched input/output lists | 741 |
| Missing/invalid canonical URL | 5,224 |
| Picture flag | 630 |

The current Phase 16 code verifier supports function-call tasks only;
standard-input/output tasks require a separately tested sandboxed verifier.
Test-list shape is not proof that a reference answer is correct. Dataset
solutions were not run. A first inventory attempt stopped on an unusually
large JSON integer; the final parser reads JSON integers as strings because
it only counts test shapes. The partial `taco_inventory_v1/` directory is
incomplete and **must not be used**; all figures here use `v2/`.

## Exact overlap and lexical prefilter

`../scripts/taco_exact_overlap.py` scanned 393 local JSON/JSONL artifacts
from Phases 1–15 plus the excluded Phase 16 engineering pack. It reported
zero JSON parse errors and flagged 490 TACO rows matching historical
normalized question text and/or canonical URL. Within TACO it found 544
exact-text duplicate groups containing 1,172 rows. This is a conservative
flagging pass, not completed source-family or near-duplicate adjudication.
Audit SHA-256:
`6d2eeb9f79d9e6b5d9803b01d1f91ce2c55108c38642188bed2241a56421a7bc`.

After excluding historical exact hits, later members of exact-text groups,
and picture-flagged rows, 2,518 function-call rows remain as *metadata-level
candidates*. `../scripts/taco_boundary_prefilter.py` searched their source
statements for broad boundary wording and flagged 375: 251 CodeWars, 100
LeetCode, and 24 GeeksForGeeks. The terms include `between`, `at most`,
`inclusive` and interval notation; they intentionally over-include unrelated
meanings and can miss other phrasings. Prefilter SHA-256:
`46c70a52f902ae64e9cfac0c6bbba68fa3b947223b5ae99e4c27f089d21cb3c3`.

**Do not infer 375 eligible tasks.** Even if all 375 passed review, they
would be fewer than the proposed 400 teaching tasks alone, but the lexical
filter is not exhaustive. A source-only human/independent-oracle review and
near-duplicate audit must decide whether the C branch has enough capacity.
No A0 model output has been generated on TACO and no GPU was used here.
