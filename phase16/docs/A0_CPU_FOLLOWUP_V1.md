# A0 CPU follow-up after sandbox closure

Status: **in progress; no correction admitted**.

The 142 records in the original oracle-ready queue were executed in the CPU
sandbox. Subsequent contract review found two incorrect frozen definitions,
so the 141 initially reported sandbox mismatches are observations, not 141
certified residuals. The original review, sandbox artifacts and summary are
immutable historical evidence. Use `../results/a0_cpu_followup_worklist_v1.json`
for current gate states.

## Contract audit and amendments

The input-shape triage compared only broad types and argument counts, without
reading published expected values to construct an oracle. It flagged two of
142 positions:

- Position 343 (`bucketize`): one independent test uses eight positional
  arguments, a count not present in the discovery samples. The prompt itself
  explicitly demonstrates that variadic call and gives an eight-argument
  example. This is not an input-shape defect; the full contract audit remains
  pending.
- Position 392 (`pay_cheese`): the frozen oracle and tests expect one scalar
  and return a number, while the prompt specifies an array of five daily
  values and a currency-formatted string. The sandbox runtime error therefore
  does not establish a model error on valid input. The currency symbol is `£`
  (U+00A3), verified in the frozen bank and pinned TACO shard; the prior report
  of a replacement character was a terminal rendering error. The exact number
  of fractional digits in the returned string is not stated. This record is
  held pending resolution of serialization. The source check and pinned-shard
  hash are recorded in `../results/a0_position392_source_audit_v1.json`.

Shape agreement does not prove that inputs obey the contract or that expected
values are correct. A separate prompt review found that the frozen position
398 (`big_primefac_div`) oracle reversed the specified result order and used
the wrong result for fractional values. The versioned amendment
`../results/a0_cpu_sandbox_v2/position_398_amendment_v2.json` binds the same
frozen item to corrected, mutually checked contract implementations. Its
networkless sandbox rerun completed with two of six independent cases matching;
the original sandbox observation is superseded for that position. Independent
review of the amendment remains pending.

The worklist records 140 full contract audits pending, one amended oracle
awaiting independent review, and one held contract. All 142 still have
semantic-history, cross-history/near-duplicate, corrected-implementation and
human-approval gates pending. An exact or near-duplicate check does not by
itself produce an approved correction.

## Existing history evidence

`../results/a0_existing_history_evidence_v1.json` binds each candidate to the
previous exact-overlap scan and its hashed metadata. The prior scan covered
393 local JSON/JSONL files with no parse errors, and found no exact hit among
the 142 worklist records. This step did not reopen historical task text. The
old scan excludes other formats and remote registries and cannot establish
semantic non-overlap; all semantic and manual cross-history gates remain open.

No pool has been certified. GPU training and protected data opening remain
unauthorized. A human must perform the explicit approval gate; the existence
of a user request to continue work is not an approval of individual corrections
that have not yet been produced and reviewed.
