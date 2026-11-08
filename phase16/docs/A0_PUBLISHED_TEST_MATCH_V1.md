# P16-G0 published-test match v1 (2026-11-08)

Status: **CPU definition for the next remote preflight**. This does not
authorize A0, training, a protected opening, or reuse of the failed 1/400
lock. The candidate bank hashes stay unchanged.

## Why this exists

The 2026-11-08 remote preflight on the G0 bank scanned all 1,323 families and
accepted 1 (`taco_train_08742`). The job then failed with
`preflight selected 1 of required 400`. Sandbox execution itself completed
for 3,892 reference runs; almost all had `match_count == 0`.

A reproduced case is `taco_train_14050` (`remove_parentheses`): the published
call record stores expected output `["exampleexample"]` while both reference
solutions return the string `"exampleexample"`. Strict `got == expected`
rejects them. This wrapping is common in TACO call-based JSON (414 of 1,323
tasks have a singleton string list as `test_outputs[0]`; 908 have some other
list; 1 has a bare string). It is a serialization shape, not a second
function argument.

`TACO_SOURCE_AMENDMENT.md` still forbids coercing **standard-input** rows into
the function-only verifier. This amendment does not touch that. It only
defines how two already-parsed call-based values are compared.

## Frozen comparison

Add `published_test_match(got, expected)` and use it in
`scripts/colab_a0_discovery.py` wherever a published call-based test is
checked:

1. If `got == expected`, the test matches.
2. Else if `expected` is a `list` of length 1 and `got == expected[0]`, the
   test matches.
3. Otherwise the test fails.

The rule unwraps **one singleton expected list** only. It does not unwrap
`got`, does not recurse, does not stringify, and does not accept a length-2+
list by taking the first element.

Residual risk: a task whose true return value is a one-element list can match
a scalar of that element. Dual-reference still requires two distinct passing
codes, and dataset tests remain discovery evidence rather than an independent
oracle.

## What stays frozen

- Bank SHA-256
  `1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352`.
- Target 400 distinct families, two passing references, tokenizer cap 2,048.
- No lowering of 400 after seeing model output or this failed preflight.
- Failed remote audit `results/a0_preflight_fail_v1.jsonl` is kept. Do not
  overwrite it or splice it into a later lock.

## Host note (not a comparison change)

`sandbox_preflight()` now uses the same `/lib` and `/lib64` symlinks as
`sandbox_execute()`. Amazon Linux 2023 cannot exec `/usr/bin/id` inside
bubblewrap without them. Ubuntu merged-usr hosts keep working.

## Next remote run

Start a **new** output directory after this code is on the host. If the
instance was stopped, copy the updated `colab_a0_discovery.py` before
`preflight`. A0 remains unauthorized until the new preflight writes a lock
with exactly 400 families.
