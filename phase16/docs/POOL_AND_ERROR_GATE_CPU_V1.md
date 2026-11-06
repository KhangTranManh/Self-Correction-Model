# Phase 16 S0: pool and error-verifier CPU gate (2026-11-05)

**Decision: NOT READY for a scientific GPU screen.** This is a source and
scorer preflight, not a replacement for `SIGNAL_ONLY_CPU_V1.md` and not a
model result. It does not alter Phase 15 or the original 40-episode protocol.

## Frozen unit of analysis

One source task has a stable source/revision/ID, public contract, language or
answer format, family key, deterministic independent oracle, and a fixed
test-case hash. Numeric variants or paraphrases of one template are one
family, not independent tasks. Exact and near-duplicate source/history checks
must precede assignment. The S0 target remains 300 unique tasks **per
feasible domain**, assigned before A0: 160 screen, 40 applicable probes, 20
near-negatives, 20 unrelated controls, 60 sealed reserves. At least 20 of
the 160 must show a natural named error and 20 must be initially correct.
Until the complete manifest is signed off, none of these role counts is met.

## Code: C-upper-inclusive

The `C-upper-inclusive` scorer is `../scripts/score_boundary.py`. It requires
the candidate to pass ordinary/lower/outside cases and miss two independent
`R`-addition pairs while the oracle changes. This is a **behavioral
signature**, not an inference that the program literally uses `< R`.
Full correctness uses all fixed cases. Syntax, runtime and timeout failures
are distinct. The scorer consumes trusted expected/actual transcripts; it
does not generate tests or safely run code.

The source-only audit `../scripts/audit_c_contract_pool.py` examined the 375
TACO train lexical candidates from the pinned inventory, after the earlier
exact-history and within-TACO exclusions. Counts are 54 with explicit
lower/upper-bound terms, 162 with inclusive/closed language, 225 with a
count/sum/select-like term, 313 with sequence terms, and **15 matching all
four**. Results and per-ID flags are in
`../results/c_contract_text_audit.json`. These regexes have false positives
and false negatives; 15 is not a proven upper bound. Inspection of the 15
shows unrelated uses of these words (e.g. calendar years, skyline) as well
as potentially relevant range problems. **Zero tasks are certified** by
this audit. In particular, 15 text hits cannot be promoted to the roughly
200 rule-applicable screen/probe tasks S0 needs.

CodeContests' first three train shards yielded 302 metadata candidates only
(`MULTISOURCE_F0_RECHECK.md`). They are standard-I/O contest tasks, so the
function-call C scorer cannot be attached to them without new task-specific
contract review, reference tests and a sandboxed stdio verifier. Incorrect
published submissions are not natural A0 errors. They therefore do not fill
this named-rule pool today.

## Math: M-negate-difference

`../scripts/score_negate_difference.py` accepts a **structured** expansion
for `-(a-b)` and a final integer. An AST allowlist admits only `a`, `b`,
bounded integers, `+`, `-`, `*` and parentheses; it never executes submitted
text. SymPy checks symbolic equivalence to `-a+b` and the distinct wrong
formula `-a-b`; `b=0` is rejected so the two formulas differ. An incorrect
final value alone is `other_failure`; missing/ambiguous/unsupported expansion
is `unscorable`. Five synthetic unit tests pass. This scorer does **not**
extract intermediate expressions from natural model prose, nor certify a
source pool. Human-blinded parser review on a frozen sample is still needed.

The pinned DeepMind mathematics_dataset emits Q/A, not necessarily an
explicit `-(a-b)` expansion, so its generated inventory cannot be counted
as 300 eligible tasks. A controlled generator could produce valid exercises,
but changing constants alone would not create independent families. A new
design would need multiple predeclared expression scaffolds, family-disjoint
splits, oracle fixtures, historical-overlap audit, and a natural-error A0
screen. Existing MATH/GSM8K reuse must be tracked rather than assumed clean.

## CPU-to-GPU release conditions

1. Publish source manifests for **one chosen domain** with all 300 role IDs,
   revision/content hashes, family keys, license/provenance and exclusion
   report. Never fill missing roles after observing A0 output.
2. Validate each task's public contract and oracle. For code, run candidate
   code only inside an isolated, resource-limited sandbox; independently
   verify two reference implementations on boundary/metamorphic and ordinary
   cases. For math, freeze prompt and expansion extractor and audit ambiguous
   outputs blind to treatment arm.
3. Run scorer unit/adversarial fixtures plus an oracle-dispute audit. Reject
   tasks with ambiguous specs, inconsistent references or scorer disagreement.
4. Lock model revision, decoding, seed, prompts and hashes; only then spend
   GPU on the staged A0 screen. Stop if fewer than 20 natural named errors or
   20 correct checks occur in the frozen 160-task prefix. Do not reinterpret
   generic wrong answers as the named error.

The existing AWS budget plan is a **cost ceiling**, not authorization or
evidence that gates 1-4 have passed. No GPU run was launched here. If source
capacity keeps failing, explicitly amend the error type or shrink the
claim/episode target *before* any model screening; do not silently relabel
the current hypothesis.
