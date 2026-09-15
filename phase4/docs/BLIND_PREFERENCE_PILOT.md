# Blind preference pilot V2

Final status: collection completed; the frozen coverage gate failed. Available
train KEEP/REVISE pairs were 14/44 and dev 2/4, leaving balanced 28/four rows.
DPO V2 was not trained. Phase 4 is closed; the protocol below is preserved.

Hypothesis: preferences between actual neutral reviews transfer better to
selective correction than constructed KEEP/REVISE alternatives.

Collect 16 seeded reviews per source from selected V2, temperature 1.2 and
1,200-token cap, on the fixed 904 training and 100 development correction
sources. Every prompt uses the deployment system and neutral review text.
Verifier results, references, labels, and guided correction targets are absent
from generation prompts. Verify initials freshly before collection and verify
all strict revised answers after generation. Store every output, including
invalid and truncated outputs, in a resumable audit.

Use one pair per source. Initially wrong sources need an actual verified fix;
prefer an actual failed revision as the rejected response, falling back to an
actual KEEP only if necessary. Initially correct sources need an actual KEEP
and an actual harmful revision. Both sides are generated under identical
prompts. Balance by action separately within the unchanged source splits.
Training requires at least 20 pairs per action in train and five per action in
development. Insufficient coverage is a negative collection result; do not
manufacture responses or increase sampling after inspecting development.

If the gate passes, train one QLoRA epoch from selected V2, capped at 50 steps,
learning rate 1e-6 and beta 0.1, with checkpoint backup during the run. Freeze
the configuration and exact dataset hashes locally before upload.

Evaluate selected V2 and the new adapter using reused Cycle 0 initial answers
at three seeds (20260914, 20260915, 20260916). Report each seed, aggregate
accuracy, wrong-to-correct and correct-to-wrong counts, and contract validity.
This is a development diagnostic, not a confirmation result. A gain must
improve mean accuracy without increasing mean harmful revisions. Prepare a
fresh balanced math/code diagnostic set before any promotion claim. Existing
confirmation candidates remain unread and Phase 3 remains closed.

Fresh diagnostic preparation uses 200 GSM8K training problems at deterministic
offset 2,800 after the historical exclusion, plus 144 non-overlapping HumanEval
problems. The 144 trusted code reference solutions pass the local verifier.
Selected V2 generates one natural initial answer per candidate, without
feedback. Fix 20 sources in each domain-by-initial-correctness bucket by seeded
hash rank, before candidate training. If any bucket lacks 20 natural examples,
record insufficient coverage rather than manufacturing errors. This 80-row
set is a fresh development diagnostic, separate from scientific confirmation.

The fixed diagnostic has 80 sources: 20 natural correct and 20 natural wrong
answers in each domain. Candidate bucket counts were math 150 correct/50 wrong
and code 37 correct/107 wrong. All 344 cached outcomes agree with fresh local
verification under CPython 3.10/SymPy 1.14.0. An initial local CPython 3.13 check
disagreed on a trailing-currency answer (`8.0£`); the historical verifier code
was preserved, and the local verification runtime was aligned with the GPU.

Local verification uses `.venv-phase4-verify/Scripts/python.exe`. To recreate:

```powershell
python -m uv venv --python 3.10 .venv-phase4-verify
python -m uv pip install --python .venv-phase4-verify/Scripts/python.exe -r phase4/requirements-verify.txt
```
