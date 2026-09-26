# Phase 8 run status

2026-09-25, local CPU:

- Fresh 400-source protected pool frozen. Manifest SHA-256:
  `a4d2829140292cce1b86ccf3c758e7b30641ee829452698a9ca139462eae7e24`.
- Phase 7 as-run audit confirmed initial temperature 0.7 versus paired greedy;
  blind user text adds a fixed suffix to the initial prompt on all 334 rows.
  The Phase 7 protected split contains 80 wrong and 80 correct first answers.
- Exact Phase 5 probe `.joblib` artifacts remain absent locally. Existing
  Phase 5 train/development sources are intact and source-disjoint from the
  Phase 8 pool.
- Preregistration written before inspecting any Phase 8 answer, verifier
  outcome, probe score, or aggregate metric. The clarification fixing
  historical layer/C rather than reselecting on development was committed to
  the local protocol after first-pass collection had started but before any
  output content or correctness result was inspected. Treat this as a timing
  deviation from a strict before-generation freeze; do not hide it.

2026-09-25, remote GPU:

- New host: RTX 3090 24 GB, Python 3.10.12, vLLM 0.7.0, Torch 2.5.1+cu124.
- Code bundle copied to `/root/AGI_phase8/`, SHA-256
  `5067ed8b226c8718f9b99b1c8d4c995caeaeedd9cea65435f9825a39457d1028`.
  Remote candidate manifest hash matches local.
- Final locked code bundle V2 was also copied and verified remotely, SHA-256
  `6a20baae3995587c8597d809795b676b27c2e751799107a536db605aa192ade6`.
- The pinned tokenizer audit confirmed that rendered initial and blind prompts
  differ on all 334 historical initial rows; its JSON report is under
  `phase8/data/phase7_rendered_prompt_audit_v1.json` on both machines.
- First-pass `initial` collection started with append-only audit under
  `/root/AGI_phase8/outputs/phase8_first_pass_v1/initial/`.
- Two Phase 4 adapters were copied for V2/V3. No probe fit, distractor
  generation, final analysis, or non-oracle result has completed.
- Both adapter weight SHA-256 values are now verified on the GPU host. The
  remote sequential pipeline is waiting for first-pass completion. A local
  mirror process copies append-only audits and later probe/result artifacts
  into `outputs/phase8_remote_3090/`.
- The first execution lock (`4b30895a...`) exposed a preflight path typo for
  the V2 adapter (`phase4_warmstart_v2` instead of
  `phase4_exploration_warmstart_v2`). The preflight script and its hash entry
  were corrected before three-arm/probe generation or protected outcome
  inspection. Amended lock SHA-256:
  `a1152ae1b41b7b29090544c628cc30d4bb9dd8ddb7d81180db4be23d7b4dc78c`.
- All 400 first-pass completions were durably recorded, but summary export
  initially failed with `KeyError: 'row'` because newly generated rows were
  appended in a different in-memory shape from resumed audit rows. No output
  content or correctness label was inspected. The export-only wrapper was
  corrected; rerunning read the existing 400 audit entries without generating
  another answer. Updated execution lock SHA-256:
  `b4b713e5d7925f1bac5441640b5973c46f70a88bfe15387ea0e41de1a2b65187`.
- First-pass summary: 400 rows, audit SHA-256
  `2efa2af4aab59bea56bc310ca9c9b86f672c75accde8cc85de50e36389f13e38`,
  answers SHA-256
  `b3a019a6d86431732b219590515d9c22613027f80223b2863d3e2585e5762ae9`.
  The local mirrored answers match this remote-generated hash. The pipeline
  restarted and entered `sample_repeat`.

2026-09-26, progress check (before opening fresh protected labels):

- All three 400-row original-model first-pass collections completed:
  `initial`, independent `sample_repeat`, and `greedy_same_prompt`. Their
  summaries and append-only audits are mirrored locally under
  `outputs/phase8_remote_3090/`; no fresh correctness result was inspected.
- Exploratory same-seed Phase 7 replay completed for 80 previously known wrong
  cases. Exact full-text reproduction was 0/80 and parsed final-answer
  reproduction was 13/80; this historical set cannot estimate new protected
  efficacy. The replay artifact is mirrored locally.
- Distractor matching froze 398/400 source rows within the preregistered
  token-length window; two rows have no eligible donor. The matching report
  and assignments are mirrored locally.
- The remote sequential pipeline entered the original solver's three-arm
  generation. Latest mirrored progress at this check: 240/1198 prompts, with
  256 append-only audit lines already mirrored. V2/V3 generation, probe fits,
  routing, and protected analysis remain pending.
- During a subsequent progress/timing check, a diagnostic command printed the
  final raw output row of the original solver's three-arm audit (index 569).
  This was inadvertent; no gold label or correctness result for Phase 8 was
  opened, and no protocol choice was changed based on that output. Later
  progress checks must print counts and metadata only.

2026-09-26, connectivity incident:

- Local mirror stopped updating at 02:57 local time, with 873/1198 original
  three-arm completions. At 18:32 local time, direct SSH to port 12017 and
  Jupyter port 12018 both failed TCP connection. Remote execution state is
  therefore unknown; the 873-row local audit is the last confirmed backup,
  not proof that the remote job completed or is still running.
- No Phase 8 protected report exists locally. Do not infer efficacy or resume
  on another host until remote availability and append-only audit state are
  reconciled. The prior completion-time estimate is no longer valid.

Do not report Phase 8 efficacy until all outputs are complete, checked and
copied locally. The Phase 7 protected follow-up is exploratory because it was
opened in the completed prior phase.
