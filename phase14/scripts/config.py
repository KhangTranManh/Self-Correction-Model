"""Phase 14 fixed settings: candidate models, datasets, decoding, and split sizes.

Everything that the preregistration freezes lives here, so the execution lock
covers it with one file. Model selection is final only when
docs/PREREGISTRATION.md is frozen; candidates not selected are never run on
the main holdout.

Comparability with Phases 12-13 is the first constraint: prompts are imported
unchanged from the earlier phases (never retyped), and the primary view uses
the same 768-token cap and 4096-token context. Attempts are generated once up
to GENERATION_CAP; with a fixed per-request seed the first 768 tokens equal a
768-cap run, so the primary view is the 768-token prefix and the full output
is a cap-sensitivity view (checked in scripts/smoke_model.py).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "phase14"
RAW = PHASE / "data/raw"
SOURCES = PHASE / "data/sources_v1"
OUT = ROOT / "outputs/phase14_v1"

# Revisions pinned from the Hugging Face API on 2026-10-12. "gated" models need
# an accepted license and HF_TOKEN on the GPU host.
MODELS = {
    # Amendment 2026-10-13 (before any generation): Meta's gated repo was still awaiting
    # access review, so Llama-3.1-8B-Instruct is loaded from the ungated NousResearch copy.
    # Its four weight shards are byte-identical to meta-llama/Llama-3.1-8B-Instruct at
    # 0e9e39f249a16976918f6564b8830bc894c89659 (LFS sha256 below, compared via the Hub API),
    # as are config.json, generation_config.json, tokenizer.json and
    # special_tokens_map.json; only tokenizer_config.json (the chat template) differs.
    "llama31_8b": {"repo": "NousResearch/Meta-Llama-3.1-8B-Instruct",
                   "revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
                   "family": "llama", "gated": False,
                   "weights_identical_to": "meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659",
                   "weight_sha256": {
                       "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
                       "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
                       "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
                       "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b"},
                   # NousResearch's tokenizer_config.json is an older copy (no system date block in
                   # the chat template, clean_up_tokenization_spaces=True). The tokenizer is taken
                   # from this ungated repo, whose tokenizer_config.json, tokenizer.json and
                   # special_tokens_map.json have the same git blob ids as Meta's @ 0e9e39f2.
                   "tokenizer_repo": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
                   "tokenizer_revision": "442e7f522277df12f53d63e8384087af31fbfc4b"},
    "mistral_7b": {"repo": "mistralai/Mistral-7B-Instruct-v0.3",
                   "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
                   "family": "mistral", "gated": False},
    "gemma2_9b": {"repo": "google/gemma-2-9b-it",
                  "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
                  "family": "gemma", "gated": True},
    "phi35_mini": {"repo": "microsoft/Phi-3.5-mini-instruct",
                   "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
                   "family": "phi", "gated": False},
    # Anchors (rows 1-2 of the report's generalization table): the checkpoint of
    # Phases 5-13 and its untrained base, run unchanged on the same data.
    "original_solver": {"repo": "Kxck/Self_Correction_v1",
                        "revision": "6437f947999168a0ce2a98a86e4252fc77160a33",
                        "family": "qwen", "gated": False},
    "qwen25_instruct": {"repo": "Qwen/Qwen2.5-7B-Instruct",
                        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                        "family": "qwen", "gated": False},
}
# The two new models, chosen by the user on 2026-10-12: Llama-3.1-8B-Instruct
# (Meta) and Mistral-7B-Instruct-v0.3 (Mistral AI), two families other than
# Qwen. A model that fails scripts/smoke_model.py is replaced only by
# phi35_mini, and the swap is recorded before any holdout run. gemma2_9b stays
# listed but is not run (BF16-trained; FP16 on a V100 is risky).


def tokenizer_source(spec: dict) -> tuple[str, str]:
    """(repo, revision) of the tokenizer: the model repo unless a separate one is pinned."""
    return spec.get("tokenizer_repo", spec["repo"]), spec.get("tokenizer_revision", spec["revision"])


SELECTED = ("llama31_8b", "mistral_7b")
ANCHOR = ("original_solver", "qwen25_instruct")  # comparison rows; not counted in the generality rule
# Amendment 2026-10-13 after the pilot (GPU time: ~7.5 h left of 10 h, full main ~18 h; decided by
# the user): the main run covers only the two new models. The anchors were run in smoke and pilot
# but are not regenerated in main; the report's Self_Correction and Qwen-Instruct GSM8K results come
# from Phases 12-13 and the paper checks.
RUN_MODELS = SELECTED
# Amendment 2026-10-13 after the protected opening (decided by the user, same host still
# running): a completion run fills the whole grid with the identical protocol - the anchors on
# all four datasets, the new models on GSM8K and SVAMP, the bridge set, the base_c regression
# re-run. These cells are added with analyze.py --add-comparison; E1/E2 are not recomputed.
RUN_MODELS = (*SELECTED, *ANCHOR)
PILOT_MODELS = (*SELECTED, *ANCHOR)  # models of the completed pilot

DATASETS = ("asdiv", "math13")  # endpoint datasets: fresh, easier-or-equal to GSM8K; MATH levels 1-3
# Comparison datasets: the frozen Phase 13 (GSM8K test 750-1318, 561 problems)
# and Phase 12 (SVAMP, 960) holdouts, solved and judged by each model itself,
# so every model fills the report's GSM8K and SVAMP columns. Their labels were
# opened earlier; untrained models are only evaluated on them, never tuned.
# All four models, Self_Correction included, run the identical Phase 14
# protocol here (fresh attempts, both judges with log-probabilities), so every
# measure is computed the same way for every model.
COMPARISON = ("gsm8k13", "svamp12")
# Regression-only sets: the same problems, but read from the report's own
# Phase 13 / Phase 12 outputs (Self_Correction attempts and base judgments).
# Used only to check that Phase 14 code reproduces the report; never in the
# main tables, which use the uniform fresh runs above.
REGRESSION_SETS = {"gsm8k13_report": "phase13", "svamp12_report": "phase12"}
# Optional: problem ids excluded by the GSM8K audit (42 of 561 seen in an old
# evaluation); when present every GSM8K number is also reported on the 519.
GSM8K_AUDIT_EXCLUDED = PHASE / "data/gsm8k13_audit_excluded.json"
# Numbers printed in the report draft (2026-10-12) for the untrained base judge
# of the original solver, on all 561 GSM8K and 960 SVAMP problems. The
# regression-only sets reuse the report's own outputs, so regression.py must
# reproduce these values to the printed precision.
REPORT_REFERENCE = {
    "original_solver/gsm8k13_report": {
        "base": {"one_right_pairs": 288, "single_accuracy": 0.523, "skill_index": 0.215,
                 "position_2_rate": 0.537, "invented_rate": 0.167, "tie_break": 0.601,
                 "consistent_accuracy": 0.710, "coverage": 0.479, "all_pairs_accuracy": 0.340},
        # Phase 13 final report, untrained constrained judge; Phase 14 re-runs it (greedy,
        # same prompt) to obtain log-probabilities, so small batching differences are allowed.
        "base_c": {"single_accuracy": 0.549, "position_2_rate": 0.370, "invented_rate": 0.049,
                   "tie_break": 0.568, "consistent_accuracy": 0.638, "coverage": 0.490},
    },
    "original_solver/svamp12_report": {
        "base": {"skill_index": 0.248, "position_2_rate": 0.543, "invented_rate": 0.132,
                 "tie_break": 0.609, "consistent_accuracy": 0.687, "coverage": 0.583,
                 "all_pairs_accuracy": 0.401},
    },
}
EXACT = {"base"}               # reused judgments must match the report to its printed precision
RERUN_TOLERANCE = 0.03         # re-generated judgments: allowed absolute difference per measure
# Report section 4.5 (descriptive): Qwen2.5-7B-Instruct judging its own GSM8K pairs
# (paper check T1: 768-token judge view, Phase 13 seeds tagged "t1"). Keys use the
# Phase 14 mode names: "lenient" = the project parser (what the paper checks call
# "strict"), "rescued" = the T4b reading (what they call "lenient"). Corrected
# 2026-10-12 before any lock: the earlier values 0.655 / 0.658 came from the
# 1536-token view (T3) and had the two mode names swapped. Phase 14 uses new seeds,
# so only agreement within the bootstrap interval is checked, not equality.
INSTRUCT_REFERENCE = {"qwen25_instruct/gsm8k13": {"tie_break_lenient": 0.629, "tie_break_rescued": 0.605}}
# Bridge to the old results: the 486 judged pairs of Phase 13 (GSM8K test
# 750-1318, solutions by the original solver) re-judged by each model. The
# solutions are Phase 13's own outputs, so only the judge differs. Requires
# outputs/phase13_v1/ (samples + analysis) on the GPU host.
BRIDGE = "bridge13"

# Hidden states (Transformers forward pass over the exact text, after vLLM
# generation, as in Phases 5 and 13). All layers, float16.
#   answer site: last token of each judged solution (s1, s2, s3) after the
#                solve prompt, plus its mean token log-probability
#   judge site:  last prompt token of every judged task (base and base_c)
#   verdict:     teacher-forced log P(A), log P(B) after "Verdict: Solution"
#                (base_c), to check against vLLM's log-probabilities
#   direct:      constrained judge prompt + "Verdict: Solution" forced right
#                after the prompt (no written check); log P(A), log P(B) at the
#                final layer and the logit lens (final norm + LM head) at every
#                layer, both orders -> Ren et al.-style order averaging at full
#                coverage and the layer where the choice becomes decodable
HIDDEN_SITES = ("answer", "judge", "verdict", "direct")
EAGER_ATTENTION = {"gemma"}          # Gemma-2 logit soft-capping needs eager attention in Transformers

PRIMARY_CAP = 768              # identical to Phases 8-13; all judging uses this view
GENERATION_CAP = 1536          # attempts only; the prefix beyond 768 is a sensitivity view
MAX_MODEL_LEN = 4096           # identical to Phases 8-13
TEMPERATURE = 0.7              # attempts; judges are greedy (as in Phases 9-13)
SAMPLES = 5
JUDGED = (2, 3)                # pairs (s1, s2) and (s1, s3)
VERDICT_TOP_LOGPROBS = 20      # read A/B log-probabilities at the verdict token
PILOT_SIZE = 100               # per dataset; burned after the pilot report
TARGET_ONE_RIGHT_PAIRS = 300   # per model x dataset cell
SVAMP_JACCARD_MAX = 0.5        # ASDiv rows this close to a SVAMP problem are excluded
SEED = 20261012
