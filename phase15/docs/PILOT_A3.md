# Phase 15 dev diagnostic, amendment P15-A3

Recorded 2026-11-03 before P15-A3 model generation. P15-A2 completed its 40
frozen S-format episodes, but its A0, A1, and A2 arms used different sampling
seeds. Consequently the observed arm differences mix intervention effects with
decoding variation. This diagnostic addresses that confound; it is not a new
evaluation split or a confirmation run.

Use the exact P15-A2 40-episode manifest (SHA-256
`1839aba51f447c4f0da8b1b95a8fea9775072a81736f50f1faf96e418ac1c2e9`),
Qwen/Qwen2.5-7B-Instruct, the same pinned package versions, NF4 double-quant
FP16 compute, LoRA rank/modules, 3 steps at 5e-4, prompts, 0.7 temperature,
top-p 1, and 768-token output cap. For each episode evaluate its two probes
and unrelated control under A0 base, A1 context, and A2 naive LoRA. Use
`seed_for(episode_id + "|a0") + item_index` for all three arms on that item.
Reset A2 to one frozen initial adapter before every episode and verify the
post-episode reset. Omit A3 self-edit and screening because they do not answer
the seed-confound question. Run all 40 episodes in a single dedicated Colab
process, with a 7-hour wall and append-only per-item audit. A partial run is
descriptive only; do not splice it with P15-A2.

Primary paired outcome: probe joint correctness (correct final answer and
format-clean). Report A1-A0 first because the preregistered harness gate
depends on A1; report A2-A0 descriptively. Also report format-clean rate,
answer correctness, per-item fix/harm counts, and unrelated-control
regressions. Preserve all raw completions and independently rescore them.
No protected evaluation split is opened; no Phase 15 scientific go/no-go will
be based on this development diagnostic alone.
