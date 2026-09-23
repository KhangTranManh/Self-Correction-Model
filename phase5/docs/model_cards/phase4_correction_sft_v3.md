---
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- phase4
- research
---

# Phase 4 correction SFT V3

This is the Phase 4 **LoRA adapter**, not a merged model. Its parent is a
**merged V2 checkpoint** reconstructed by loading
[`Kxck/Self_Correction_v1`](https://huggingface.co/Kxck/Self_Correction_v1),
attaching the Phase 4 exploration warm-start V2 adapter, and merging that
adapter into the base. Loading this V3 adapter directly on the original solver
would use the wrong parent. The recorded training path
`/root/models/phase4_warmstart_v2_merged` is a local checkpoint path, not a
Hugging Face model ID.

On the 100-row Phase 4 development set, V3 had 3 fixes and 2 harms. It was not
promoted as a reliable autonomous correction improvement.

The `adapter_model.safetensors` SHA-256 is
`6181eb886cee6cbc0f5b11e4d0d521ef3c3fb550172c1dbdf2b17b137198e08a`.
The local source is `outputs/phase4_correction_sft_v3/final_adapter`.
