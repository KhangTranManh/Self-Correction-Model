---
base_model: Kxck/Self_Correction_v1
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- phase4
- research
---

# Phase 4 exploration warm-start V2

This is the Phase 4 **LoRA adapter**, not a merged model. Its parent is
[`Kxck/Self_Correction_v1`](https://huggingface.co/Kxck/Self_Correction_v1).
The original training configuration recorded the parent at the local path
`/root/models/Kxck_Self_Correction_v1`; that path refers to the same checkpoint.

The adapter is retained as a diagnostic pilot reference for Phase 5. Its 74%
development result did not reproduce in a 73% rerun. It is not a validated
autonomous correction improvement.

The `adapter_model.safetensors` SHA-256 is
`19fb45c77d3991470ed50cc2d50f12c0cc4667802d3fdd02c6c8b48d03517be4`.
The local source is `outputs/phase4_exploration_warmstart_v2/final_adapter`.
