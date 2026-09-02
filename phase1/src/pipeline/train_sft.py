"""QLoRA SFT tren data/processed/phase1_sft.jsonl bang Unsloth (FastLanguageModel).

YEU CAU PHAN CUNG: GPU compute capability >= 7.5 (Turing/Ampere/Ada/Hopper tro len --
vd T4, L40S, A100, 4090, H100...). KHONG dung duoc tren V100 (Volta, CC 7.0) -- da
verify thuc te: Unsloth/Axolotl ban moi deu ep torch len ban da bo kernel CC 7.0. Neu
quay lai may V100, phai dung ban plain transformers+peft+trl (xem git history truoc
commit doi sang Unsloth nay).

Chay tren may co GPU (24-48GB+, CC>=7.5):
    python -m src.train_sft
"""
from __future__ import annotations

import os
from pathlib import Path

# Keep model downloads in the project workspace. On this Windows host the
# default Hugging Face cache probe can block during Unsloth import, before any
# training code runs. Respect an explicit caller override when one is present.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(_HF_CACHE / "xet"))
if os.name == "nt":
    # Native torch.compile/Triton kernels currently terminate this Windows
    # host with 0xC0000005 during Qwen model loading. Unsloth's 4-bit loader,
    # LoRA patching and gradient checkpointing remain enabled with compilation
    # disabled, which preserves the memory-saving path needed on a 12GB GPU.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

# QUAN TRONG: unsloth PHAI duoc import truoc trl/transformers/peft -- neu khong,
# Unsloth khong the patch day du cac class cua trl (SFTConfig/SFTTrainer van tro
# ve ban chua patch), gay loi "eos_token ('<EOS_TOKEN>') is not found in the
# vocabulary" khi khoi tao SFTTrainer (da xac nhan thuc te: import sai thu tu la
# nguyen nhan goc, khong lien quan model nao dang dung).
import unsloth  # noqa: F401  -- phai dung dau tien, truoc moi import trl khac

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from src.config import load_config
from src.model_loading import load_model_and_tokenizer

_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def main() -> None:
    # Training does not call the correction API. Do not require or upload its
    # credentials to a rented GPU host.
    cfg = load_config(require_deepseek=False)
    train_cfg = cfg.training

    # Optional local snapshot override avoids any Hugging Face downloader work
    # on a training host after the model has already been cached and verified.
    model_name_override = os.environ.get("AGI_MODEL_PATH") or None
    model, tokenizer = load_model_and_tokenizer(
        cfg,
        train_cfg["max_seq_length"],
        model_name_override=model_name_override,
    )
    if model_name_override:
        print(f"[train_sft] loading verified local model snapshot: {model_name_override}")

    # Chup lai eos_token DUNG ngay sau khi load, truoc khi goi bat ky ham Unsloth
    # nao khac -- da phat hien get_peft_model()/patching cua Unsloth co the mutate
    # lai tokenizer.eos_token thanh placeholder hong "<EOS_TOKEN>" sau buoc nay.
    correct_eos_token = tokenizer.eos_token
    print(f"[train_sft] eos_token ngay sau load: {correct_eos_token!r}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_dataset_path = cfg.path("sft_training_dataset_out")
    if not training_dataset_path.exists():
        raise FileNotFoundError(
            f"Chua co training dataset {training_dataset_path}. "
            "Hay chay: python -m src.prepare_training_dataset"
        )
    dataset = load_dataset("json", data_files=str(training_dataset_path), split="train")

    def _to_prompt_completion(example):
        """Mask the failed attempt and verifier feedback from the training loss.

        A Phase-1 row is:
          system -> user -> failed assistant attempt -> tool feedback
          -> verified assistant correction

        TRL's conversational prompt-completion format keeps the first four
        messages as context and computes loss only on the final completion.  In
        particular, the model must never be optimized to reproduce the failed
        assistant attempt.
        """
        messages = example["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("Mau SFT khong ket thuc bang verified assistant correction")
        return {
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }

    dataset = dataset.map(_to_prompt_completion, remove_columns=dataset.column_names)

    out_dir = cfg.path("lora_out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_ok = is_bfloat16_supported()
    print(f"[train_sft] tokenizer.eos_token ngay truoc SFTTrainer: {tokenizer.eos_token!r}")
    print(f"[train_sft] correct_eos_token da chup: {correct_eos_token!r}")

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        # Dataset o conversational prompt-completion format; chi completion
        # (verified correction cuoi) duoc tinh loss. Failed attempt va tool
        # feedback van nam trong prompt de lam context, nhung label bi mask.
        completion_only_loss=True,
        max_length=train_cfg["max_seq_length"],
        # Dung gia tri da chup lai tu truoc (correct_eos_token), KHONG doc lai
        # tokenizer.eos_token o day -- co the da bi Unsloth mutate lai thanh
        # placeholder hong "<EOS_TOKEN>" sau get_peft_model().
        eos_token=correct_eos_token,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        fp16=not bf16_ok,
        bf16=bf16_ok,
        logging_steps=10,
        # Native Windows/WDDM can reset a nearly-full GPU without giving Python
        # a catchable exception. Keep two resumable checkpoints so at most ten
        # optimizer steps are lost if that happens.
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )
    print(f"[train_sft] sft_config.eos_token ngay sau SFTConfig(): {sft_config.eos_token!r}")

    trainer = SFTTrainer(
        model=model,
        # trl>=1.0 doi ten tham so nay tu 'tokenizer' -> 'processing_class', va bo
        # 'dataset_text_field'/'max_seq_length' truc tiep tren SFTTrainer -- gio phai
        # nam trong SFTConfig (field 'max_length', khong phai 'max_seq_length').
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    checkpoints = [
        path
        for path in out_dir.glob("checkpoint-*")
        if path.is_dir()
        and path.name.removeprefix("checkpoint-").isdigit()
        and (path / "trainer_state.json").exists()
    ]
    resume_checkpoint = max(
        checkpoints,
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        default=None,
    )
    if resume_checkpoint is not None:
        print(f"[train_sft] resuming from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if train_cfg.get("push_to_hub", False):
        repo_id = cfg.huggingface.repo_id
        # push_to_hub cua transformers/peft tu luu vao 1 thu muc tam roi upload va don
        # dep, khong de lai file lon tren dia sau khi chay xong.
        model.push_to_hub(repo_id, token=cfg.huggingface.token, private=cfg.huggingface.private)
        tokenizer.push_to_hub(repo_id, token=cfg.huggingface.token, private=cfg.huggingface.private)
        print(f"Da push LoRA adapter len: https://huggingface.co/{repo_id}")
    else:
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print(f"Da luu LoRA adapter vao: {out_dir}")


if __name__ == "__main__":
    main()
