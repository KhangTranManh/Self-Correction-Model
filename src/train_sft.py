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

import argparse
from pathlib import Path

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
    # --dataset / --out-dir de chay duoc nhieu NHANH THI NGHIEM tren cung mot config
    # (vd so sanh khung reflect "tool" vs "memory") ma khong phai sua phase1.yaml
    # giua chung -- sua config giua cac lan chay la cach chac chan de sau nay khong
    # con biet adapter nao duoc train tu du lieu nao.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Mac dinh: paths.sft_dataset_out trong phase1.yaml.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Mac dinh: paths.lora_out_dir trong phase1.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config()
    train_cfg = cfg.training

    model, tokenizer = load_model_and_tokenizer(cfg, train_cfg["max_seq_length"])

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

    dataset_path = args.dataset or str(cfg.path("sft_dataset_out"))
    print(f"[train_sft] dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def _format(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(_format, remove_columns=dataset.column_names)

    out_dir = Path(args.out_dir) if args.out_dir else cfg.path("lora_out_dir")
    print(f"[train_sft] luu adapter vao: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_ok = is_bfloat16_supported()
    print(f"[train_sft] tokenizer.eos_token ngay truoc SFTTrainer: {tokenizer.eos_token!r}")
    print(f"[train_sft] correct_eos_token da chup: {correct_eos_token!r}")

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        dataset_text_field="text",
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
        # save_strategy="no": khong ghi checkpoint trung gian moi epoch xuong dia
        # (LoRA + optimizer state cong don qua nhieu epoch se ton dung luong tren
        # may thue GPU). Adapter cuoi cung se duoc push thang len HF Hub ben duoi.
        save_strategy="no",
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

    trainer.train()

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
