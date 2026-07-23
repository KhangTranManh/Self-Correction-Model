"""Phase 2 — KTO tren buoc phe binh (CAN GPU).

>>> TRANG THAI: viet theo pattern phase1/src/pipeline/train_sft.py + API KTO cua trl,
>>> nhung CHUA CHAY THAT. Phai verify bang chay that khi thue GPU (dac biet buoc
>>> khoi-tao-tu-adapter-SFT va viec trl KTOTrainer nuot dinh dang conversational
>>> prompt/completion/label -- hai cho de sai nhat, phai kiem chung khi chay).

Tai sao KTO (khong phai DPO/SFT):
- SFT day DINH DANG cua viec sua, khong day KY NANG PHAN BIET dung/sai (day la tran
  cua Phase 1 -- xem note.txt). KTO/DPO them gradient "uu tien phe binh that hon phe
  binh bia".
- KTO nhan mau LE gan nhan nhi phan -> khop du lieu Phase 1 (khong co cap san). Xem
  build_preference.py.

KTO la buoc SAU SFT: khoi tao tu adapter AGI_v3 (init_adapter trong phase2.yaml),
KHONG train tu base -- neu tu base se vut bo nang luc SFT.

Chay (may GPU CC>=7.5, sau khi sync phase1/ + phase2/, dat cwd = phase2/):
    python train_kto.py                      # doc configs/phase2.yaml
    python train_kto.py --init-adapter ""    # doi chung: KTO tu base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# unsloth PHAI import truoc trl/transformers (ly do y het train_sft.py: neu khong,
# patch cua Unsloth khong day du -> loi eos_token '<EOS_TOKEN>' khi khoi tao trainer).
import unsloth  # noqa: F401

import yaml
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

PHASE2_DIR = Path(__file__).resolve().parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))
from src.config import load_config  # noqa: E402  -- base model + HF token (.env Phase 1)

_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_phase2_cfg() -> dict:
    with open(PHASE2_DIR / "configs" / "phase2.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-adapter", type=str, default=None,
                        help="Ghi de init_adapter trong phase2.yaml. '' = train tu base.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    p2 = _load_phase2_cfg()
    cfg = load_config()  # Phase 1: base model name, HF token, .env
    kto_cfg = p2["kto"]
    lora_cfg = p2["lora"]

    init_adapter = args.init_adapter if args.init_adapter is not None else p2.get("init_adapter") or ""
    max_seq_length = kto_cfg["max_length"]

    # Khoi tao model. init_adapter != "" -> load base + adapter SFT (AGI_v3) de KTO
    # TIEP TUC tu do. Unsloth load thang tu repo/thu muc adapter (tu tim base qua
    # adapter_config.json). Neu "" -> load base roi gan LoRA moi (doi chung).
    #
    # [!] VERIFY KHI CHAY THAT: voi init_adapter, adapter phai o trang thai TRAINABLE
    #     (khong goi for_inference). Neu Unsloth load adapter o che do inference, phai
    #     goi FastLanguageModel.get_peft_model lai / bat requires_grad -- kiem tra
    #     truoc khi tin trainer dang that su cap nhat trong so.
    model_name = init_adapter or cfg.small_model["name_or_path"]
    print(f"[train_kto] khoi tao tu: {model_name} (init_adapter={init_adapter!r})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=cfg.small_model.get("load_in_4bit", True),
        dtype=None,
    )
    correct_eos_token = tokenizer.eos_token
    print(f"[train_kto] eos_token ngay sau load: {correct_eos_token!r}")

    if not init_adapter:
        model = FastLanguageModel.get_peft_model(
            model, r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"], target_modules=_TARGET_MODULES,
            bias="none", use_gradient_checkpointing="unsloth",
        )

    dataset_path = args.dataset or p2["dataset"]["path"]
    print(f"[train_kto] dataset: {dataset_path}")
    # KTO doc thang cot prompt/completion/label (conversational). trl tu ap chat
    # template. [!] VERIFY: mot so ban trl doi cot conversational qua
    # apply_chat_template san -- neu loi schema, map thu cong bang tokenizer o day.
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    out_dir = Path(args.out_dir) if args.out_dir else PHASE2_DIR / p2["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16_ok = is_bfloat16_supported()

    kto_config = KTOConfig(
        output_dir=str(out_dir),
        beta=kto_cfg["beta"],
        desirable_weight=kto_cfg["desirable_weight"],
        undesirable_weight=kto_cfg["undesirable_weight"],
        max_length=kto_cfg["max_length"],
        max_prompt_length=kto_cfg["max_prompt_length"],
        num_train_epochs=kto_cfg["epochs"],
        per_device_train_batch_size=kto_cfg["per_device_batch_size"],
        gradient_accumulation_steps=kto_cfg["gradient_accumulation_steps"],
        learning_rate=kto_cfg["learning_rate"],
        fp16=not bf16_ok,
        bf16=bf16_ok,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()

    push = p2["output"].get("push_to_hub", False)
    if push and cfg.huggingface is not None:
        repo_id = p2["output"]["repo_id"]
        model.push_to_hub(repo_id, token=cfg.huggingface.token, private=cfg.huggingface.private)
        tokenizer.push_to_hub(repo_id, token=cfg.huggingface.token, private=cfg.huggingface.private)
        print(f"Da push adapter KTO len: https://huggingface.co/{repo_id}")
    else:
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print(f"Da luu adapter KTO vao: {out_dir}")


if __name__ == "__main__":
    main()
