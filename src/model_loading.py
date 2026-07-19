"""Helper dung chung: load small model qua Unsloth, ap dung chat template dung, va
bung tokenizer text ra khoi processor da phuong thuc (neu model tra ve processor VL
nhu Qwen3VLProcessor thay vi tokenizer text thuan -- gap voi Jackrong/Qwopus3.5-9B-v3).

Dung chung boi generate_attempts.py, train_sft.py, evaluate_self_correction.py de
tranh 3 noi tu xu ly rieng le roi lech nhau.
"""
from __future__ import annotations

# unsloth PHAI import truoc transformers/trl/peft de patch day du (xem ghi chu
# chi tiet trong train_sft.py) -- anh huong ca cac module import module nay sau.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from transformers import AutoTokenizer

from src.config import Config


def load_model_and_tokenizer(
    cfg: Config, max_seq_length: int, model_name_override: str | None = None
):
    model_name = model_name_override or cfg.small_model["name_or_path"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=cfg.small_model["load_in_4bit"],
    )

    chat_template = cfg.small_model.get("chat_template")
    if chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # Bung tokenizer text that ra khoi processor da phuong thuc -- an toan cho ca
    # model VL-wrapped (co attribute .tokenizer) lan model text thuan (hasattr fail
    # -> giu nguyen tokenizer nhu cu, khong anh huong).
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    # Doi voi mot so checkpoint goc la VL (vd Jackrong/Qwopus3.5-9B-v3), qua trinh
    # load/xu ly cua Unsloth co the ghi nham tokenizer.eos_token thanh placeholder
    # hong (vd literal "<EOS_TOKEN>") khong ton tai trong vocab, gay loi khi train.
    # Doi chieu lai voi tokenizer goc doc thang qua transformers (khong qua Unsloth)
    # -- neu eos_token bi lech va gia tri dung ton tai trong vocab, tu sua lai.
    ref_tokenizer = AutoTokenizer.from_pretrained(model_name)
    correct_eos = ref_tokenizer.special_tokens_map.get("eos_token")
    if correct_eos and tokenizer.eos_token != correct_eos and correct_eos in tokenizer.get_vocab():
        print(
            f"[model_loading] Fix eos_token: '{tokenizer.eos_token}' (hong) -> '{correct_eos}' (dung)"
        )
        tokenizer.eos_token = correct_eos

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
