"""Push 1 thu muc adapter da co san (local) len HuggingFace Hub.

Dung khi: da train xong tu truoc (co local checkpoint do save_strategy khac,
hoac push_to_hub=false luc train), gio muon day len Hub roi xoa local de tiet
kiem dung luong tren may thue GPU -- khong can train lai.

Chay:
    python -m src.push_to_hub --local-dir outputs/phase1_lora
    python -m src.push_to_hub --local-dir outputs/phase1_lora --repo-id user/ten-repo-khac
"""
from __future__ import annotations

import argparse

from huggingface_hub import HfApi

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True, help="Thu muc chua adapter (vd outputs/phase1_lora)")
    parser.add_argument("--repo-id", default=None, help="Mac dinh lay tu configs/phase1.yaml (huggingface.repo_id)")
    parser.add_argument("--public", action="store_true", help="Push public thay vi private (mac dinh private)")
    args = parser.parse_args()

    cfg = load_config()
    if cfg.huggingface is None:
        raise RuntimeError("HF_TOKEN chua duoc set trong .env -- can de push len HuggingFace Hub")

    repo_id = args.repo_id or cfg.huggingface.repo_id
    private = not args.public

    api = HfApi(token=cfg.huggingface.token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=args.local_dir, repo_id=repo_id, repo_type="model")

    print(f"Da push '{args.local_dir}' len: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
