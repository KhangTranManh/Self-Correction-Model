"""Upload the two checked Phase 4 LoRA adapters to private HF repos.

The project-root .env is read for HF_TOKEN but is never uploaded or printed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


ROOT = Path(__file__).resolve().parents[3]
MODELS = (
    (
        "Kxck/phase4_exploration_warmstart_v2",
        ROOT / "outputs/phase4_exploration_warmstart_v2/final_adapter",
        ROOT / "phase5/docs/model_cards/phase4_exploration_warmstart_v2.md",
        "19fb45c77d3991470ed50cc2d50f12c0cc4667802d3fdd02c6c8b48d03517be4",
    ),
    (
        "Kxck/phase4_correction_sft_v3",
        ROOT / "outputs/phase4_correction_sft_v3/final_adapter",
        ROOT / "phase5/docs/model_cards/phase4_correction_sft_v3.md",
        "6181eb886cee6cbc0f5b11e4d0d521ef3c3fb550172c1dbdf2b17b137198e08a",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    api = HfApi(token=token)
    if api.whoami()["name"] != "Kxck":
        raise RuntimeError("HF_TOKEN does not belong to the expected Kxck account")

    # Verify both local artifacts before creating either remote repository.
    for repo_id, folder, card, expected in MODELS:
        if not folder.is_dir() or not card.is_file():
            raise FileNotFoundError(f"Missing adapter or model card for {repo_id}")
        actual = sha256(folder / "adapter_model.safetensors")
        if actual != expected:
            raise RuntimeError(f"Local adapter hash mismatch for {repo_id}")
        print(f"Local adapter verified: {repo_id}", flush=True)

    for repo_id, folder, card, expected in MODELS:
        try:
            existing = api.model_info(repo_id)
        except RepositoryNotFoundError:
            existing = None
        if existing is None:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True)
            print(f"Created private repo: {repo_id}", flush=True)
        elif not existing.private:
            raise RuntimeError(f"Refusing to upload to public repo: {repo_id}")

        api.upload_folder(
            folder_path=str(folder), repo_id=repo_id, repo_type="model",
            ignore_patterns=["README.md"],
            commit_message="Upload checked Phase 4 LoRA adapter and tokenizer",
        )
        api.upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md",
            repo_id=repo_id, repo_type="model",
            commit_message="Document adapter lineage and Phase 4 limits",
        )
        info = api.model_info(repo_id, files_metadata=True)
        weights = next((file for file in info.siblings
                        if file.rfilename == "adapter_model.safetensors"), None)
        if weights is None or weights.size != (folder / "adapter_model.safetensors").stat().st_size:
            raise RuntimeError(f"Remote adapter size mismatch for {repo_id}")
        remote_sha = (weights.lfs or {}).get("sha256")
        if remote_sha and remote_sha != expected:
            raise RuntimeError(f"Remote adapter hash mismatch for {repo_id}")
        print(f"Verified private repo: {repo_id} revision={info.sha} sha256={remote_sha or 'unavailable'}", flush=True)


if __name__ == "__main__":
    main()
