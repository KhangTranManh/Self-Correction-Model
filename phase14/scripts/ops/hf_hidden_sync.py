"""Mirror Phase 14 hidden-state arrays through a private Hugging Face dataset repo.

  host side:  hf_hidden_sync.py push <repo_id>   (on the GPU host, loops until the run ends)
  local side: hf_hidden_sync.py pull <repo_id> <local outputs root>   (loops, downloads new files)

A hidden-state file is pushed only once its .json summary exists (extraction complete). The
SSH link to the host is ~250 KB/s; the Hub path is much faster. HF_TOKEN from .env (write
scope for push). Paths in the repo mirror outputs/phase14_v1/hidden/.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DONE_MARKERS = ("PHASE14_COMPLETION_DONE",)


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def push(repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    base = ROOT / "outputs/phase14_v1/hidden"
    log = ROOT / "logs/phase14_complete.log"
    sent = set(f for f in api.list_repo_files(repo, repo_type="dataset"))
    while True:
        finished = log.exists() and any(m in log.read_text(encoding="utf-8", errors="replace") for m in DONE_MARKERS)
        for npz in sorted(base.rglob("*.npz")):
            rel = npz.relative_to(base).as_posix()
            if rel in sent or not npz.with_suffix(".json").exists():
                continue
            for f in (npz.with_suffix(".json"), npz):
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f.relative_to(base).as_posix(),
                                repo_id=repo, repo_type="dataset", commit_message=f"add {f.name}")
            sent.add(rel)
            print(time.strftime("%H:%M:%S"), "pushed", rel, round(npz.stat().st_size / 2**30, 2), "GB", flush=True)
        pending = [p for p in base.rglob("*.npz") if p.relative_to(base).as_posix() not in sent]
        if finished and not pending:
            api.upload_file(path_or_fileobj=b"done\n", path_in_repo="PUSH_DONE_MARKER", repo_id=repo,
                            repo_type="dataset", commit_message="all hidden states pushed")
            print("PUSH_DONE", flush=True)
            return
        time.sleep(120)


def pull(repo: str, local_root: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token())
    dest = Path(local_root) / "outputs/phase14_v1/hidden"
    while True:
        files = api.list_repo_files(repo, repo_type="dataset")
        for f in sorted(files):
            if not f.endswith((".npz", ".json")) or (dest / f).exists():
                continue
            hf_hub_download(repo, f, repo_type="dataset", local_dir=str(dest), token=token())
            print(time.strftime("%H:%M:%S"), "pulled", f, flush=True)
        if "PUSH_DONE_MARKER" in files:
            print("PULL_DONE", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    mode, repo = sys.argv[1], sys.argv[2]
    push(repo) if mode == "push" else pull(repo, sys.argv[3])
