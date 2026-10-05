"""Mirror Phase 11 remote output folders to local storage (credentials from a local JSON file).

Keeps syncing until the remote done/failed marker has been copied, then makes
one final pass so late outputs are never missed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import time

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/AGI_phase11"
LOCAL = ROOT / "outputs/phase11_remote_v100"
DIRS = ("outputs/phase11_v1",)
FILES = ("logs/phase11_pipeline.log", "logs/phase11_analysis_stdout.txt",
         "logs/PHASE11_DONE", "logs/PHASE11_FAILED", "phase11/data/execution_lock_v1.json")


def walk(sftp: paramiko.SFTPClient, directory: str) -> list[str]:
    try:
        entries = sftp.listdir_attr(directory)
    except FileNotFoundError:
        return []
    found = []
    for entry in entries:
        path = f"{directory}/{entry.filename}"
        found += walk(sftp, path) if stat.S_ISDIR(entry.st_mode) else [path]
    return found


def sync_once(sftp: paramiko.SFTPClient) -> int:
    remote_files = [f"{REMOTE}/{rel}" for rel in FILES]
    for rel in DIRS:
        remote_files += walk(sftp, f"{REMOTE}/{rel}")
    copied = 0
    for remote in remote_files:
        try:
            info = sftp.stat(remote)
        except FileNotFoundError:
            continue
        local = LOCAL / remote[len(REMOTE) + 1:]
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size == info.st_size \
                and local.stat().st_mtime >= info.st_mtime:
            continue
        temporary = local.with_name(local.name + ".part")
        sftp.get(remote, str(temporary))
        if local.suffix == ".jsonl":
            data = temporary.read_bytes()
            complete = data.rfind(b"\n") + 1
            if complete != len(data):
                temporary.write_bytes(data[:complete])
        temporary.replace(local)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    final_pass = False
    while True:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(config["host"], port=int(config["port"]), username=config["user"],
                           password=config["password"], timeout=20)
            sftp = client.open_sftp()
            print(f"{time.strftime('%H:%M:%S')} copied={sync_once(sftp)}", flush=True)
            sftp.close()
        except Exception as exc:
            if args.once:
                raise
            print(f"sync_retry={type(exc).__name__}: {exc}", flush=True)
        finally:
            client.close()
        done = (LOCAL / "logs/PHASE11_DONE").exists() or (LOCAL / "logs/PHASE11_FAILED").exists()
        if args.once or final_pass:
            break
        final_pass = done
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
