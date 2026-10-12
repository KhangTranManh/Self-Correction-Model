"""Mirror Phase 14 remote outputs to local storage (credentials from a local JSON file).

The JSON file holds host, port, user, password, and optionally remote_root
(default /root/AGI). It stays outside the repository. Hidden-state .npz files
(several GB) are skipped unless --with-hidden is given; their .json summaries
are always copied. Keeps syncing until a
remote done/failed marker has been copied, then makes one final pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import time

import paramiko

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "outputs/phase14_remote"
DIRS = ("outputs/phase14_v1", "logs")
FILES = ("phase14/data/execution_lock_pilot.json", "phase14/data/execution_lock_main.json",
         "phase14/data/main_size.json")
MARKERS = [f"logs/PHASE14_{mode}_{end}" for mode in ("SMOKE", "PILOT", "MAIN", "COMPARISON") for end in ("DONE", "FAILED")]


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


def sync_once(sftp: paramiko.SFTPClient, remote: str, with_hidden: bool) -> int:
    remote_files = [f"{remote}/{rel}" for rel in FILES]
    for rel in DIRS:
        remote_files += [p for p in walk(sftp, f"{remote}/{rel}")
                         if (rel != "logs" or "phase14" in p.lower()) and (with_hidden or not p.endswith(".npz"))]
    copied = 0
    for path in remote_files:
        try:
            info = sftp.stat(path)
        except FileNotFoundError:
            continue
        local = LOCAL / path[len(remote) + 1:]
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size == info.st_size and local.stat().st_mtime >= info.st_mtime:
            continue
        temporary = local.with_name(local.name + ".part")
        sftp.get(path, str(temporary))
        if local.suffix == ".jsonl":  # keep only complete lines of an append-only audit
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
    parser.add_argument("--with-hidden", action="store_true", help="also copy hidden-state .npz files (GBs)")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    remote = config.get("remote_root", "/root/AGI")
    final_pass = False
    while True:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(config["host"], port=int(config["port"]), username=config["user"],
                           password=config["password"], timeout=20)
            sftp = client.open_sftp()
            print(f"{time.strftime('%H:%M:%S')} copied={sync_once(sftp, remote, args.with_hidden)}", flush=True)
            sftp.close()
        except Exception as exc:
            if args.once:
                raise
            print(f"sync_retry={type(exc).__name__}: {exc}", flush=True)
        finally:
            client.close()
        done = any((LOCAL / marker).exists() for marker in MARKERS)
        if args.once or final_pass:
            break
        final_pass = done
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
