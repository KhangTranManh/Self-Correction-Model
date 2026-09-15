"""Copy stable training checkpoints and final evidence from a GPU to local disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import posixpath
import stat
import time

import paramiko


def inventory(sftp, root: str, prefix: str = "") -> list[tuple[str, int, int]]:
    result = []
    for entry in sftp.listdir_attr(posixpath.join(root, prefix)):
        if entry.filename in {".", ".."} or "/" in entry.filename or "\\" in entry.filename:
            raise ValueError("Unsafe remote filename")
        relative = posixpath.join(prefix, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            result.extend(inventory(sftp, root, relative))
        elif stat.S_ISREG(entry.st_mode):
            result.append((relative, entry.st_size, entry.st_mtime))
    return sorted(result)


def copy_tree(sftp, remote: str, local: Path, files: list[tuple[str, int, int]], snapshot_jsonl: bool = False) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for relative, size, _ in files:
        destination = local.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == size:
            continue
        temporary = destination.with_name(destination.name + ".downloading")
        if snapshot_jsonl and relative.endswith(".jsonl"):
            # Active audits can grow during download. Copy only the byte range
            # observed by inventory, rather than following their moving EOF.
            with sftp.open(posixpath.join(remote, relative), "rb") as source, temporary.open("wb") as target:
                remaining = size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(f"Remote audit shrank during snapshot: {relative}")
                    target.write(chunk)
                    remaining -= len(chunk)
        else:
            sftp.get(posixpath.join(remote, relative), str(temporary))
        if temporary.stat().st_size != size:
            raise RuntimeError(f"Incomplete download: {relative}")
        if snapshot_jsonl and relative.endswith(".jsonl"):
            payload = temporary.read_bytes()
            if payload and not payload.endswith(b"\n"):
                temporary.write_bytes(payload[:payload.rfind(b"\n") + 1])
        temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote-root", default="/root/agi")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=int, default=120)
    parser.add_argument("--output-name", default="phase4_preference_dpo_v1")
    parser.add_argument("--run-dir", default="phase4/runs/preference_v1")
    parser.add_argument("--collection-dir")
    parser.add_argument("--collection-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    previous = {}
    output_name = args.output_name
    remote_output = posixpath.join(args.remote_root, "outputs", output_name)
    local_output = args.local_root / "outputs" / output_name
    local_runs = args.local_root / args.run_dir
    local_runs.mkdir(parents=True, exist_ok=True)

    while time.monotonic() - started < args.timeout_minutes * 60:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            client.connect(args.host, port=args.port, username="root",
                           key_filename=str(args.identity), timeout=25)
            sftp = client.open_sftp()
            remote_runs = posixpath.join(args.remote_root, args.run_dir)
            try:
                for entry in sftp.listdir_attr(remote_runs):
                    if stat.S_ISREG(entry.st_mode) and entry.filename.endswith(".log"):
                        sftp.get(posixpath.join(remote_runs, entry.filename), str(local_runs / entry.filename))
            except FileNotFoundError:
                pass
            if args.collection_dir:
                remote_collection = posixpath.join(args.remote_root, args.collection_dir)
                try:
                    collection_files = inventory(sftp, remote_collection)
                    copy_tree(sftp, remote_collection, args.local_root / args.collection_dir, collection_files, snapshot_jsonl=True)
                    if args.collection_only and "summary.json" in {item[0] for item in collection_files}:
                        print("COLLECTION_EVIDENCE_SAVED", flush=True)
                        return
                except FileNotFoundError:
                    pass
            try:
                entries = sftp.listdir(remote_output)
            except FileNotFoundError:
                entries = []
            for name in sorted(entries):
                if not name.startswith("checkpoint-") or not name[11:].isdigit():
                    continue
                destination = local_output / name
                if (destination / "LOCAL_COPY_COMPLETE.json").is_file():
                    continue
                remote_checkpoint = posixpath.join(remote_output, name)
                files = inventory(sftp, remote_checkpoint)
                required = {"trainer_state.json", "optimizer.pt", "scheduler.pt"}
                if required.issubset({item[0] for item in files}) and previous.get(name) == files:
                    print(f"COPYING {name}", flush=True)
                    copy_tree(sftp, remote_checkpoint, destination, files)
                    (destination / "LOCAL_COPY_COMPLETE.json").write_text(
                        json.dumps({"remote": remote_checkpoint, "files": files}, indent=2),
                        encoding="utf-8",
                    )
                    print(f"SAVED {name}", flush=True)
                previous[name] = files
            if "train_report.json" in entries:
                files = inventory(sftp, remote_output)
                final_files = [item for item in files if not item[0].startswith("checkpoint-")]
                copy_tree(sftp, remote_output, local_output, final_files)
                print("FINAL_EVIDENCE_SAVED", flush=True)
                return
        except Exception as error:
            print(f"BACKUP_RETRY {type(error).__name__}: {error}", flush=True)
        finally:
            client.close()
        time.sleep(args.poll_seconds)
    raise SystemExit("Backup deadline reached; inspect local and remote evidence")


if __name__ == "__main__":
    main()
