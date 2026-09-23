"""Mirror a growing Phase 5 SSH audit locally without storing a password."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

import paramiko


ROOT = Path(__file__).resolve().parents[3]
REMOTE_ROOT = "/root/AGI_phase5/phase5"


def validate(data: bytes, candidate_ids: list[str]) -> int:
    lines = data.decode("utf-8").splitlines()
    if not lines:
        raise ValueError("Empty remote audit")
    metadata = json.loads(lines[0])
    if metadata.get("type") != "metadata":
        raise ValueError("Missing audit metadata")
    if metadata.get("settings", {}).get("candidates_sha256") != (
        "3dc103e168ea149f51d16e5d22fad66e97b1af9e956fba1d8914b714120e954a"
    ):
        raise ValueError("Candidate manifest mismatch")
    for index, line in enumerate(lines[1:]):
        entry = json.loads(line)
        if index >= len(candidate_ids) or entry.get("index") != index:
            raise ValueError("Audit order mismatch")
        if entry.get("type") != "completion" or entry.get("row", {}).get("problem_id") != candidate_ids[index]:
            raise ValueError("Audit source mismatch")
    if not data.endswith(b"\n"):
        raise ValueError("Remote audit ended during append; retry later")
    return len(lines) - 1


def copy_audit(sftp, remote: str, local: Path, candidate_ids: list[str]):
    with sftp.open(remote, "rb") as source:
        data = source.read()
    count = validate(data, candidate_ids)
    if local.exists():
        previous = local.read_bytes()
        if not data.startswith(previous):
            raise ValueError("Remote audit does not extend local backup")
    local.parent.mkdir(parents=True, exist_ok=True)
    pending = local.with_suffix(local.suffix + ".pending")
    pending.write_bytes(data)
    os.replace(pending, local)
    return count, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    ids = [json.loads(line)["id"] for line in (ROOT / "phase5/data/candidates_v1/candidate_problems.jsonl").read_text(encoding="utf-8").splitlines()]
    destination = ROOT / "outputs/phase5_remote_v100"
    secret = getpass.getpass("SSH password (kept in memory only): ")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=secret,
                   look_for_keys=False, allow_agent=False, timeout=20)
    secret = None
    try:
        sftp = client.open_sftp()
        while True:
            try:
                count, digest = copy_audit(
                    sftp, REMOTE_ROOT + "/data/initials_v1/initial_rollouts.audit.jsonl",
                    destination / "initial_rollouts.audit.jsonl", ids,
                )
                print(f"backed_up={count} sha256={digest}", flush=True)
                remote_summary = REMOTE_ROOT + "/data/initials_v1/summary.json"
                try:
                    sftp.stat(remote_summary)
                except FileNotFoundError:
                    pass
                else:
                    pending_summary = destination / "summary.json.pending"
                    sftp.get(remote_summary, str(pending_summary))
                    json.loads(pending_summary.read_text(encoding="utf-8"))
                    os.replace(pending_summary, destination / "summary.json")
                    print("summary copied; collection finished", flush=True)
                    break
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"backup retry: {type(error).__name__}: {error}", flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        client.close()


if __name__ == "__main__":
    main()
