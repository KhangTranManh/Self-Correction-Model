"""Mirror Phase 8 remote audits to local storage without embedding SSH secrets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import paramiko


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/AGI_phase8"
PATHS = ["logs/phase8_initial.log", "logs/phase8_pipeline.log",
         "phase8/data/distractors_v1/assignments.jsonl",
         "phase8/data/distractors_v1/report.json",
         "outputs/phase8_phase7_replay_v1/audit.jsonl",
         "outputs/phase8_phase7_replay_v1/answers.jsonl",
         "outputs/phase8_phase7_replay_v1/summary.json",
         "outputs/phase8_analysis_v1/verdicts.jsonl",
         "outputs/phase8_analysis_v1/report.json"]
for stage in ("initial", "sample_repeat", "greedy_same_prompt"):
    PATHS.extend(f"outputs/phase8_first_pass_v1/{stage}/{name}"
                 for name in ("audit.jsonl", "answers.jsonl", "summary.json"))
for checkpoint in ("original_solver", "warmstart_v2", "correction_sft_v3"):
    PATHS.extend(f"outputs/phase8_three_arms_v1/{checkpoint}/{name}"
                 for name in ("audit.jsonl", "answers.jsonl", "summary.json"))
    PATHS.extend(f"outputs/phase8_probe_scores_v1/{checkpoint}/{name}"
                 for name in ("scores.audit.jsonl", "scores.jsonl", "summary.json"))
    PATHS.extend((
        f"outputs/phase8_probe_v1/selection/{checkpoint}_probe.joblib",
        f"outputs/phase8_probe_v1/selection/{checkpoint}_probe_selection.json",
    ))
    for split in ("train", "development"):
        PATHS.extend((
            f"outputs/phase8_probe_v1/activations/{checkpoint}_{split}.npz",
            f"outputs/phase8_probe_v1/activations/{checkpoint}_{split}_summary.json",
        ))


def sync_once(sftp: paramiko.SFTPClient) -> int:
    copied = 0
    for relative in PATHS:
        remote = f"{REMOTE}/{relative}"
        try:
            info = sftp.stat(remote)
        except FileNotFoundError:
            continue
        local = ROOT / "outputs/phase8_remote_3090" / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size == info.st_size:
            continue
        temporary = local.with_name(local.name + ".part")
        remaining = info.st_size
        with sftp.open(remote, "rb") as source, temporary.open("wb") as target:
            while remaining:
                block = source.read(min(1024 * 1024, remaining))
                if not block:
                    break
                target.write(block)
                remaining -= len(block)
        if remaining:
            temporary.unlink(missing_ok=True)
            continue
        if local.suffix == ".jsonl":
            data = temporary.read_bytes()
            complete = data.rfind(b"\n") + 1
            if complete == 0:
                temporary.unlink(missing_ok=True)
                continue
            if complete != len(data):
                temporary.write_bytes(data[:complete])
        temporary.replace(local)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    password = os.environ.get("PHASE8_SSH_PASSWORD")
    if not password:
        raise RuntimeError("Set PHASE8_SSH_PASSWORD in the process environment")
    while True:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect("209.121.195.118", port=12017, username="root",
                           password=password, timeout=20)
            sftp = client.open_sftp()
            print(f"copied={sync_once(sftp)}", flush=True)
            sftp.close()
        except Exception as exc:
            if args.once:
                raise
            print(f"sync_retry={type(exc).__name__}: {exc}", flush=True)
        finally:
            client.close()
        if args.once or (ROOT / "outputs/phase8_remote_3090/outputs/phase8_analysis_v1/report.json").exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
