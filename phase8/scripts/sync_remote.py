"""Mirror Phase 8 remote audits and results to local storage without embedding SSH secrets.

The SSH host, port, user, and password are read from a local JSON file passed
with --config (kept outside the repository), never from this source file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import paramiko


ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/AGI_phase8"
LOCAL = ROOT / "outputs/phase8_remote_v100"
PATHS = ["logs/pipeline.log", "logs/materialize_v2.log", "logs/phase8_analysis_stdout.json",
         "logs/PIPELINE_DONE", "logs/PIPELINE_FAILED",
         "phase8/data/execution_lock_v2.json",
         "phase8/data/distractors_v2/assignments.jsonl",
         "phase8/data/distractors_v2/report.json",
         "outputs/phase7_initials_regen_v2/initial_rollouts.audit.jsonl",
         "outputs/phase7_initials_regen_v2/initial_rollouts.jsonl",
         "outputs/phase7_initials_regen_v2/summary.json",
         "outputs/phase8_analysis_v2/verdicts.jsonl",
         "outputs/phase8_analysis_v2/report.json"]
for stage in ("initial", "sample_repeat", "greedy_same_prompt"):
    PATHS.extend(f"outputs/phase8_first_pass_v2/{stage}/{name}"
                 for name in ("audit.jsonl", "answers.jsonl", "summary.json"))
for checkpoint in ("original_solver", "warmstart_v2", "correction_sft_v3"):
    PATHS.extend(f"outputs/phase8_three_arms_v2/{checkpoint}/{name}"
                 for name in ("audit.jsonl", "answers.jsonl", "summary.json"))
    PATHS.extend(f"outputs/phase8_probe_scores_v2/{checkpoint}/{name}"
                 for name in ("scores.audit.jsonl", "scores.jsonl", "summary.json"))


def sync_once(sftp: paramiko.SFTPClient) -> int:
    copied = 0
    for relative in PATHS:
        remote = f"{REMOTE}/{relative}"
        try:
            info = sftp.stat(remote)
        except FileNotFoundError:
            continue
        local = LOCAL / relative
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
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON with host, port, user, password (outside the repo)")
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
        # The done marker can arrive before the last outputs; sync once more after it.
        done = (LOCAL / "logs/PIPELINE_DONE").exists() or (LOCAL / "logs/PIPELINE_FAILED").exists()
        if args.once or final_pass:
            break
        final_pass = done
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
