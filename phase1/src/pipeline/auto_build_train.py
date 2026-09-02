"""Finish the resumable SFT build, validate it, then train immediately.

This process is intended for a rented GPU host where dataset construction is
API-bound.  It never runs two builders at once.  If a provider outage leaves
IDs unresolved, it starts another resumable pass after a cooldown.  Training
starts only when every attempt ID is durably present in the ledger and the SFT
JSONL passes structural checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT_DIR, load_config

_OUTPUTS = ROOT_DIR / "outputs"
_LOCK_PATH = _OUTPUTS / "auto_build_train.lock"
_WATCH_LOG = _OUTPUTS / "auto_build_train.log"
_BUILD_LOG = _OUTPUTS / "build_dataset.log"
_BUILD_STATUS = _OUTPUTS / "build_dataset.status"
_TRAIN_LOG = _OUTPUTS / "train_sft.log"
_TRAIN_STATUS = _OUTPUTS / "train_sft.status"
_VALIDATION_OUT = _OUTPUTS / "phase1_sft_validation.json"
_POLL_SECONDS = 30
_RETRY_COOLDOWN_SECONDS = 60


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with _WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _builder_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", r"^python3 -m src\.build_dataset$"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _dataset_state() -> tuple[set[str], set[str]]:
    cfg = load_config()
    attempts = {
        json.loads(line)["problem_id"]
        for line in _nonempty_lines(cfg.path("attempts_out"))
    }
    ledger_path = cfg.path("sft_dataset_out").parent / "build_dataset_seen_ids.txt"
    return attempts, set(_nonempty_lines(ledger_path))


def _run_builder_pass(pass_number: int) -> int:
    if _builder_running():
        raise RuntimeError("Refusing to launch a second build_dataset process")

    _log(f"Starting resumable builder pass {pass_number}")
    _BUILD_STATUS.write_text("RUNNING\n", encoding="utf-8")
    with _BUILD_LOG.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n[auto_build_train] BEGIN PASS {pass_number}\n")
        log_f.flush()
        result = subprocess.run(
            [sys.executable, "-m", "src.build_dataset"],
            cwd=ROOT_DIR,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_f.write(f"[auto_build_train] END PASS {pass_number} exit={result.returncode}\n")
    _BUILD_STATUS.write_text(f"{result.returncode}\n", encoding="utf-8")
    return result.returncode


def _validate_sft(attempt_ids: set[str], seen_ids: set[str]) -> dict:
    cfg = load_config()
    rows = [json.loads(line) for line in _nonempty_lines(cfg.path("sft_dataset_out"))]
    sft_ids = [row.get("problem_id") for row in rows]

    if seen_ids != attempt_ids:
        missing = len(attempt_ids - seen_ids)
        extra = len(seen_ids - attempt_ids)
        raise RuntimeError(f"Ledger is incomplete: missing={missing}, extra={extra}")
    if not rows:
        raise RuntimeError("SFT dataset is empty")
    if len(sft_ids) != len(set(sft_ids)):
        raise RuntimeError("SFT dataset contains duplicate problem_id values")

    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 5:
            raise RuntimeError(f"SFT row {index} has an invalid messages sequence")
        if messages[-1].get("role") != "assistant":
            raise RuntimeError(f"SFT row {index} does not end with an assistant correction")
        if not str(messages[-1].get("content", "")).strip():
            raise RuntimeError(f"SFT row {index} has an empty final correction")

    dataset_path = cfg.path("sft_dataset_out")
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_ids": len(attempt_ids),
        "seen_ids": len(seen_ids),
        "sft_rows": len(rows),
        "unique_sft_ids": len(set(sft_ids)),
        "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "final_correction_only_target": True,
    }
    tmp = _VALIDATION_OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(_VALIDATION_OUT)
    return report


def _run_training() -> int:
    _log("Dataset validation passed; starting QLoRA training")
    _TRAIN_STATUS.write_text("RUNNING\n", encoding="utf-8")
    with _TRAIN_LOG.open("w", encoding="utf-8") as log_f:
        result = subprocess.run(
            [sys.executable, "-m", "src.train_sft"],
            cwd=ROOT_DIR,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    _TRAIN_STATUS.write_text(f"{result.returncode}\n", encoding="utf-8")
    _log(f"Training exited with status {result.returncode}")
    return result.returncode


def main() -> None:
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Orchestrator lock already exists: {_LOCK_PATH}") from exc

    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.close(lock_fd)

        while _builder_running():
            attempts, seen = _dataset_state()
            _log(f"Waiting for active builder: seen={len(seen)}/{len(attempts)}")
            time.sleep(_POLL_SECONDS)

        pass_number = 1
        while True:
            attempts, seen_before = _dataset_state()
            if seen_before == attempts:
                break

            returncode = _run_builder_pass(pass_number)
            attempts, seen_after = _dataset_state()
            _log(
                f"Builder pass {pass_number} finished: exit={returncode}, "
                f"seen={len(seen_after)}/{len(attempts)}, "
                f"new={len(seen_after - seen_before)}"
            )
            pass_number += 1

            if seen_after != attempts:
                _log(f"Unresolved IDs remain; cooling down {_RETRY_COOLDOWN_SECONDS}s")
                time.sleep(_RETRY_COOLDOWN_SECONDS)

        attempts, seen = _dataset_state()
        report = _validate_sft(attempts, seen)
        _log(f"SFT validation passed: rows={report['sft_rows']}, sha256={report['sha256']}")
        if _run_training() != 0:
            raise RuntimeError(f"Training failed; inspect {_TRAIN_LOG}")
    finally:
        _LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
