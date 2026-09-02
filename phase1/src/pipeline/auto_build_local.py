"""Complete the resumable Phase-1 SFT build on a CPU/API-only computer.

The supervisor repeatedly runs ``src.build_dataset`` until all attempt IDs are
durably committed in the ledger.  It then validates the SFT JSONL and exits;
it intentionally does not load a model or start GPU training.
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
_LOCK = _OUTPUTS / "auto_build_local.lock"
_STATUS = _OUTPUTS / "auto_build_local.status"
_LOG = _OUTPUTS / "auto_build_local.log"
_BUILDER_LOG = _OUTPUTS / "build_dataset_local.log"
_VALIDATION = _OUTPUTS / "phase1_sft_validation.json"
_RETRY_COOLDOWN_SECONDS = 60


def _log(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with _LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _state() -> tuple[set[str], set[str]]:
    cfg = load_config()
    attempt_ids = {
        json.loads(line)["problem_id"]
        for line in _nonempty_lines(cfg.path("attempts_out"))
    }
    ledger = cfg.path("sft_dataset_out").parent / "build_dataset_seen_ids.txt"
    seen_ids = set(_nonempty_lines(ledger))
    if not seen_ids <= attempt_ids:
        raise RuntimeError("Ledger contains IDs absent from attempts.jsonl")
    return attempt_ids, seen_ids


def _run_builder(pass_number: int) -> int:
    _log(f"Starting resumable local builder pass {pass_number}")
    child_env = os.environ.copy()
    # Windows otherwise gives a detached Python process the legacy cp1252
    # stdout encoding. API/model diagnostics can contain Vietnamese or other
    # Unicode characters, and merely logging them must never crash the build.
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    with _BUILDER_LOG.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n[auto_build_local] BEGIN PASS {pass_number}\n")
        log_f.flush()
        result = subprocess.run(
            [sys.executable, "-m", "src.build_dataset"],
            cwd=ROOT_DIR,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=child_env,
            check=False,
        )
        log_f.write(f"[auto_build_local] END PASS {pass_number} exit={result.returncode}\n")
    return result.returncode


def _validate(attempt_ids: set[str], seen_ids: set[str]) -> dict:
    cfg = load_config()
    dataset = cfg.path("sft_dataset_out")
    rows = [json.loads(line) for line in _nonempty_lines(dataset)]
    sft_ids = [row.get("problem_id") for row in rows]

    if seen_ids != attempt_ids:
        raise RuntimeError(f"Incomplete ledger: {len(seen_ids)}/{len(attempt_ids)}")
    if not rows:
        raise RuntimeError("SFT dataset is empty")
    if len(sft_ids) != len(set(sft_ids)):
        raise RuntimeError("SFT dataset contains duplicate problem IDs")
    if not set(sft_ids) <= seen_ids:
        raise RuntimeError("SFT dataset contains an ID absent from the ledger")

    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 5:
            raise RuntimeError(f"Invalid messages in SFT row {index}")
        if messages[-1].get("role") != "assistant":
            raise RuntimeError(f"SFT row {index} does not end in an assistant correction")
        if not str(messages[-1].get("content", "")).strip():
            raise RuntimeError(f"SFT row {index} has an empty final correction")

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_ids": len(attempt_ids),
        "seen_ids": len(seen_ids),
        "sft_rows": len(rows),
        "unique_sft_ids": len(set(sft_ids)),
        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "final_correction_only_target": True,
        "training_started": False,
    }
    temporary = _VALIDATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(_VALIDATION)
    return report


def main() -> None:
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Local builder supervisor already active: {_LOCK}") from exc

    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.close(lock_fd)
        _STATUS.write_text("RUNNING\n", encoding="utf-8")

        pass_number = 1
        while True:
            attempts, seen_before = _state()
            if seen_before == attempts:
                break

            returncode = _run_builder(pass_number)
            attempts, seen_after = _state()
            _log(
                f"Pass {pass_number} finished: exit={returncode}, "
                f"seen={len(seen_after)}/{len(attempts)}, "
                f"new={len(seen_after - seen_before)}"
            )
            pass_number += 1
            if seen_after != attempts:
                _log(f"Unresolved IDs remain; retrying after {_RETRY_COOLDOWN_SECONDS}s")
                time.sleep(_RETRY_COOLDOWN_SECONDS)

        attempts, seen = _state()
        report = _validate(attempts, seen)
        _STATUS.write_text("0\n", encoding="utf-8")
        _log(f"COMPLETE: rows={report['sft_rows']}, sha256={report['sha256']}")
    except Exception:
        _STATUS.write_text("1\n", encoding="utf-8")
        raise
    finally:
        _LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
