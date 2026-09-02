"""Wait for an idle GPU, then launch Phase-1 QLoRA training exactly once."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT_DIR

_OUTPUTS = ROOT_DIR / "outputs"
_LOCK = _OUTPUTS / "wait_for_gpu_and_train.lock"
_GUARD_LOG = _OUTPUTS / "gpu_wait.log"
_GUARD_STATUS = _OUTPUTS / "gpu_wait.status"
_TRAIN_LOG = _OUTPUTS / "train_sft.log"
_TRAIN_STATUS = _OUTPUTS / "train_sft.status"
_MIN_FREE_MIB = int(os.environ.get("AGI_MIN_FREE_VRAM_MIB", "22000"))
_POLL_SECONDS = int(os.environ.get("AGI_GPU_POLL_SECONDS", "30"))
_CONSECUTIVE_FREE_CHECKS = int(os.environ.get("AGI_GPU_FREE_CHECKS", "3"))


def _log(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with _GUARD_LOG.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def _free_vram_mib() -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    first_gpu = result.stdout.strip().splitlines()[0]
    free_text, total_text = (part.strip() for part in first_gpu.split(",", 1))
    return int(free_text), int(total_text)


def _training_running() -> bool:
    if os.name == "nt":
        # Windows has no pgrep. Query only active CUDA compute applications and
        # regard a Python process as an existing training/model job. Desktop
        # WDDM applications (Chrome, DWM, etc.) are intentionally ignored.
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return any("python" in line.lower() for line in result.stdout.splitlines())

    result = subprocess.run(
        ["pgrep", "-f", r"python3? -m src\.train_sft"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"GPU training guard already active: {_LOCK}") from exc

    try:
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.close(lock_fd)
        _GUARD_STATUS.write_text("WAITING\n", encoding="utf-8")

        if _TRAIN_STATUS.exists() and _TRAIN_STATUS.read_text(encoding="utf-8").strip() == "0":
            _log("Training is already complete; refusing to start it again")
            _GUARD_STATUS.write_text("0\n", encoding="utf-8")
            return
        if _training_running():
            raise RuntimeError("A train_sft process is already running")

        consecutive = 0
        while consecutive < _CONSECUTIVE_FREE_CHECKS:
            free_mib, total_mib = _free_vram_mib()
            if free_mib >= _MIN_FREE_MIB:
                consecutive += 1
            else:
                consecutive = 0
            _log(
                f"GPU free={free_mib}/{total_mib} MiB; "
                f"required={_MIN_FREE_MIB} MiB; stable_checks={consecutive}/{_CONSECUTIVE_FREE_CHECKS}"
            )
            if consecutive < _CONSECUTIVE_FREE_CHECKS:
                time.sleep(_POLL_SECONDS)

        # Recheck for a race immediately before starting the expensive process.
        free_mib, total_mib = _free_vram_mib()
        if free_mib < _MIN_FREE_MIB or _training_running():
            _log("GPU state changed before launch; returning to the wait loop")
            consecutive = 0
            while consecutive < _CONSECUTIVE_FREE_CHECKS:
                time.sleep(_POLL_SECONDS)
                free_mib, total_mib = _free_vram_mib()
                consecutive = consecutive + 1 if free_mib >= _MIN_FREE_MIB else 0
                _log(
                    f"GPU free={free_mib}/{total_mib} MiB; "
                    f"required={_MIN_FREE_MIB} MiB; stable_checks={consecutive}/{_CONSECUTIVE_FREE_CHECKS}"
                )

        _log("GPU availability gate passed; starting QLoRA training")
        _GUARD_STATUS.write_text("TRAINING\n", encoding="utf-8")
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
        _GUARD_STATUS.write_text(f"{result.returncode}\n", encoding="utf-8")
        _log(f"Training exited with status {result.returncode}")
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        _LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
