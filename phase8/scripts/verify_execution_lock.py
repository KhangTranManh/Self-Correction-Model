"""Verify Phase 8 source, protocol, and analysis files against a frozen lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "phase8/data/execution_lock_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Phase 8 execution lock mismatch: {relative}")
    print(json.dumps({"status": "verified", "files": len(lock["files"]),
                      "lock_sha256": sha256(LOCK)}))


if __name__ == "__main__":
    main()
