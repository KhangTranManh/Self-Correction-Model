"""Score only completed episodes from a live P15-A3 append-only snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescore_pilot_a3 import score


class CompletedSnapshot:
    def __init__(self, path: Path):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        done = {row["episode_id"] for row in rows if row["type"] == "episode_complete"}
        kept = [row for row in rows if row["type"] == "metadata" or row.get("episode_id") in done]
        self.data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n").encode("utf-8")

    def read_text(self, encoding="utf-8") -> str:
        return self.data.decode(encoding)

    def read_bytes(self) -> bytes:
        return self.data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(CompletedSnapshot(args.audit)), indent=2))
