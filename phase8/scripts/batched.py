"""Batched, resumable vLLM generation shared by the Phase 8 v2 collectors.

Amendment v2 (2026-09-30): requests are submitted in fixed-size ordered chunks
instead of one at a time. Prompts, per-request seeds, temperature, top-p/top-k,
and the 768-token cap are unchanged. Each completion is still appended durably
in task order, so an interrupted run resumes from the last saved row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

BATCH_SIZE = 64


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_durable(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def open_audit(audit: Path, settings: dict, keys: list, key_of: Callable[[dict], object]) -> list[dict]:
    """Return completed rows from an ordered-prefix audit, or create a new audit."""
    if audit.exists():
        records = read_jsonl(audit)
        if not records or records[0] != {"type": "metadata", "settings": settings}:
            raise RuntimeError(f"Existing audit settings differ: {audit}")
        rows = []
        for index, entry in enumerate(records[1:]):
            if (entry.get("type") != "completion" or entry.get("index") != index or
                    key_of(entry["row"]) != keys[index]):
                raise RuntimeError(f"Existing audit is not an ordered prefix: {audit}")
            rows.append(entry["row"])
        return rows
    audit.parent.mkdir(parents=True, exist_ok=True)
    append_durable(audit, {"type": "metadata", "settings": settings})
    return []


def generate_ordered(llm, requests: list[tuple[str, object]], start: int, audit: Path,
                     make_row: Callable[[int, object], dict], label: str,
                     lora_request=None, stop: Callable[[list[dict]], bool] | None = None,
                     done: list[dict] | None = None) -> list[dict]:
    """Generate requests[start:] in ordered chunks and append each row durably.

    ``stop`` is checked after each saved row; generation beyond the stop point in
    the same chunk is discarded and reported by the caller.
    """
    done = done if done is not None else []
    for chunk_start in range(start, len(requests), BATCH_SIZE):
        chunk = requests[chunk_start:chunk_start + BATCH_SIZE]
        results = llm.generate([prompt for prompt, _ in chunk], [params for _, params in chunk],
                               lora_request=lora_request, use_tqdm=False)
        for offset, result in enumerate(results):
            index = chunk_start + offset
            row = make_row(index, result.outputs[0])
            append_durable(audit, {"type": "completion", "index": index, "row": row})
            done.append(row)
            if stop and stop(done):
                print(f"{label}: stop rule met at {len(done)}", flush=True)
                return done
        print(f"{label}: {len(done)}/{len(requests)}", flush=True)
    return done
