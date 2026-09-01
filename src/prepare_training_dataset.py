"""Create the length-safe training view of the verified Phase-1 dataset.

The raw dataset remains an immutable audit artifact.  This preparation step
removes only an optional leading ``<thinking>...</thinking>`` trace from the
final assistant message.  If a row is still too long, only its loss-masked
failed-assistant context is replaced with a short marker.  The problem,
verifier feedback, and complete verifier-approved correction are always
preserved, and every resulting chat must fit the configured training length.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from src.config import ROOT_DIR, load_config

_THINKING_BLOCK = re.compile(r"^\s*<thinking>\s*.*?\s*</thinking>\s*", re.DOTALL)
_COMPACTED_FAILED_ATTEMPT = (
    "[Failed assistant attempt omitted for context length. "
    "Use the verifier feedback below to produce the corrected answer.]"
)


def _token_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Return chat-template token IDs across Transformers 4.x and 5.x."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return encoded


def main() -> None:
    # Dataset preparation does not call the correction API and must not require
    # access to its credentials.
    cfg = load_config(require_deepseek=False)
    source = cfg.path("sft_dataset_out")
    destination = cfg.path("sft_training_dataset_out")
    max_length = int(cfg.training["max_seq_length"])
    model_source = os.environ.get("AGI_MODEL_PATH") or cfg.small_model["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=bool(os.environ.get("AGI_MODEL_PATH")),
    )

    rows: list[dict] = []
    removed = 0
    source_ids: set[str] = set()
    maximum_tokens = 0
    maximum_tokens_before_compaction = 0
    failed_attempts_compacted = 0

    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        problem_id = str(row.get("problem_id", ""))
        if not problem_id or problem_id in source_ids:
            raise RuntimeError(f"Duplicate/missing problem_id at source line {line_number}")
        source_ids.add(problem_id)

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 5:
            raise RuntimeError(f"Invalid messages at source line {line_number}")
        if messages[-1].get("role") != "assistant":
            raise RuntimeError(f"Final message is not assistant at source line {line_number}")

        cleaned_messages = [dict(message) for message in messages]
        original = str(cleaned_messages[-1].get("content", ""))
        cleaned = _THINKING_BLOCK.sub("", original)
        if cleaned != original:
            removed += 1
        if not cleaned.strip():
            raise RuntimeError(f"Empty correction after cleanup at source line {line_number}")
        cleaned_messages[-1]["content"] = cleaned

        token_count = len(_token_ids(tokenizer, cleaned_messages))
        maximum_tokens_before_compaction = max(maximum_tokens_before_compaction, token_count)

        if token_count > max_length:
            failed_index = next(
                (
                    index
                    for index, message in enumerate(cleaned_messages[:-1])
                    if message.get("role") == "assistant"
                ),
                None,
            )
            if failed_index is None:
                raise RuntimeError(
                    f"Training row {problem_id} exceeds max_length but has no failed attempt to compact"
                )
            cleaned_messages[failed_index]["content"] = _COMPACTED_FAILED_ATTEMPT
            failed_attempts_compacted += 1
            token_count = len(_token_ids(tokenizer, cleaned_messages))

        maximum_tokens = max(maximum_tokens, token_count)
        if token_count > max_length:
            raise RuntimeError(
                f"Training row {problem_id} has {token_count} tokens, above max_length={max_length}"
            )

        rows.append({"problem_id": problem_id, "messages": cleaned_messages})

    if not rows:
        raise RuntimeError("Verified SFT source dataset is empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "rows": len(rows),
        "unique_problem_ids": len(source_ids),
        "thinking_blocks_removed": removed,
        "failed_attempts_compacted": failed_attempts_compacted,
        "max_length": max_length,
        "maximum_row_tokens_before_compaction": maximum_tokens_before_compaction,
        "maximum_row_tokens": maximum_tokens,
        "all_rows_fit": True,
        "all_verified_corrections_preserved": True,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    report_path = ROOT_DIR / "outputs" / "phase1_sft_training_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
