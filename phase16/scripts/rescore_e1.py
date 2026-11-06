"""Re-execute saved P16-E1 completions in the isolated verifier, no model load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path("/content/phase16")
OUT = ROOT / "gpu_smoke_v1"
EXPECTED_AUDIT = "11b5f13417fe5af11b353512f1df227c1ff30d7d725417a2c32979c29433e420"
EXPECTED_TASKS = "cabac7e679b2de020e9de2cefa9c8d01d0f4b3c4f6f364600508aaafb7154cde"
sys.path.insert(0, str(ROOT))
from verify_code import verify  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    audit_path = OUT / "audit.jsonl"
    task_path = ROOT / "tasks.json"
    if digest(audit_path) != EXPECTED_AUDIT or digest(task_path) != EXPECTED_TASKS:
        raise RuntimeError("Frozen E1 input hash mismatch")
    task_map = {
        task["task_id"]: task for task in
        json.loads(task_path.read_text(encoding="utf-8"))["tasks"]
    }
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line]
    completions = [row for row in rows if row["type"] == "completion"]
    if len(completions) != 9 or len({row["task_id"] for row in completions}) != 9:
        raise RuntimeError("Expected nine unique saved completions")
    labels = []
    disagreements = []
    for row in completions:
        rescored = verify(task_map[row["task_id"]], row["output"])
        labels.append(rescored["label"])
        if rescored != row["score"]:
            disagreements.append(row["task_id"])
    result = {
        "schema_version": "phase16_e1_saved_output_rescore_v1",
        "source_audit_sha256": EXPECTED_AUDIT,
        "tasks_sha256": EXPECTED_TASKS,
        "n": len(completions),
        "labels": {label: labels.count(label) for label in sorted(set(labels))},
        "disagreements": disagreements,
        "model_generation": False,
    }
    output = OUT / "rescore.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite existing rescore")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
