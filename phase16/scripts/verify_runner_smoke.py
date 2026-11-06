"""Verify the locked engineering task pack with trusted code inside bwrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path("/content/phase16")
sys.path.insert(0, str(ROOT))
from verify_code import verify  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trusted_code(task: dict, *, inclusive: bool) -> str:
    shape, variant = task["shape"], task["variant"]
    if shape == "integers":
        key, payload = "row", None
    elif shape == "pairs":
        key, payload = "row[0]", "row[1]"
    else:
        fields = {
            "dict_score": ("score", "tag"),
            "dict_day_amount": ("day", "amount"),
            "dict_key_weight": ("key", "weight"),
            "dict_price_name": ("price", "name"),
            "dict_rank_id": ("rank", "id"),
        }
        key_field, payload_field = fields[shape]
        key, payload = f"row[{key_field!r}]", f"row[{payload_field!r}]"
    right = "<=" if inclusive else "<"
    condition = f"lower <= {key} {right} upper"
    if variant == "count":
        expression = f"sum(1 for row in rows if {condition})"
    elif variant == "sum":
        expression = f"sum({payload} for row in rows if {condition})"
    else:
        expression = f"[{payload} for row in rows if {condition}]"
    return f"def {task['function_name']}(rows, lower, upper):\n    return {expression}\n"


def main() -> None:
    task_path = ROOT / "tasks.json"
    manifest = json.loads(task_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "phase16_engineering_smoke_tasks_v1":
        raise AssertionError("Unexpected engineering task pack")
    result = {
        "schema_version": "phase16_colab_runner_smoke_v1",
        "purpose": "trusted_fixture_mechanics_only",
        "model_generation": False,
        "tasks_sha256": sha256(task_path),
        "scorer_sha256": sha256(ROOT / "score_boundary.py"),
        "runner_sha256": sha256(ROOT / "verify_code.py"),
        "tasks": [],
    }
    for task in manifest["tasks"]:
        correct = verify(task, trusted_code(task, inclusive=True))
        exclusive = verify(task, trusted_code(task, inclusive=False))
        if correct["label"] != "correct" or correct["passed_cases"] != 10:
            raise AssertionError(f"Trusted correct fixture failed: {task['task_id']}: {correct}")
        if exclusive["label"] != "upper_boundary_signature" or exclusive["signature_pairs"] != 2:
            raise AssertionError(f"Trusted exclusive fixture failed: {task['task_id']}: {exclusive}")
        result["tasks"].append({
            "task_id": task["task_id"], "correct": correct,
            "exclusive_upper": exclusive,
        })
        print(f"{task['task_id']}: correct=10/10, exclusive=signature/2", flush=True)
    output = ROOT / "runner_smoke.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite existing runner smoke")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {len(result['tasks'])} tasks", flush=True)


if __name__ == "__main__":
    main()
