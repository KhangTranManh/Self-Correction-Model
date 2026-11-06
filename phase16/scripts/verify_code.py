"""Run Phase 16 Python candidates in a minimal, networkless Colab sandbox."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from score_boundary import score


FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
RESULT_PREFIX = "__P16_RESULT__="
TIMEOUT_SECONDS = 8


def extract_code(text: str) -> str:
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


def _failure(task: dict, status: str) -> dict:
    return score({
        "task_id": task["task_id"],
        "variant": task["variant"],
        "execution_status": status,
    })


def verify(task: dict, completion: str) -> dict:
    """Score one completion; expected values never enter the sandbox."""
    code = extract_code(completion)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _failure(task, "syntax_error")
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == task["function_name"] for node in tree.body):
        return _failure(task, "runtime_error")

    inputs = {
        "function_name": task["function_name"],
        "cases": [{"case_id": case["case_id"], "args": case["args"]}
                  for case in task["cases"]],
    }
    # The sandbox sees only the candidate source and test inputs. The trusted
    # oracle answers and task prompt remain in the parent process.
    child = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
record = json.load(sys.stdin)
namespace = {"__name__": "__candidate__"}
with open("/work/candidate.py", encoding="utf-8") as source:
    exec(compile(source.read(), "candidate.py", "exec"), namespace)
function = namespace[record["function_name"]]
answers = [function(*case["args"]) for case in record["cases"]]
print("__P16_RESULT__=" + json.dumps(answers, ensure_ascii=False))
'''
    with tempfile.TemporaryDirectory(prefix="p16_candidate_") as directory:
        work = Path(directory)
        os.chmod(work, 0o755)
        candidate_path = work / "candidate.py"
        candidate_path.write_text(code + "\n", encoding="utf-8")
        os.chmod(candidate_path, 0o644)
        command = [
            "bwrap", "--die-with-parent", "--unshare-user", "--uid", "65534",
            "--gid", "65534", "--unshare-net", "--unshare-pid",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--tmpfs", "/tmp",
            "--ro-bind", str(work), "/work", "--dev", "/dev",
            "--chdir", "/tmp", "/usr/bin/python3.13", "-I", "-S", "-c", child,
        ]
        try:
            run = subprocess.run(
                command, input=json.dumps(inputs), capture_output=True,
                text=True, timeout=TIMEOUT_SECONDS,
                env={"PATH": "/usr/bin", "HOME": "/tmp",
                     "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return _failure(task, "timeout")
    if run.returncode != 0:
        return _failure(task, "runtime_error")
    marked = [line[len(RESULT_PREFIX):] for line in run.stdout.splitlines()
              if line.startswith(RESULT_PREFIX)]
    if len(marked) != 1:
        return _failure(task, "runtime_error")
    try:
        answers = json.loads(marked[0])
    except json.JSONDecodeError:
        return _failure(task, "runtime_error")
    if not isinstance(answers, list) or len(answers) != len(task["cases"]):
        return _failure(task, "runtime_error")
    cases = [
        {key: value for key, value in case.items() if key != "args"}
        | {"actual": answer}
        for case, answer in zip(task["cases"], answers)
    ]
    return score({
        "task_id": task["task_id"], "variant": task["variant"],
        "execution_status": "ok", "cases": cases,
    })
