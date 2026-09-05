"""Deterministic executable verifier for APPS stdin and call-based tests.

Every test runs in a fresh subprocess with a wall timeout and Linux resource
limits.  On Linux the child also drops from root to the ``nobody`` user before
executing model-generated code.  This is research isolation, not a production
security sandbox; run it only on a disposable worker.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


_FENCES = (
    re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE),
    re.compile(r"```\s*\n(.*?)```", re.DOTALL),
)
_RESULT_PREFIX = "__APPS_RESULT__="

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def _verification_python() -> str:
    # The pilot venv lives below /root and is intentionally inaccessible after
    # the child drops privileges. APPS introductory solutions should use the
    # Python standard library, so execute them with the system interpreter.
    system_python = Path("/usr/bin/python3")
    return str(system_python) if system_python.exists() else sys.executable


def extract_code(text: str) -> str:
    for pattern in _FENCES:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return text.strip()


def _preexec(memory_limit_mb: int, timeout_seconds: int):
    if sys.platform.startswith("win"):
        return None

    def limit_and_drop_privileges() -> None:
        import resource

        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)

    return limit_and_drop_privileges


def _run(
    argv: list[str],
    *,
    cwd: Path,
    stdin: str,
    timeout_seconds: int,
    memory_limit_mb: int,
    max_output_bytes: int,
) -> tuple[bool, str, str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "HOME": str(cwd),
    }
    try:
        result = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",
            cwd=cwd,
            env=env,
            timeout=timeout_seconds,
            preexec_fn=_preexec(memory_limit_mb, timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"")
        stderr = (exc.stderr or b"")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return False, stdout[-max_output_bytes:], stderr[-max_output_bytes:], "timeout"
    stdout = result.stdout[-max_output_bytes:]
    stderr = result.stderr[-max_output_bytes:]
    return result.returncode == 0, stdout, stderr, f"exit_code_{result.returncode}"


def _normal_text(value: str) -> str:
    lines = [line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _stdout_matches(actual: str, expected: Any) -> bool:
    if isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        expected_values = ["\n".join(expected)]
    else:
        expected_values = [str(expected)]
    actual_normal = _normal_text(actual)
    for value in expected_values:
        expected_normal = _normal_text(value)
        if actual_normal == expected_normal:
            return True
        actual_tokens = actual_normal.split()
        expected_tokens = expected_normal.split()
        if actual_tokens == expected_tokens:
            return True
        if len(actual_tokens) == len(expected_tokens) and actual_tokens:
            try:
                if all(
                    math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
                    for a, b in zip(actual_tokens, expected_tokens)
                ):
                    return True
            except ValueError:
                pass
    return False


def _canonical(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    return value


def _call_matches(actual: Any, expected: Any) -> bool:
    actual = _canonical(actual)
    expected = _canonical(expected)
    if actual == expected:
        return True
    if isinstance(expected, list) and len(expected) == 1 and actual == expected[0]:
        return True
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        try:
            return all(
                math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
                for a, b in zip(actual, expected)
            )
        except (TypeError, ValueError):
            return False
    return False


_CALL_RUNNER = r'''
import importlib.util
import inspect
import json
import sys

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

PREFIX = "__APPS_RESULT__="

def clean(value):
    if inspect.isgenerator(value):
        value = list(value)
    if isinstance(value, tuple):
        return [clean(item) for item in value]
    if isinstance(value, set):
        return sorted((clean(item) for item in value), key=repr)
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    return value

payload = json.loads(sys.stdin.read())
spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target = module.Solution() if hasattr(module, "Solution") else module
method = getattr(target, payload["fn_name"])
inputs = payload["inputs"]
if not isinstance(inputs, list):
    inputs = [inputs]
result = clean(method(*inputs))
sys.__stdout__.write(PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
'''


def verify(
    candidate_text: str,
    input_output: dict[str, Any],
    *,
    per_test_timeout_seconds: int = 3,
    memory_limit_mb: int = 512,
    max_output_bytes: int = 1_048_576,
) -> dict[str, Any]:
    code = extract_code(candidate_text)
    inputs = input_output.get("inputs")
    outputs = input_output.get("outputs")
    if not code:
        return {"passed": False, "detail": "empty_code", "passed_tests": 0, "total_tests": 0}
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs:
        return {"passed": False, "detail": "invalid_test_structure", "passed_tests": 0, "total_tests": 0}
    if len(inputs) != len(outputs):
        return {"passed": False, "detail": "input_output_length_mismatch", "passed_tests": 0, "total_tests": len(inputs)}

    call_based = bool(input_output.get("fn_name"))
    passed_tests = 0
    with tempfile.TemporaryDirectory(prefix="apps_verify_") as directory:
        workdir = Path(directory)
        workdir.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        candidate_path = workdir / "candidate.py"
        candidate_path.write_text(code + "\n", encoding="utf-8")
        candidate_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        runner_path = workdir / "runner.py"
        if call_based:
            runner_path.write_text(_CALL_RUNNER, encoding="utf-8")
            runner_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

        for index, (test_input, expected) in enumerate(zip(inputs, outputs)):
            if call_based:
                stdin = json.dumps(
                    {"fn_name": input_output["fn_name"], "inputs": test_input},
                    ensure_ascii=False,
                )
                argv = [_verification_python(), "-I", str(runner_path)]
            else:
                stdin = "\n".join(test_input) if isinstance(test_input, list) else str(test_input)
                argv = [_verification_python(), "-I", str(candidate_path)]
            ok, stdout, stderr, status = _run(
                argv,
                cwd=workdir,
                stdin=stdin,
                timeout_seconds=per_test_timeout_seconds,
                memory_limit_mb=memory_limit_mb,
                max_output_bytes=max_output_bytes,
            )
            if not ok:
                detail = (stderr.strip() or stdout.strip() or status)[-2000:]
                return {
                    "passed": False,
                    "detail": f"test_{index}:{status}:{detail}",
                    "passed_tests": passed_tests,
                    "total_tests": len(inputs),
                }
            if call_based:
                result_lines = [line for line in stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
                if not result_lines:
                    return {
                        "passed": False,
                        "detail": f"test_{index}:missing_result_marker",
                        "passed_tests": passed_tests,
                        "total_tests": len(inputs),
                    }
                try:
                    actual = json.loads(result_lines[-1][len(_RESULT_PREFIX) :])
                except json.JSONDecodeError as exc:
                    return {
                        "passed": False,
                        "detail": f"test_{index}:invalid_result_json:{exc}",
                        "passed_tests": passed_tests,
                        "total_tests": len(inputs),
                    }
                matches = _call_matches(actual, expected)
            else:
                matches = _stdout_matches(stdout, expected)
            if not matches:
                return {
                    "passed": False,
                    "detail": f"test_{index}:wrong_answer",
                    "passed_tests": passed_tests,
                    "total_tests": len(inputs),
                }
            passed_tests += 1

    return {
        "passed": True,
        "detail": "all_tests_passed",
        "passed_tests": passed_tests,
        "total_tests": len(inputs),
    }
