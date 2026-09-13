"""Verifier cho bai code: chay candidate code + unit tests trong subprocess co timeout/memory limit.

Luu y: day la sandbox muc "nghien cuu" (subprocess + resource limit), du dung cho
bai code ngan tu cac problem set tin cay (GSM8K/MBPP-style tu sinh), KHONG phai
sandbox chong code doc hai muc production. Neu mo rong sang nguon problem khong
tin cay, can container hoa (docker/firejail) truoc khi chay.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from src.core.schema import Problem, VerifierResult

_CODE_FENCE_PATTERNS = [
    r"```python\s*\n(.*?)```",
    r"```\s*\n(.*?)```",
]


def _extract_code(text: str) -> str:
    for pattern in _CODE_FENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1)
    return text  # fallback: coi ca text la code


def _build_preexec_fn(memory_limit_mb: int):
    if sys.platform.startswith("win"):
        return None  # RLIMIT_AS khong ho tro tren Windows

    import resource

    def _limit():
        limit_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return _limit


class CodeVerifier:
    def __init__(self, timeout_seconds: int = 5, memory_limit_mb: int = 256):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

    def verify(self, problem: Problem, candidate_text: str) -> VerifierResult:
        code = _extract_code(candidate_text)
        script = code + "\n\n" + "\n".join(problem.tests) + "\nprint('__ALL_TESTS_PASSED__')\n"

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            script_path = Path(f.name)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                preexec_fn=_build_preexec_fn(self.memory_limit_mb),
            )
        except subprocess.TimeoutExpired:
            return VerifierResult(passed=False, detail=f"Timeout sau {self.timeout_seconds}s (co the vong lap vo han)")
        finally:
            script_path.unlink(missing_ok=True)

        if result.returncode == 0 and "__ALL_TESTS_PASSED__" in result.stdout:
            return VerifierResult(passed=True, detail="OK")

        error_detail = result.stderr.strip() or result.stdout.strip() or "Loi khong xac dinh"
        return VerifierResult(passed=False, detail=error_detail[-1000:])
