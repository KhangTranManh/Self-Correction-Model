"""Interface chung cho moi verifier: nhan (Problem, candidate_text) -> VerifierResult khach quan."""
from __future__ import annotations

from typing import Protocol

from src.core.schema import Problem, VerifierResult


class Verifier(Protocol):
    def verify(self, problem: Problem, candidate_text: str) -> VerifierResult: ...
