"""Cac dataclass dung chung xuyen suot pipeline Phase 1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


Domain = Literal["math", "code"]


@dataclass
class Problem:
    id: str
    domain: Domain
    question: str
    # math: dap so tham chieu (string, se duoc verifier normalize/parse)
    reference_answer: Optional[str] = None
    # code: ten ham can implement + danh sach assert statement dung de test
    entry_point: Optional[str] = None
    tests: list[str] = field(default_factory=list)
    # math (chi GSM8K): chuoi phep tinh trung gian tu-danh-so cua GSM8K, dang
    # [["48/2", "24"], ["48+24", "72"]] -- trich tu annotation "<<48/2=24>>"
    # trong raw answer. Dung de dinh vi buoc sai (xem locate_wrong_step trong
    # src/data/verifiers/math.py) thay vi chi bao dap so cuoi sai. None/rong
    # neu khong co (vd bai MATH-hard tu Hendrycks, hoac bai code).
    calc_steps: Optional[list[list[str]]] = None


@dataclass
class Attempt:
    problem_id: str
    text: str  # raw completion tu small model


@dataclass
class VerifierResult:
    passed: bool
    detail: str  # ly do fail (vd: "test case 2 loi: expected 4 got 3") hoac "OK"


@dataclass
class CorrectionRecord:
    """1 sample hoan chinh, da duoc verify 2 lan (attempt sai + correction dung)."""

    problem: Problem
    attempt_text: str
    verifier_detail: str
    error_location: str
    error_reason: str
    corrected_solution: str
