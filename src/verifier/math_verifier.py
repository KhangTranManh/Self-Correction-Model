"""Verifier cho bai toan: trich xuat dap so cuoi cung tu candidate_text va so sanh
voi reference_answer bang sympy (khong phu thuoc LLM phan xet)."""
from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import parse_expr

from src.data.schema import Problem, VerifierResult

# Uu tien tim theo cac pattern tuong minh, fallback ve so cuoi cung xuat hien trong text.
_ANSWER_PATTERNS = [
    r"\\boxed\{([^}]*)\}",
    r"(?:Dap so|Đáp số|Ket qua|Kết quả|Answer)\s*[:=]\s*([^\n]+)",
]


def _extract_answer(text: str) -> str | None:
    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


# Bo ky hieu tien te/don vi di truoc so (vd "$18", "€5").
_CURRENCY_PREFIX_RE = re.compile(r"^[\$€£¥₫\s]+")
# Dau phay la thousands separator kieu Anh-My (vd "70,000" hay "1,234,567"),
# KHONG PHAI dau thap phan -- phai bo dau phay nay chu khong duoc doi thanh dau cham.
_THOUSANDS_SEP_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
# Chi lay phan so/phan thuc dan dau, bo qua don vi di sau (vd "84 cm²", "5 kg").
_LEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")


def _to_sympy(value: str):
    value = value.strip().rstrip(".")
    value = _CURRENCY_PREFIX_RE.sub("", value).strip()
    if _THOUSANDS_SEP_RE.match(value):
        value = value.replace(",", "")
    else:
        # dau phay o day la dau thap phan kieu VN (vd "84,5" -> "84.5")
        value = value.replace(",", ".")
    match = _LEADING_NUMBER_RE.match(value)
    if match:
        value = match.group(0)
    return parse_expr(value, evaluate=True)


class MathVerifier:
    def verify(self, problem: Problem, candidate_text: str) -> VerifierResult:
        extracted = _extract_answer(candidate_text)
        if extracted is None:
            return VerifierResult(passed=False, detail="Khong trich xuat duoc dap so tu candidate_text")

        try:
            candidate_value = _to_sympy(extracted)
            reference_value = _to_sympy(problem.reference_answer)
        except Exception as e:
            # Bat rong (khong chi SympifyError/TypeError/ValueError/SyntaxError): sympy's
            # parse_expr co the nem ca tokenize.TokenError (chuoi extract di dang, vd thieu
            # dau ngoac kep) hoac cac loi noi bo khac tuy phien ban -- moi loi parse deu
            # nghia la "khong verify duoc" -> coi la fail, khong duoc de crash ca eval run.
            return VerifierResult(passed=False, detail=f"Loi parse bieu thuc: {e}")

        try:
            is_equal = sympy.simplify(candidate_value - reference_value) == 0
        except TypeError:
            is_equal = candidate_value == reference_value

        if is_equal:
            return VerifierResult(passed=True, detail="OK")
        return VerifierResult(
            passed=False,
            detail=f"Dap so sai: model dua ra '{extracted}', dung phai la '{problem.reference_answer}'",
        )
