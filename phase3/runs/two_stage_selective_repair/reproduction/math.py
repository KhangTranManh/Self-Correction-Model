"""Verifier cho bai toan: trich xuat dap so cuoi cung tu candidate_text va so sanh
voi reference_answer bang sympy (khong phu thuoc LLM phan xet)."""
from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from src.core.schema import Problem, VerifierResult

# implicit_multiplication_application: can THIET cho dap an LaTeX cua bo MATH,
# noi dau nhan thuong bi bo ngam (vd "3\pi" -> sau khi doi \pi -> "pi" thanh
# "3pi", KHONG co dau '*'). Neu khong bat transformation nay, parse_expr("3pi")
# se raise loi (khong phai sai am tham -- van an toan nho except o duoi -- nhung
# se mat oan nhieu case dung dinh dang pho bien). Transformation nay khong doi
# hanh vi cua cac case cu (so thuan da duoc rut gon truoc do, khong co token ke
# nhau can nhan ngam).
_MATH_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def extract_last_boxed(text: str) -> str | None:
    """Tim noi dung ben trong \\boxed{...} CUOI CUNG, dem do sau ngoac cho dung.

    KHONG dung regex don gian kieu r"\\boxed\\{([^}]*)\\}": dap an LaTeX thuong co
    ngoac long nhau (vd "\\boxed{\\frac{1}{2}}" co 2 dau '}' truoc dau '}' dong
    boxed that su) -- regex khong dem do sau se cat cut o dau '}' dau tien, lay
    nham "\\frac{1" thay vi "\\frac{1}{2}". Dung CHUNG boi 2 noi: _extract_answer()
    o duoi (doc dap an model tu viet ra) va prepare_datasets.prepare_math_hard()
    (doc dap an CHUAN cua bo MATH) -- ca 2 gap chung 1 rui ro nen chung 1 ham,
    tranh 2 ban sao co the lech nhau ve sau (giong ly do src/core/prompts.py ton tai).
    """
    key = "\\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    depth = 1
    buf: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf).strip() if depth == 0 else None


# Fallback khi khong co \boxed{}: tim theo pattern tuong minh, roi moi den so cuoi
# cung xuat hien trong text.
#
# PHAT HIEN THUC TE (2026-07-22, qua eval AGI_v3): sau khi train them du lieu MATH
# (dinh dang \boxed{}/markdown), model bat dau viet dap so kieu LAP HAI LOP:
#     "### Đáp số:\nĐáp số: 15"
# Voi \s* (khop ca xuong dong) sau dau ':', regex CU nuot qua dong dau, bat TRON
# CA DONG SAU lam gia tri -> extract ra "Đáp số: 15" thay vi "15" -> parse_expr
# fail -> BI TINH LA SAI dù model giai dung. Da do duoc: 112/170 case "sai toan"
# cua adapter v3 thuc ra la loi trich xuat nay, khong phai sai logic that.
# Sua: gioi han khoang trang SAU dau ':' chi trong CUNG DONG ([ \t]*, khong \s*)
# -- "nhan: gia tri" ve ban chat luon nam tron 1 dong, khong co ly do gi de nhan
# ngang qua xuong dong.
_ANSWER_PATTERNS = [
    r"(?:Dap so|Đáp số|Ket qua|Kết quả|Answer)[ \t]*[:=][ \t]*([^\n]+)",
]


def _extract_answer(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed

    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


# --- Ho tro dap an LaTeX (bo MATH/Hendrycks) -------------------------------
# CO CHU DICH: day KHONG phai mot parser LaTeX day du, chi xu ly tap con cu
# phap thuong gap nhat trong dap an \boxed{} cua MATH. Neu sau khi ap dung het
# cac buoc duoi day van con lenh LaTeX chua biet (dau '\' theo sau chu), COI LA
# KHONG PARSE DUOC thay vi co gang doan -- xem _to_sympy(): loi do se duoc
# MathVerifier.verify() bat va tra ve fail, KHONG BAO GIO am tham sai. Trieu
# hoc (toa do, tap hop, ma tran...) vuot qua tap con nay se tu dong bi loai
# (bai bi bo qua khoi ket qua, khong bi cham sai).
#
# \frac{a}{b} / \dfrac{a}{b} -- so sau "d" tuy chon (\dfrac = display fraction).
_LATEX_FRAC_RE = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_SQRT_N_RE = re.compile(r"\\sqrt\[(\d+)\]\{([^{}]*)\}")  # \sqrt[n]{a}
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")
_LATEX_TEXT_RE = re.compile(r"\\text\{[^{}]*\}")  # \text{...} -- thuong la don vi/nhan, bo di
_LATEX_STRIP_RE = re.compile(r"\\left|\\right|\\!|\\,|\\;")
_LATEX_MACROS = (
    (r"\cdot", "*"),
    (r"\times", "*"),
    (r"\div", "/"),
    (r"\pi", "pi"),
    (r"\infty", "oo"),
)


def _latex_to_plain(s: str) -> str:
    """Doi cu phap LaTeX pho bien trong dap an MATH thanh dang sympy doc duoc.

    Chay frac/sqrt/text qua NHIEU VONG LAP (toi da 5) de xu ly duoc long nhau o
    muc vua phai (vd "\\frac{1}{2\\sqrt{3}}") -- moi vong giai quyet lop trong
    cung (khong con dau ngoac con) truoc, lam lo ra lop tiep theo cho vong sau.
    """
    for _ in range(5):
        prev = s
        s = _LATEX_FRAC_RE.sub(r"((\1)/(\2))", s)
        s = _LATEX_SQRT_N_RE.sub(r"((\2)**(1/\1))", s)
        s = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s)
        s = _LATEX_TEXT_RE.sub("", s)
        if s == prev:
            break
    s = _LATEX_STRIP_RE.sub("", s)
    for macro, replacement in _LATEX_MACROS:
        s = s.replace(macro, replacement)
    return s


# Bo ky hieu tien te/don vi di truoc so (vd "$18", "€5").
_CURRENCY_PREFIX_RE = re.compile(r"^[\$€£¥₫\s]+")
# Dau phay la thousands separator kieu Anh-My (vd "70,000" hay "1,234,567"),
# KHONG PHAI dau thap phan -- phai bo dau phay nay chu khong duoc doi thanh dau cham.
_THOUSANDS_SEP_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
# Chi coi la "so co dau phay" (roi moi quyet dinh thousands-sep hay VN-decimal)
# neu CA CHUOI chi gom so/dau phay/1 dau cham -- vd "70,000" hay "84,5". PHAT
# HIEN THUC TE (2026-07-22, chay that tren may co Python, khong phai doan): neu
# ap dung mu quang logic nay len bieu thuc phuc hop tu LaTeX (vd toa do cuc
# "( 3, pi/2 )"), dau phay phan tach toa do bi doi nham thanh dau cham, roi
# implicit_multiplication_application noi "3." voi "(pi/2)" thanh PHEP NHAN
# "3 * pi/2" = 1.5*pi -- MOT GIA TRI SCALAR SAI nhung PARSE DUOC, khong crash.
# Hai toa do khac nhau (vd "(3, pi/2)" va "(6, pi/4)") se cung cho ra 1.5*pi ->
# CHAM NHAM la bang nhau. Day la loai "sai am tham" nguy hiem nhat, khong phai
# gia thuyet -- da bat duoc bang test that (xem note.txt). Voi gate nay, dau
# phay trong bieu thuc phuc hop duoc GIU NGUYEN, de parse_expr hieu dung nhu
# dau phay Python (tuple/list) thay vi bi bien doi sai.
_BARE_NUMBER_RE = re.compile(r"^-?[\d,]+(\.\d+)?$")
# Chi lay phan so/phan thuc dan dau, bo qua don vi di sau (vd "84 cm²", "5 kg").
_LEADING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?")
# A bare numeric answer followed by a prose unit must be normalized before the
# full SymPy parse.  With implicit multiplication enabled, strings such as
# ``10 quả`` or ``25 phút`` can otherwise parse successfully as symbolic
# multiplication instead of raising and reaching the numeric fallback below.
# Restrict this fast path to a numeric prefix plus a tail containing no digits
# or arithmetic syntax so expressions such as ``3pi`` and ``2 + 3`` retain the
# normal expression parser.
_NUMBER_WITH_PROSE_UNIT_RE = re.compile(
    r"^(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s+[^\d+\-*/^()=]+$"
)


# Cac case hoi quy PHAI giu nguyen ket qua neu sua file nay -- DA CHAY THAT (khong
# chi doan) qua test_math_verifier.py tren may co Python, 2026-07-22, 13/13 pass:
#   "$18" vs "18"                    -> pass (cu, GSM8K)
#   "84 cm²" vs "84"                 -> pass (cu, GSM8K, don vi vat ly)
#   "$70,000" vs "70000"             -> pass (cu, thousands separator)
#   "1,234,567" vs "1234567"         -> pass (cu, thousands separator nhieu nhom)
#   "84,5" vs "84.5"                 -> pass (cu, dau phay thap phan kieu VN)
#   "17" vs "18"                     -> FAIL (cu, phai bat duoc sai that)
#   "\frac{1}{2}" vs "0.5"           -> pass (moi, LaTeX phan so)
#   "\dfrac{14}{3}" vs "14/3"        -> pass (moi, dfrac + so sanh voi phan so thuong)
#   "3\pi" vs "3\pi"                 -> pass (moi, nhan ngam qua implicit_multiplication_application)
#   "\sqrt{4}" vs "2"                -> pass (moi, can bac 2 hoan hao)
#   "(3, pi/2)" vs "(3, pi/2)"        -> pass (moi, tuple giong nhau -- parse_expr
#       tu hieu dung nhu Python tuple, so sanh qua == fallback trong verify())
#   "(6, pi/4)" vs "(3, pi/2)"        -> FAIL (moi, tuple KHAC nhau)
#
# CANH BAO DA SUA (2026-07-22, phat hien qua chay test THAT, KHONG phai doan
# bang tay -- lan doan tay truoc do bo sot loi nay): logic "dau phay la thousands-
# sep hay VN-decimal" TRUOC DAY ap dung mu quang len MOI gia tri, ke ca bieu thuc
# phuc hop. Voi toa do cuc "( 3, pi/2 )", dau phay phan tach bi doi nham thanh
# dau cham -> "( 3. (pi/2) )" -> implicit_multiplication_application NOI "3."
# voi "(pi/2)" thanh phep nhan -> 1.5*pi -- MOT GIA TRI SAI nhung PARSE DUOC,
# khong crash. Hai toa do KHAC NHAU (vd (3,pi/2) va (6,pi/4)) se CUNG ra 1.5*pi
# -> cham nham la bang nhau (false positive, loai sai am tham nguy hiem nhat).
# Da sua bang _BARE_NUMBER_RE: chi dong bo dau phay khi gia tri la so tran.
def _to_sympy(value: str):
    value = value.strip().rstrip(".")
    # LaTeX (bo MATH) PHAI xu ly TRUOC logic tien te/thousands-sep ben duoi: logic
    # do gia dinh gia tri la mot so thuan (vd "$18,000"), khong danh cho bieu thuc
    # dai so. Neu dap an khong phai LaTeX (vd GSM8K), buoc nay la no-op vo hai.
    value = _latex_to_plain(value)
    if "\\" in value:
        # Con lenh LaTeX chua ho tro (vd toa do "(3, \pi/2)", tap hop, ma tran)
        # -- KHONG doan, coi thang la khong parse duoc. parse_expr o duoi se
        # raise (vi con ky tu '\'), duoc MathVerifier.verify() bat va tra ve
        # fail -- an toan, khong bao gio am tham sai.
        raise ValueError(f"Con cu phap LaTeX chua ho tro: {value!r}")
    value = _CURRENCY_PREFIX_RE.sub("", value).strip()
    # CHI dong bo dau phay khi gia tri la SO TRAN (xem _BARE_NUMBER_RE o tren de
    # biet ly do gate nay bat buoc phai co). Bieu thuc phuc hop (co ngoac/chu tu
    # LaTeX) giu nguyen dau phay -- parse_expr se tu hieu dung nhu dau phay Python.
    if _BARE_NUMBER_RE.match(value):
        if _THOUSANDS_SEP_RE.match(value):
            value = value.replace(",", "")
        else:
            # dau phay o day la dau thap phan kieu VN (vd "84,5" -> "84.5")
            value = value.replace(",", ".")

    unit_match = _NUMBER_WITH_PROSE_UNIT_RE.match(value)
    if unit_match:
        return parse_expr(
            unit_match.group(1), transformations=_MATH_TRANSFORMATIONS, evaluate=True
        )

    # Thu parse CA CHUOI truoc. QUAN TRONG cho bieu thuc dai so (vd "3pi",
    # "sqrt(2)", "(14)/(3)") -- buoc cat-lay-so-dan-dau ben duoi von thiet ke cho
    # GSM8K (cat don vi vat ly kieu "84 cm²" -> "84") se pha hong bieu thuc dai
    # so neu ap dung mu quang: "3pi" se bi cat con "3", MAT HAN "pi" ma khong co
    # loi nao bao -- sai am tham, dung loai bug nguy hiem nhat. Chi khi parse ca
    # chuoi that bai (con don vi/ky tu la) moi fallback ve cach cu.
    try:
        return parse_expr(value, transformations=_MATH_TRANSFORMATIONS, evaluate=True)
    except Exception:
        match = _LEADING_NUMBER_RE.match(value)
        if not match:
            raise
        return parse_expr(match.group(0), transformations=_MATH_TRANSFORMATIONS, evaluate=True)


# Nhan so trong van ban tu do cua model, dung de doi chieu voi tung calc_step.
# Khop ca so nguyen am/duong va so thap phan; dau phay duoc chuan hoa rieng ben
# duoi (khong dua vao regex nay tach thousands-sep, vi van ban tu do model viet
# khong theo dinh dang chuan nhu dap an).
_NUMBER_IN_TEXT_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def locate_wrong_step(calc_steps: list[list[str]] | None, attempt_text: str) -> str | None:
    """Doi chieu tung buoc tinh chuan (calc_steps, tu GSM8K) voi cac con so model
    TU VIET RA trong attempt_text, tim buoc DAU TIEN ma gia tri chuan KHONG xuat
    hien dau trong bai lam -- danh dau diem loi kha di CO THE bat dau tu do.

    CHU DICH thiet ke -- CHI dinh vi buoc nghi van, KHONG tiet lo dap so dung (so
    voi thong bao cu "dung phai la X"): muc tieu la thu nghiem xem viec chi RA
    VI TRI loi (tuong tu traceback cua CodeVerifier tro dung dong/ham) co giup
    model tu sua tot hon la duoc "mom" san dap an hay khong -- xem note.txt,
    phan ban chat "addressability" cua toan hoc so voi code.

    Day la HEURISTIC THUAN TUY khop chuoi so trong van ban tu do, KHONG PHAI
    logic toan hoc: co the sai neu model gop nhieu buoc lam mot, tinh nham roi
    tinh lai dung, hoac trinh bay theo thu tu khac. Vi vay KHONG duoc dung ket
    qua nay de quyet dinh pass/fail (xem MathVerifier.verify -- pass/fail luon
    chi dua tren so sanh dap so cuoi cung qua sympy). Tra ve None neu khong co
    calc_steps hoac khong dinh vi duoc gi, de noi goi fallback ve thong bao cu.
    """
    if not calc_steps:
        return None

    raw_numbers = _NUMBER_IN_TEXT_RE.findall(attempt_text)
    # so sanh ca dang goc lan dang da bo dau phay (model co the viet "1,234"),
    # dung set de tra cuu O(1) thay vi quet lai chuoi cho tung buoc.
    attempt_numbers = set(raw_numbers) | {n.replace(",", "") for n in raw_numbers}

    for expr, value in calc_steps:
        if value not in attempt_numbers and value.replace(",", "") not in attempt_numbers:
            return (
                f"Gợi ý: bước tính liên quan tới '{expr}' trong lời giải có vẻ "
                f"chưa đúng hoặc chưa được thực hiện — hãy kiểm tra và tính lại "
                f"bước đó."
            )
    return None


class MathVerifier:
    def __init__(self, localize_steps: bool = False):
        # Mac dinh TAT (giu nguyen hanh vi cu, "dap so sai: dung phai la X") --
        # localize_steps=True la CO CHU DICH opt-in cho thu nghiem so sanh (giong
        # cach --blind/--reflect-role da duoc them lam flag rieng thay vi doi
        # mac dinh ngay), tranh am tham doi hanh vi cua build_dataset/eval hien
        # co truoc khi co ket qua so sanh ro rang.
        self.localize_steps = localize_steps

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

        if self.localize_steps:
            localized = locate_wrong_step(getattr(problem, "calc_steps", None), candidate_text)
            if localized is not None:
                return VerifierResult(passed=False, detail=localized)

        return VerifierResult(
            passed=False,
            detail=f"Dap so sai: model dua ra '{extracted}', dung phai la '{problem.reference_answer}'",
        )
