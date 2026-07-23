"""Tai bo du lieu chuan GSM8K (toan) va MBPP (code) tu HuggingFace datasets,
convert sang dinh dang JSONL cua project -- ghi de data/problems/math.jsonl
va data/problems/code.jsonl (bo mau 3+3 truoc do qua de, model lam dung het).

QUAN TRONG ve split: ban dau (150+150 mau) lay tu split "test" cua ca 2 dataset,
index 0-149. evaluate_self_correction.py (held-out eval) CUNG lay tu split
"test" nhung offset 150 tro len -- khong dung nhau VOI DIEU KIEN limit <= 150.
Neu scale limit len qua 150 ma van dung split "test", se DE LEN vung held-out
dang dung de danh gia (lam ho ca chi so da theo doi xuyen suot). Vi vay ham
prepare_math_extra/prepare_code_extra ben duoi dung split "train" (GSM8K:
7473 dong, hoan toan tach biet voi "test") va "full/train" cua MBPP (374 dong,
tach biet voi "full/test" ma eval dang dung) de sinh THEM du lieu train, APPEND
vao file cu (khong ghi de), voi id prefix rieng de khong dam voi id cu.

Chay (khong can GPU, chi can internet de tai dataset):
    python -m src.pipeline.prepare_datasets --math-limit 50 --code-limit 50        # ghi de file goc (0-149 lan dau)
    python -m src.pipeline.prepare_datasets --math-extra 600 --code-extra 374      # APPEND them tu split train, khong dam voi eval
    python -m src.pipeline.prepare_datasets --math-hard-limit 300 --math-hard-levels 2,3
        # them 300 bai TOAN KHO (MATH, level 2-3) -- van APPEND vao math.jsonl,
        # domain van la "math" nen KHONG dung gi toi code.jsonl/CodeVerifier.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from src.data.verifiers.math import extract_last_boxed

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # file nam trong src/<package>/, can 3 cap moi toi goc du an


def _extract_gsm8k_answer(answer_field: str) -> str:
    # GSM8K answer format: "... reasoning ... #### 42"
    match = re.search(r"####\s*([\-0-9,.]+)", answer_field)
    if not match:
        raise ValueError(f"Khong tim thay dap so cuoi trong: {answer_field!r}")
    return match.group(1).replace(",", "")


_CALC_STEP_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def _extract_calc_steps(answer_field: str) -> list[list[str]]:
    """Trich cac phep tinh trung gian tu-danh-so cua GSM8K (vd '<<48/2=24>>'
    -> ['48/2', '24']), theo DUNG THU TU xuat hien trong loi giai chuan. Dung
    de dinh vi buoc sai trong loi giai cua model (xem locate_wrong_step trong
    src/data/verifiers/math.py) thay vi chi noi dap so cuoi cung sai.

    Rong neu khong tim thay annotation nao -- luon xay ra voi cac nguon toan
    khac GSM8K (vd MATH-hard tu Hendrycks khong co dinh dang calculator nay);
    cac noi goi phai xu ly rong nhu "khong co thong tin dinh vi", KHONG loi.
    """
    return [[expr.strip(), value.strip()] for expr, value in _CALC_STEP_RE.findall(answer_field)]


def _extract_entry_point(code: str) -> str | None:
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else None


def prepare_math(limit: int, out_path: Path) -> None:
    # "gsm8k" (khong namespace) la dataset script cu, huggingface_hub/datasets ban moi
    # khong con resolve duoc -> phai dung id chinh tac moi tren Hub (cung field question/answer).
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(limit, len(ds))))

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            record = {
                "id": f"gsm8k_{i:04d}",
                "question": row["question"],
                "reference_answer": _extract_gsm8k_answer(row["answer"]),
                "calc_steps": _extract_calc_steps(row["answer"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Da ghi {len(ds)} bai toan (GSM8K) vao {out_path}")


def prepare_code(limit: int, out_path: Path) -> None:
    # "mbpp" (khong namespace) cung la dataset script cu, cung ly do nhu gsm8k o tren.
    # load_dataset("Muennighoff/mbpp", split="test") bi CastError vi ban parquet auto-convert
    # cua HF gop nham 2 config (full/sanitized, khac schema) -> tro thang toi file parquet
    # cua dung config "full/test" (giu nguyen field task_id/text/code/test_list/test_setup_code).
    ds = load_dataset(
        "parquet",
        data_files=(
            "hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/full/test/0000.parquet"
        ),
        split="train",
    )
    ds = ds.select(range(min(limit, len(ds))))

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            entry_point = _extract_entry_point(row["code"])
            if entry_point is None:
                continue  # bo qua neu khong parse duoc ten ham tu code mau

            tests = list(row["test_list"])
            if row.get("test_setup_code"):
                tests = [row["test_setup_code"]] + tests

            record = {
                "id": f"mbpp_{row['task_id']}",
                "question": row["text"],
                "entry_point": entry_point,
                "tests": tests,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Da ghi {written} bai code (MBPP) vao {out_path}")


def _existing_ids(path: Path) -> set[str]:
    """Doc id da co trong file de KHONG append trung.

    Can thiet vi cac ham *_extra deu mo file o che do "a": chay lai voi limit lon hon
    (vd 626 -> 1350 de scale them) se append de len chinh nhung bai da co neu khong
    loc. Trung lap trong tap train khong bao loi -- no chi am tham lam lech phan bo
    du lieu, kieu bug rat kho phat hien ve sau.
    """
    if not path.exists():
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def prepare_math_extra(limit: int, out_path: Path) -> None:
    # split "train" (7473 dong) -- hoan toan tach biet voi split "test" ma ca
    # prepare_math() (data train ban dau) lan evaluate_self_correction.py
    # (held-out eval) deu dang dung -- an toan de scale ma khong dam vao eval.
    #
    # limit o day la TONG so bai train muon co, khong phai "them bao nhieu": bai da
    # co se bi bo qua nho _existing_ids, nen chay lai voi limit lon hon la cach dung
    # de scale (vd 626 -> 1350 se chi them 724 bai moi).
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.select(range(min(limit, len(ds))))

    seen = _existing_ids(out_path)
    written = skipped = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            record_id = f"gsm8k_train_{i:04d}"
            if record_id in seen:
                skipped += 1
                continue
            record = {
                "id": record_id,
                "question": row["question"],
                "reference_answer": _extract_gsm8k_answer(row["answer"]),
                "calc_steps": _extract_calc_steps(row["answer"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"Da APPEND {written} bai toan moi (GSM8K split=train) vao {out_path}"
        f" (bo qua {skipped} bai da co)"
    )


# MBPP co 974 bai, chia theo task_id thanh 4 khoi CO DINH:
#     prompt      task_id   1-10    (10 bai)
#     test        task_id  11-510   (500 bai)
#     validation  task_id 511-600   (90 bai)
#     train       task_id 601-974   (374 bai)
# "full/train" da duoc dung het (374/374). Con lai co the dung de train ma KHONG
# dam vao eval: validation (90) + prompt (10) = 100 bai.
#
# CANH BAO -- khong duoc lay split "test" qua duong nay: 11-160 da dung de train,
# con 161 tro len la vung held-out cua evaluate_self_correction.py. Cac ban MBPP
# gop san kieu "974 dong 1 file jsonl" (vd tren Kaggle) KHONG co nhan split -- nap
# thang vao se train len chinh de eval, va khong the hoan tac: moi so do sau do deu
# vo nghia vi model da nhin thay de.
_MBPP_SPLIT_SAFE_FOR_TRAIN = ("train", "validation", "prompt")


def prepare_code_extra(limit: int, out_path: Path, split: str = "train") -> None:
    if split not in _MBPP_SPLIT_SAFE_FOR_TRAIN:
        raise ValueError(
            f"split={split!r} khong duoc phep dung lam du lieu train. "
            f"Chi chap nhan {_MBPP_SPLIT_SAFE_FOR_TRAIN}. Split 'test' la vung "
            f"held-out cua eval -- train len no lam hong toan bo phep do."
        )

    ds = load_dataset(
        "parquet",
        data_files=(
            f"hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/full/{split}/0000.parquet"
        ),
        split="train",
    )
    ds = ds.select(range(min(limit, len(ds))))

    seen = _existing_ids(out_path)
    written = skipped = no_entry_point = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for row in ds:
            record_id = f"mbpp_{split}_{row['task_id']}"
            if record_id in seen:
                skipped += 1
                continue

            entry_point = _extract_entry_point(row["code"])
            if entry_point is None:
                no_entry_point += 1
                continue

            tests = list(row["test_list"])
            if row.get("test_setup_code"):
                tests = [row["test_setup_code"]] + tests

            record = {
                "id": record_id,
                "question": row["text"],
                "entry_point": entry_point,
                "tests": tests,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"Da APPEND {written} bai code moi (MBPP split=full/{split}) vao {out_path}"
        f" (bo qua {skipped} da co, {no_entry_point} khong tach duoc entry_point)"
    )


# MATH (Hendrycks et al.) -- toan THI DAU, kho hon han GSM8K (toan tieu hoc). Dung
# split "train" cua nguon cong khai EleutherAI/hendrycks_math (MIT license, KHONG
# gated -- da xac nhan qua HF Hub API 2026-07-22), tach biet split "test" cua chinh
# no (khong dung o dau khac trong du an -- an toan tuyet doi voi held-out).
#
# LY DO CAN NGUON NAY: GSM8K qua de voi model dang dung -- model giai dung ngay
# ~84.3% (xem note.txt phat hien so 2), nen them bai GSM8K gan nhu vo ich (84%
# se bi bo vi "already_correct"). MATH co do kho phan cap 1-5 (level), cho phep
# chon vung kho vua phai de tang ty le SAI (tang tin hieu train) ma khong nhay
# thang len do kho Olympic (da danh gia va LOAI ShadenA/MathNet vi phan lon la
# bai CHUNG MINH -- final_answer=null, khong co gi de verifier cham; xem note.txt).
_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def prepare_math_hard(levels: set[str], limit: int, out_path: Path) -> None:
    """APPEND bai toan KHO (MATH) vao math.jsonl. domain van la "math" nen tu
    dong di qua dung MathVerifier + prompt toan hien co (khong dung gi toi code).

    KHAC GSM8K: khong co cot dap so rieng -- phai trich \\boxed{...} tu loi giai
    (extract_last_boxed, dung chung voi verifier), va dap an la LATEX (vd "\\frac{14}{3}"), can
    MathVerifier._to_sympy da duoc mo rong hieu LaTeX (xem
    src/data/verifiers/math.py) moi verify dung.

    limit: TONG so bai MOI muon co (gop ca 7 mon), giong quy uoc cua
    prepare_math_extra/prepare_code_extra. limit<=0 nghia la KHONG chay (giu
    nguyen quy uoc "0 = tat" cua cac buoc *_extra khac trong main()).
    """
    seen = _existing_ids(out_path)
    written = skipped_seen = skipped_no_answer = skipped_level = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for subject in _MATH_SUBJECTS:
            if written >= limit:
                break
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
            for i, row in enumerate(ds):
                if written >= limit:
                    break
                if row["level"] not in levels:
                    skipped_level += 1
                    continue

                record_id = f"mathhard_{subject}_{i:04d}"
                if record_id in seen:
                    skipped_seen += 1
                    continue

                answer = extract_last_boxed(row["solution"])
                if answer is None:
                    # Khong tim thay \boxed{} -- KHONG doan dap so va bo qua bai
                    # nay. Mot "dap so chuan" sai se dau kin trong toan bo pipeline
                    # (verifier so sanh voi mot chuan sai ma khong ai biet) --
                    # nghiem trong hon nhieu so voi don gian mat 1 bai.
                    skipped_no_answer += 1
                    continue

                record = {
                    "id": record_id,
                    "question": row["problem"],
                    "reference_answer": answer,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(
        f"Da APPEND {written} bai toan KHO (MATH, level={sorted(levels)}) vao {out_path}\n"
        f"  bo qua: {skipped_seen} da co / {skipped_level} sai level / "
        f"{skipped_no_answer} khong trich duoc dap so tu \\boxed{{}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--math-limit", type=int, default=0,
        help="So bai GSM8K se lay tu split=test (GHI DE math.jsonl -- chi dung lan dau/reset)",
    )
    parser.add_argument(
        "--code-limit", type=int, default=0,
        help="So bai MBPP se lay tu split=full/test (GHI DE code.jsonl -- chi dung lan dau/reset)",
    )
    parser.add_argument(
        "--math-extra", type=int, default=0,
        help="So bai GSM8K THEM tu split=train (APPEND vao math.jsonl, khong dam voi held-out eval)",
    )
    parser.add_argument(
        "--code-extra", type=int, default=0,
        help="So bai MBPP THEM (APPEND vao code.jsonl). Bai da co se bi bo qua, nen "
             "chay lai voi so lon hon la cach dung de scale.",
    )
    parser.add_argument(
        "--code-extra-split", type=str, default="train",
        choices=list(_MBPP_SPLIT_SAFE_FOR_TRAIN),
        help="Split MBPP de lay --code-extra. 'train' (374) da dung het; con "
             "'validation' (90) va 'prompt' (10). Split 'test' bi chan co y: 161 tro "
             "len la vung held-out cua eval.",
    )
    parser.add_argument(
        "--math-hard-limit", type=int, default=0,
        help="So bai toan KHO (MATH, khac GSM8K) muon them, gop ca 7 mon "
             "(algebra/geometry/...). 0 (mac dinh) = KHONG chay. Van APPEND vao "
             "cung math.jsonl -- domain van la 'math', khong dung gi toi code.",
    )
    parser.add_argument(
        "--math-hard-levels", type=str, default="2,3",
        help="Cac level MATH muon lay, phan cach dau phay (vd '2,3'). "
             "1=de nhat, 5=kho nhat (Olympic). Mac dinh 2,3: du kho de tao tin hieu "
             "sai nhung chua qua kho toi muc model 'thay' (API) cung khong sua noi.",
    )
    args = parser.parse_args()

    math_path = ROOT_DIR / "data" / "problems" / "math.jsonl"
    code_path = ROOT_DIR / "data" / "problems" / "code.jsonl"

    if args.math_limit > 0:
        prepare_math(args.math_limit, math_path)
    if args.code_limit > 0:
        prepare_code(args.code_limit, code_path)
    if args.math_extra > 0:
        prepare_math_extra(args.math_extra, math_path)
    if args.code_extra > 0:
        prepare_code_extra(args.code_extra, code_path, split=args.code_extra_split)
    if args.math_hard_limit > 0:
        levels = {f"Level {x.strip()}" for x in args.math_hard_levels.split(",") if x.strip()}
        prepare_math_hard(levels, args.math_hard_limit, math_path)


if __name__ == "__main__":
    main()
