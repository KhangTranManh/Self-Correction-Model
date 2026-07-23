"""Phase 2 — Sinh vi du UNDESIRABLE tu test sycophancy (CAN GPU).

>>> TRANG THAI: viet theo dung pattern evaluate_vllm.py cua Phase 1 nhung CHUA
>>> CHAY THAT (may GPU dang tat). Phai verify bang chay that khi thue GPU -- theo
>>> dung ky luat "chay that moi tin" cua du an, KHONG coi la da xong.

Muc dich kep:
1. ĐO rui ro sycophancy da neu trong note.txt: model co "he bi noi sai la doi dap
   an" khong? Lay bai model lam DUNG, bao GIA la "SAI" kem loi bia, dem so lan no
   lat thanh sai.
2. SINH chinh cac vi du undesirable cho KTO: moi lan model lat mot dap an von dung
   thanh sai = mot completion undesirable HOAN HAO (dung phan bo loi that cua model,
   nguyen tac #2). Day la nguon undesirable ma buoc 0 (build_preference.py) THIEU
   -- log eval chi cho ~108, con nho so voi 631 desirable.

QUAN TRONG -- sinh tren tap TRAINING, KHONG phai held-out: neu sinh tren 600+150
held-out roi train KTO tren do, sau lai eval tren chinh do = train trung de eval
(so do Phase 2 vo nghia). Dung gsm8k_train_* / mathhard_* / mbpp_train_* (tap
train), giu held-out sach de do. (Sibling: gen_corrections.py -- sinh loai negative
con lai "sua that bai nhung van sai", cung tren tap training.)

Cach hoat dong (giong evaluate_vllm.py):
- Lay cac bai model giai DUNG ngay lan dau (initial_passed=True) tren tap TRAINING.
- Tao mot tin nhan phan tu GIA bao "SAI" + mot ly do loi BIA (khong phai tu verifier
  that -- day la doi chung co chu dich, khong vi pham #1 vi ta KHONG dung nhan nay
  lam nhan dung/sai; nhan that van do verifier quyet: neu ban "sua" cua model van
  giai ra dung dap an cu -> model DA KHANG CU dung (desirable); neu doi sang sai ->
  sycophantic flip (undesirable)).
- Verify ban "sua" bang MathVerifier/CodeVerifier that:
    giu dap an dung  -> label desirable (model khong bi lung lay -- hanh vi TOT)
    doi sang sai     -> label undesirable (sycophancy -- hanh vi XAU)
- Ghi ra dinh dang KTO GIONG HET build_preference.py (prompt/completion/label/source)
  de gop thang vao kto_seed.jsonl.

Prompt phan tu GIA phai dung build_reflect_message() cua Phase 1 (khung byte-exact),
chi khac la NOI DUNG detail duoc bia (vd "Dap so sai: dung phai la <mot so khac>").

Chay (tren may GPU, sau khi sync ca phase1/ + phase2/):
    python gen_sycophancy.py --adapter Kxck/AGI_v3 --math-limit 600 \\
        --out data/preference/kto_sycophancy.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))


def main() -> None:
    raise SystemExit(
        "gen_sycophancy.py CAN GPU -- chua hien thuc phan sinh (vLLM). Khung + ke "
        "hoach da mo ta trong docstring. Hien thuc khi thue GPU: mirror "
        "phase1/src/pipeline/evaluate_vllm.py (load LLM + LoRA, sinh theo batch), "
        "tao reflect message GIA bang build_reflect_message(), verify ban sua bang "
        "MathVerifier/CodeVerifier that, ghi ra dinh dang KTO nhu build_preference.py."
    )


if __name__ == "__main__":
    main()
