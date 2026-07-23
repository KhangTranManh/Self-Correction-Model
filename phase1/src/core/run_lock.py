"""Khoa chong chay 2 instance cung luc cua cung mot script ghi file.

VI SAO CAN: build_dataset.py va generate_attempts.py deu mo file output o che do
append. Hai tien trinh chay song song tren cung duong dan se ghi xen ke nhau -- tao
ban ghi trung, dong JSON hong, va (voi build_dataset) TON TIEN API GAP DOI cho cung
mot bai.

Truoc day day chi la mot dong canh bao trong tai lieu ("luon `ps aux | grep` truoc
khi chay"). Da vi pham thuc te: hai chuoi dieu phoi cung goi build_dataset, chay
song song ~5 phut, sinh ra 8 ban ghi trung. Canh bao bang van ban khong ngan duoc
loi nay -- nen bien no thanh khoa that.

Khoa la file chua PID. Neu PID trong khoa khong con song (vd may khoi dong lai, hoac
tien trinh bi kill -9), khoa duoc coi la cu va tu don -- de khong bi ket vinh vien
sau mot lan crash.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def acquire_single_instance(name: str, lock_dir: Path) -> None:
    """Ban khong-context-manager: giu khoa den khi tien trinh thoat.

    Dung khi phan can bao ve trai dai gan het ham main() -- boc ca khoi do vao `with`
    chi de lay khoa se lam thut le toan bo ham ma khong them y nghia gi.
    """
    import atexit

    cm = single_instance(name, lock_dir)
    cm.__enter__()
    atexit.register(lambda: cm.__exit__(None, None, None))


@contextmanager
def single_instance(name: str, lock_dir: Path):
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{name}.lock"

    if lock_path.exists():
        try:
            other = int(lock_path.read_text().strip() or 0)
        except ValueError:
            other = 0
        if other and _pid_alive(other):
            raise RuntimeError(
                f"'{name}' dang chay o tien trinh PID {other} (khoa: {lock_path}).\n"
                f"KHONG chay 2 instance cung luc -- chung ghi de len cung file output.\n"
                f"Neu chac chan tien trinh do da chet: xoa file khoa roi chay lai."
            )
        print(f"[run_lock] Don khoa cu cua PID {other} (khong con song).")

    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        try:
            if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
                lock_path.unlink()
        except OSError:
            pass
