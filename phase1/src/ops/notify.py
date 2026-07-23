"""Gui thong bao tien do pipeline qua Telegram.

CO Y khong dung thu vien ngoai (chi urllib + json cua stdlib) va tu doc .env thay vi
qua src/config.py: script nay phai chay duoc bang BAT KY python nao tren may -- ke ca
python3 he thong khong co pip -- vi no duoc goi tu shell script dieu phoi pipeline,
ngoai moi venv.

Cau hinh (them vao .env, cung cho voi DEEPSEEK_API_KEY):
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=123456789

Neu thieu 1 trong 2 bien -> im lang bo qua (exit 0). Pipeline dang train 3 tieng khong
duoc phep chet chi vi khong gui duoc tin nhan.

Dung:
    python3 src/ops/notify.py "Buoc 1 xong: 1324 attempt"
    echo "noi dung dai" | python3 src/ops/notify.py --stdin
    python3 src/ops/notify.py --file outputs/report.html "Bao cao ket qua v3"
"""
from __future__ import annotations

import json
import mimetypes
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # file nam trong src/<package>/, can 3 cap moi toi goc du an
MAX_LEN = 4000  # gioi han cua Telegram la 4096, chua cho phan bo sung
MAX_CAPTION_LEN = 1024  # gioi han caption cua sendDocument (khac voi sendMessage)


def _read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def send(text: str) -> bool:
    env = _read_env()
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[notify] Chua cau hinh TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID -- bo qua.")
        return False

    if len(text) > MAX_LEN:
        text = text[: MAX_LEN - 20] + "\n...(da cat bot)"

    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with urllib.request.urlopen(url, data=data, timeout=20) as resp:
            ok = json.loads(resp.read().decode("utf-8")).get("ok", False)
        if not ok:
            print("[notify] Telegram tra ve ok=false")
        return bool(ok)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Khong raise: loi mang khong duoc phep giet pipeline.
        print(f"[notify] Gui that bai (bo qua): {e}")
        return False


def _multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Dung tay multipart/form-data -- KHONG dung thu vien `requests` (co chu dich,
    xem docstring dau file: script nay phai chay bang python3 he thong, khong co pip).
    """
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def send_document(file_path: str, caption: str = "") -> bool:
    """Gui 1 file (vd bao cao HTML) qua Telegram nhu document dinh kem.

    Cung fail-safe nhu send(): mat mang/thieu cau hinh -> in canh bao, tra ve
    False, KHONG raise -- goi tu shell dieu phoi khong duoc phep giet pipeline
    chi vi buoc gui bao cao that bai.
    """
    env = _read_env()
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[notify] Chua cau hinh TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID -- bo qua gui file.")
        return False

    path = Path(file_path)
    if not path.exists():
        print(f"[notify] File khong ton tai: {path}")
        return False

    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption[:MAX_CAPTION_LEN]

    body, content_type = _multipart_body(fields, "document", path)
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ok = json.loads(resp.read().decode("utf-8")).get("ok", False)
        if not ok:
            print("[notify] Telegram sendDocument tra ve ok=false")
        return bool(ok)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[notify] Gui file that bai (bo qua): {e}")
        return False


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        # python3 notify.py --file outputs/report.html "caption tuy chon"
        if len(sys.argv) < 3:
            print("[notify] Thieu duong dan file. Dung: notify.py --file <path> [caption]")
            return
        file_path = sys.argv[2]
        caption = " ".join(sys.argv[3:]).strip()
        send_document(file_path, caption)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--stdin":
        text = sys.stdin.read()
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("[notify] Khong co noi dung.")
        return
    send(text)


if __name__ == "__main__":
    main()
