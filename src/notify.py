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
    python3 src/notify.py "Buoc 1 xong: 1324 attempt"
    echo "noi dung dai" | python3 src/notify.py --stdin
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MAX_LEN = 4000  # gioi han cua Telegram la 4096, chua cho phan bo sung


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


def main() -> None:
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
