"""Thu tin nhan Telegram tu MOT chat duy nhat vao outputs/tg_inbox.jsonl.

Cung ly do nhu notify.py: chi dung stdlib, tu doc .env, de chay duoc bang python3 he
thong ngoai moi venv.

VAI TRO TRONG LUONG LAM VIEC: day KHONG phai bot tu dong thuc thi lenh. No chi la
hop thu -- ghi lai nhung gi ban nhan trong luc khong ai ngoi truoc may. Khi phien
Claude Code duoc goi, tin nhan trong file nay duoc doc va xu ly. Tin nhan gui luc
khong co phien nao chay se nam cho, khong bien mat.

CO Y KHONG tu chay lenh tu Telegram: mot bot cam quyen shell tren may GPU, dieu khien
bang tin nhan, la thu rat kho kiem soat khi co su co. Doc-roi-nguoi-quyet an toan hon
nhieu ma van giai quyet dung nhu cau "khong ngoi truoc may".

Loc theo TELEGRAM_CHAT_ID: tin tu chat khac bi bo qua (bot cua ban ai cung nhan ra
duoc neu biet ten, nhung chi chat cua ban moi duoc ghi vao hop thu).

Dung:
    python3 src/tg_inbox.py            # poll 1 lan roi thoat
    python3 src/tg_inbox.py --watch    # poll lien tuc (chay nen)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INBOX_PATH = ROOT_DIR / "outputs" / "tg_inbox.jsonl"
OFFSET_PATH = ROOT_DIR / "outputs" / ".tg_offset"
POLL_SECONDS = 30


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


def _get_updates(token: str, offset: int) -> list[dict]:
    params = urllib.parse.urlencode({"offset": offset, "timeout": 0})
    url = f"https://api.telegram.org/bot{token}/getUpdates?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        print(f"[tg_inbox] Loi mang (bo qua vong nay): {e}")
        return []
    return payload.get("result", []) if payload.get("ok") else []


def poll_once(token: str, chat_id: str) -> int:
    offset = 0
    if OFFSET_PATH.exists():
        try:
            offset = int(OFFSET_PATH.read_text().strip() or 0)
        except ValueError:
            offset = 0

    updates = _get_updates(token, offset)
    if not updates:
        return 0

    INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(INBOX_PATH, "a", encoding="utf-8") as f:
        for upd in updates:
            offset = max(offset, upd["update_id"] + 1)
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            # Chi nhan tu dung chat da cau hinh.
            if str(msg.get("chat", {}).get("id")) != str(chat_id):
                continue
            text = msg.get("text")
            if not text:
                continue
            f.write(
                json.dumps(
                    {
                        "ts": datetime.fromtimestamp(
                            msg.get("date", 0), tz=timezone.utc
                        ).isoformat(),
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    OFFSET_PATH.write_text(str(offset), encoding="utf-8")
    if written:
        print(f"[tg_inbox] Da ghi {written} tin nhan vao {INBOX_PATH}")
    return written


def main() -> None:
    env = _read_env()
    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[tg_inbox] Chua cau hinh TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID trong .env")
        sys.exit(1)

    watch = "--watch" in sys.argv
    if not watch:
        poll_once(token, chat_id)
        return

    print(f"[tg_inbox] Watch mode, poll moi {POLL_SECONDS}s. Ghi vao {INBOX_PATH}")
    while True:
        poll_once(token, chat_id)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
