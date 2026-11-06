"""Pack the pilot kernel into the one script Kaggle actually executes."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "phase15/kaggle/pilot_v1"
OUT = KERNEL / "phase15_kaggle_pilot_standalone.py"
NAMES = (
    "phase15_kaggle_pilot.py",
    "pilot_episodes.py",
    "pilot_score.py",
    "pilot_spec.json",
    "self_edit_gate.py",
    "screen.jsonl",
    "reserve.jsonl",
    "vendor/src/__init__.py",
    "vendor/src/core/__init__.py",
    "vendor/src/core/schema.py",
    "vendor/src/data/__init__.py",
    "vendor/src/data/verifiers/__init__.py",
    "vendor/src/data/verifiers/math.py",
)


def main() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in NAMES:
            archive.write(KERNEL / name, name)
    blob = base64.b64encode(buffer.getvalue()).decode("ascii")
    lines = [blob[index:index + 120] for index in range(0, len(blob), 120)]
    OUT.write_text(
        "\n".join([
            "import base64, io, runpy, sys, zipfile",
            "from pathlib import Path",
            "ROOT = Path('/tmp/phase15_pilot_bundle')",
            "ROOT.mkdir(parents=True, exist_ok=True)",
            "BLOB = (",
            *(f"    '{line}'" for line in lines),
            ")",
            "with zipfile.ZipFile(io.BytesIO(base64.b64decode(BLOB))) as archive:",
            "    archive.extractall(ROOT)",
            "sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]",
            "runpy.run_path(str(ROOT / 'phase15_kaggle_pilot.py'), run_name='__main__')",
            "",
        ]),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT} bytes {OUT.stat().st_size}")


if __name__ == "__main__":
    main()
