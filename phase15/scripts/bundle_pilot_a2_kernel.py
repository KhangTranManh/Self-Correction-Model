"""Pack amendment P15-A2 into the one script Kaggle executes."""

from __future__ import annotations

import base64
import argparse
import io
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "phase15/kaggle/pilot_s_format_continuation_v1"
OUT = KERNEL / "phase15_kaggle_pilot_a2_standalone.py"
NAMES = (
    "phase15_kaggle_pilot_continuation.py",
    "pilot_episodes.py",
    "pilot_score.py",
    "continuation_spec.json",
    "episodes.json",
    "self_edit_gate.py",
    "vendor/src/__init__.py",
    "vendor/src/core/__init__.py",
    "vendor/src/core/schema.py",
    "vendor/src/data/__init__.py",
    "vendor/src/data/verifiers/__init__.py",
    "vendor/src/data/verifiers/math.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in NAMES:
            archive.write(KERNEL / name, name)
    blob = base64.b64encode(buffer.getvalue()).decode("ascii")
    lines = [blob[index:index + 120] for index in range(0, len(blob), 120)]
    source = "\n".join([
            "import base64, io, runpy, sys, zipfile",
            "from pathlib import Path",
            "ROOT = Path('/tmp/phase15_pilot_a2_bundle')",
            "ROOT.mkdir(parents=True, exist_ok=True)",
            "BLOB = (",
            *(f"    '{line}'" for line in lines),
            ")",
            "with zipfile.ZipFile(io.BytesIO(base64.b64decode(BLOB))) as archive:",
            "    archive.extractall(ROOT)",
            "sys.path[:0] = [str(ROOT), str(ROOT / 'vendor')]",
            "runpy.run_path(str(ROOT / 'phase15_kaggle_pilot_continuation.py'), run_name='__main__')",
            "",
        ])
    if args.stdout:
        sys.stdout.reconfigure(encoding="utf-8")
        print(source, end="")
        return
    OUT.write_text(source, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} bytes {OUT.stat().st_size}")


if __name__ == "__main__":
    main()
