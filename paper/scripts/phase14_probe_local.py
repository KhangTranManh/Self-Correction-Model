"""Run phase14/scripts/probe_analysis.py on this machine (CPU), from the local mirror and the hidden
states on F:, without touching the mirrored probe_report.json. Use when the host's probe step did not
finish before the GPU was released. Output: paper/results/phase14_probe_local.json (+ printed summary).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import phase14_data as P  # noqa: E402,F401  (points generate at the local mirror)
from phase14.scripts import probe_analysis as pa  # noqa: E402

pa.OUT = P.MIRROR  # report.json check reads the mirror
pa.destination = lambda split, model, dataset, site: P.HIDDEN / split / model / dataset / f"{site}.npz"
pa.REPORT = ROOT / "paper/results/phase14_probe_local.json"
pa.main()
