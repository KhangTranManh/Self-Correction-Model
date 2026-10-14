"""One loader for every Phase 14 artifact on this machine (no GPU, nothing is regenerated).

    import sys; sys.path.insert(0, "D:/AGI")
    from paper.scripts import phase14_data as P
    P.cells()                                   # every (split, model, dataset) with outputs
    rows  = P.problems("asdiv")                 # frozen problems + gold (hash-checked)
    s     = P.samples("llama31_8b", "asdiv")    # {(pid, k): row}, k = 1..5
    j     = P.judgments("llama31_8b", "asdiv", "base_c")   # {(pid, k, order): row}
    ok    = P.correct("llama31_8b", "asdiv")    # {(pid, k): bool}, project parser ("lenient")
    pairs = P.one_right_pairs("llama31_8b", "asdiv")       # [(pid, k, first_is_right)]
    h     = P.hidden("llama31_8b", "asdiv", "judge")       # numpy NpzFile (lazy)
    rep   = P.report(); probes = P.probe_report()

Locations (see phase14/DATA_CATALOG.md): text outputs are the verified mirror
outputs/phase14_remote/outputs/phase14_v1 (synced from the GPU host with sha256 checks);
hidden-state arrays live in F:/AGI_data/phase14_hidden (D:/AGI/outputs/phase14_hidden is a
junction to it). Every stage is read through phase14.scripts.generate.completed(), which refuses
a stage whose answers.jsonl no longer matches the sha256 in its summary.json.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _to_sympy  # noqa: E402
from phase14.scripts import generate as _g  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402
from phase14.scripts.config import JUDGED  # noqa: E402

MIRROR = ROOT / "outputs/phase14_remote/outputs/phase14_v1"
HIDDEN = Path("F:/AGI_data/phase14_hidden")
if not HIDDEN.exists():
    HIDDEN = ROOT / "outputs/phase14_hidden"
_g.OUT = MIRROR  # every generate.folder()/completed() call reads the local mirror

# The bridge and regression sets reuse Phase 13/12 outputs; point their loaders at the local mirrors.
for _mod, _cands in (("phase13.scripts.generate", ("outputs/phase13_v1", "outputs/phase13_remote_v100/outputs/phase13_v1")),
                     ("phase12.scripts.generate", ("outputs/phase12_v1", "outputs/phase12_remote_v100/outputs/phase12_v1"))):
    _m = __import__(_mod, fromlist=["OUT"])
    _found = next((ROOT / c for c in _cands if (ROOT / c / "samples/summary.json").exists()), None)
    if _found is not None:
        _m.OUT = _found


def cells(split: str = "main") -> list[tuple[str, str]]:
    """(model, dataset) pairs that have a complete samples or judge stage in the mirror."""
    out = []
    for d in sorted((MIRROR / split).glob("*/*")):
        if d.is_dir() and any((d / s / "summary.json").exists() for s in ("samples", "judge_base", "judge_base_c")):
            out.append((d.parent.name, d.name))
    return out


@lru_cache(maxsize=None)
def problems(dataset: str, split: str = "main") -> tuple[dict, ...]:
    return tuple(_g.problems(dataset, split))


@lru_cache(maxsize=None)
def samples(model: str, dataset: str, split: str = "main") -> dict:
    return _g.load_samples(model, dataset, split)


@lru_cache(maxsize=None)
def judgments(model: str, dataset: str, judge: str, split: str = "main") -> dict:
    """judge in {"base", "base_c", "base_1536", "base_c_1536"}."""
    return _g.completed(_g.folder(model, dataset, split), f"judge_{judge}")


def settings(model: str, dataset: str, stage: str, split: str = "main") -> dict:
    """Decoding settings recorded when the stage ran (model, revision, temperature, caps, batch, GPU)."""
    path = _g.folder(model, dataset, split) / stage / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")).get("settings", {})


@lru_cache(maxsize=None)
def correct(model: str, dataset: str, split: str = "main", mode: str = "lenient") -> dict:
    """{(pid, k): verifier-correct} for the primary (768-token) view; mode lenient|strict|rescued."""
    ref = {r["id"]: _to_sympy(str(r["reference_answer"])) for r in problems(dataset, split)}
    return {key: same(parse(row["output"], mode), ref[key[0]]) for key, row in samples(model, dataset, split).items()}


def one_right_pairs(model: str, dataset: str, split: str = "main") -> list[tuple[str, int, bool]]:
    """Judged pairs (s1, sk) whose final answers differ and exactly one is right: (pid, k, s1_is_right)."""
    s, ok = samples(model, dataset, split), correct(model, dataset, split)
    out = []
    for row in problems(dataset, split):
        pid = row["id"]
        p1 = parse(s[(pid, 1)]["output"])
        for k in JUDGED:
            if same(p1, parse(s[(pid, k)]["output"])) or ok[(pid, 1)] == ok[(pid, k)]:
                continue
            out.append((pid, k, ok[(pid, 1)]))
    return out


def hidden(model: str, dataset: str, site: str, split: str = "main"):
    """Hidden-state arrays (numpy NpzFile, read lazily). site in {answer, judge, direct}."""
    import numpy as np
    return np.load(HIDDEN / split / model / dataset / f"{site}.npz")


def report() -> dict:
    """Protected analysis (endpoints opened 2026-10-13 03:55 UTC, plus cells added later)."""
    return json.loads((MIRROR / "main/analysis/report.json").read_text(encoding="utf-8"))


def probe_report() -> dict:
    return json.loads((MIRROR / "main/analysis/probe_report.json").read_text(encoding="utf-8"))
