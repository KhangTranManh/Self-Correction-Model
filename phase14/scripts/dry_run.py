"""CPU dry run of the whole Phase 14 pipeline on fake generations (no GPU, no model).

    python phase14/scripts/dry_run.py --workdir <empty folder outside the repo>

1. Copies the code, frozen data and (if present) the Phase 12/13 output mirrors
   into --workdir, so nothing under the real outputs/ or phase14/data/ changes.
2. Inside the copy, replaces the two GPU steps with fakes that write exactly the
   files they would write:
     generate.py      samples/ and judge_<j>/ (answers.jsonl + summary.json),
                      with planted solve accuracy, answer-as-sentence rate,
                      judge skill, position bias and invented-answer rate;
                      base_c rows carry verdict log-probabilities
     extract_hidden   hidden/<split>/<model>/<dataset>/<site>.npz (small width),
                      with a right/wrong signal planted at known layers
   The regression-only base judgments are imported with the real generate.py
   (CPU only), from the real Phase 13/12 outputs when they are present.
3. Runs every other step with the real scripts, in run_remote.sh order: pilot
   lock, pilot report (--write), main lock, main analysis, probes, report tables,
   then the comparison step (regression, --add-comparison, probes, tables).
4. Checks that the planted values are recovered (skill sign, bias direction,
   probe layer band, verdict agreement) and prints a summary.

Fake data use labels freely; this is a test harness, never a result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]

COPY_DIRS = ("phase1/src", "phase4/lib", "phase5/scripts", "phase7/scripts", "phase8/scripts",
             "phase9/scripts", "phase11/scripts", "phase12/scripts", "phase12/data/sources_v1",
             "phase13/scripts", "phase13/data/sources_v1", "phase14/docs", "phase14/scripts",
             "phase14/data/raw", "phase14/data/sources_v1")
COPY_FILES = ("phase14/data/gsm8k13_audit_excluded.json",)
MIRRORS = {"outputs/phase13_v1": ("outputs/phase13_remote_v100/outputs/phase13_v1", "outputs/phase13_v1"),
           "outputs/phase12_v1": ("outputs/phase12_remote_v100/outputs/phase12_v1", "outputs/phase12_v1")}
MIRROR_STAGES = {"outputs/phase13_v1": ("samples", "judge_base", "judge_base_c"),
                 "outputs/phase12_v1": ("samples", "judge_base")}

# Planted per-model behaviour: solve accuracy, sentence-answer rate, judge skill
# (logit), position bias (logit toward A; < 0 = toward B), invented rate.
PLANT = {"llama31_8b": (0.62, 0.08, 1.2, -0.6, 0.10),
         "mistral_7b": (0.50, 0.03, 0.8, 0.5, 0.15),
         "original_solver": (0.66, 0.01, 1.0, -0.4, 0.15),
         "qwen25_instruct": (0.80, 0.05, 1.6, 0.6, 0.08),
         "phi35_mini": (0.55, 0.02, 0.6, 0.0, 0.10),
         "gemma2_9b": (0.70, 0.02, 1.0, 0.0, 0.10)}
WIDTH, LAYERS = 24, 29
JUDGE_LAYERS, ANSWER_LAYERS, LENS_FROM = range(16, 23), range(18, 25), 22


# ---------------------------------------------------------------- orchestrator (repo side)
def prepare(workdir: Path) -> None:
    if workdir.exists() and any(workdir.iterdir()):
        raise SystemExit(f"--workdir must be empty or absent: {workdir}")
    if workdir.resolve().is_relative_to(REPO):
        raise SystemExit("--workdir must be outside the repository")
    for rel in COPY_DIRS:
        src = REPO / rel
        if src.exists():
            shutil.copytree(src, workdir / rel, ignore=shutil.ignore_patterns("__pycache__"))
    for rel in COPY_FILES:
        if (REPO / rel).exists():
            (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, workdir / rel)
    for target, candidates in MIRRORS.items():
        src = next((REPO / c for c in candidates if (REPO / c / "samples/summary.json").exists()), None)
        if src is None:
            print(f"[dry-run] no local mirror for {target}; comparison steps will be skipped", flush=True)
            continue
        for stage in MIRROR_STAGES[target]:
            if (src / stage).exists():
                shutil.copytree(src / stage, workdir / target / stage)
    for pkg in ("phase4", "phase5", "phase7", "phase8", "phase9", "phase11", "phase12", "phase13", "phase14",
                "phase4/lib", "phase5/scripts", "phase7/scripts", "phase8/scripts", "phase9/scripts",
                "phase11/scripts", "phase12/scripts", "phase13/scripts", "phase14/scripts"):
        init = workdir / pkg / "__init__.py"
        if (workdir / pkg).exists() and not init.exists():
            init.write_text("", encoding="utf-8")


def run(workdir: Path, *args: str, log: str | None = None, allow_fail: bool = False) -> str:
    env = {**os.environ, "PYTHONPATH": str(workdir), "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, *args]
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, encoding="utf-8")
    if log:
        (workdir / "logs").mkdir(exist_ok=True)
        (workdir / "logs" / log).write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 and not allow_fail:
        print(proc.stdout[-3000:], proc.stderr[-6000:], sep="\n")
        raise SystemExit(f"[dry-run] FAILED: {' '.join(args)}")
    return proc.stdout


def orchestrate(workdir: Path, order: str = "remote", endpoints_only: bool = False) -> None:
    prepare(workdir)
    me = "phase14/scripts/dry_run.py"
    cfg = json.loads(run(workdir, me, "_config"))
    have_reports = all((workdir / m / "samples/summary.json").exists() for m in MIRRORS)
    step = lambda name: print(f"=== {name}", flush=True)  # noqa: E731

    step("pilot lock: freeze, verify")
    run(workdir, "phase14/scripts/verify_lock.py", "--stage", "pilot", "--freeze")
    run(workdir, "phase14/scripts/verify_lock.py", "--stage", "pilot")
    step("pilot: fake samples and judges")
    for model in cfg["pilot_models"]:
        for dataset in cfg["datasets"]:
            run(workdir, me, "_fake_cell", model, dataset, "pilot")
    run(workdir, "phase14/scripts/verify_lock.py", "--stage", "pilot")
    step("pilot report --write")
    print(run(workdir, "phase14/scripts/pilot_report.py", "--write", log="pilot_report.txt"), flush=True)
    run(workdir, "phase14/scripts/verify_lock.py", "--stage", "main", "--freeze")
    run(workdir, "phase14/scripts/verify_lock.py", "--stage", "main")

    step("main: fake samples, judges, hidden states (uniform cells)")
    for model in cfg["run_models"]:
        for dataset in (*cfg["datasets"], *([] if endpoints_only else cfg["comparison"])):
            run(workdir, me, "_fake_cell", model, dataset, "main")
            run(workdir, me, "_fake_hidden", model, dataset, "main")
    run(workdir, "phase14/scripts/verify_lock.py", "--stage", "main")
    if endpoints_only:
        have_reports = False  # main run limited to the endpoint cells (amendment 2026-10-13)
    if order == "sequential" and have_reports:
        # run_sequential.sh: each model's comparison generations happen in its own main
        # block, so the comparison cells already exist when the endpoints are opened.
        step("sequential order: comparison generations before the opening")
        for dataset in ("gsm8k13_report", "svamp12_report"):
            run(workdir, "phase14/scripts/generate.py", "--model", "original_solver", "--dataset", dataset,
                "--split", "main", "--stage", "judge", "--judge", "base")
            run(workdir, me, "_fake_cell", "original_solver", dataset, "main")
        for model in cfg["run_models"]:
            if model != "original_solver":
                run(workdir, me, "_fake_cell", model, cfg["bridge"], "main")
                run(workdir, me, "_fake_hidden", model, cfg["bridge"], "main")
    step("main analysis (protected opening)")
    run(workdir, "phase14/scripts/analyze.py", log="analysis.txt")
    step("probe analysis")
    run(workdir, "phase14/scripts/probe_analysis.py", log="probe.txt")
    step("report tables")
    run(workdir, "phase14/scripts/report_tables.py", log="report_tables_main.md")

    if have_reports and order == "sequential":
        step("regression check (sequential order: comparison cells scored in the opening)")
        out = run(workdir, "phase14/scripts/regression.py", log="regression.txt", allow_fail=True)
        repro = json.loads(out[out.index("{"):out.rindex("}") + 1])["report_reproduction"]
        bad = [k for k, v in repro.items() if v.get("base", {}).get("status") != "reproduced"]
        if bad:
            raise SystemExit(f"[dry-run] FAILED: report base judgments not reproduced: {bad}")
        report = json.loads((workdir / "outputs/phase14_v1/main/analysis/report.json").read_text(encoding="utf-8"))
        missing = [k for k in report["deferred_cells"]]
        print(f"[dry-run] base reproduced ({', '.join(repro)}); cells still deferred: {missing}", flush=True)
    elif have_reports:
        step("comparison: regression sets and bridge")
        for dataset in ("gsm8k13_report", "svamp12_report"):
            run(workdir, "phase14/scripts/generate.py", "--model", "original_solver", "--dataset", dataset,
                "--split", "main", "--stage", "judge", "--judge", "base")          # real import, CPU only
            run(workdir, me, "_fake_cell", "original_solver", dataset, "main")    # base_c re-run (fake)
        for model in cfg["run_models"]:
            if model != "original_solver":
                run(workdir, me, "_fake_cell", model, cfg["bridge"], "main")
                run(workdir, me, "_fake_hidden", model, cfg["bridge"], "main")
        # The base_c re-run on the report's pairs is fake here, so its mismatch is expected;
        # the imported base judgments are real and must reproduce the report exactly.
        out = run(workdir, "phase14/scripts/regression.py", log="regression.txt", allow_fail=True)
        repro = json.loads(out[out.index("{"):out.rindex("}") + 1])["report_reproduction"]
        bad = [k for k, v in repro.items() if v.get("base", {}).get("status") != "reproduced"]
        if bad:
            raise SystemExit(f"[dry-run] FAILED: report base judgments not reproduced: {bad}")
        print("[dry-run] report base judgments reproduced on the real Phase 13/12 outputs: "
              + ", ".join(repro), flush=True)
        run(workdir, "phase14/scripts/analyze.py", "--add-comparison", log="comparison.txt")
        run(workdir, "phase14/scripts/probe_analysis.py", log="probe_after_comparison.txt")
        run(workdir, "phase14/scripts/report_tables.py", log="report_tables_final.md")
    else:
        step("comparison skipped (no Phase 12/13 mirror)")

    step("checks")
    print(run(workdir, me, "_check"), flush=True)
    print(f"[dry-run] complete; logs in {workdir / 'logs'}", flush=True)


# ---------------------------------------------------------------- fakes (run inside the copy)
def _imports():
    sys.path.insert(0, str(REPO / "phase1"))
    sys.path.insert(0, str(REPO))
    from phase14.scripts import config, generate
    return config, generate


def _rng(*parts):
    import hashlib
    import numpy as np
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def _write_stage(path: Path, rows: list[dict], sha256) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "audit.jsonl").write_text("", encoding="utf-8")
    answers = path / "answers.jsonl"
    answers.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                       encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(rows), "audit_sha256": sha256(path / "audit.jsonl"),
               "answers_sha256": sha256(answers), "settings": {"fake": True, "dry_run": True}}
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")


def _answer_text(value: str, sentence: bool) -> str:
    if sentence:
        return f"Lời giải ngắn.\nĐáp số: Kết quả là {value} đơn vị."
    return f"Lời giải ngắn.\nĐáp số: {value}"


def fake_cell(model: str, dataset: str, split: str) -> None:
    import numpy as np
    config, gen = _imports()
    acc, sentence, skill, bias, invent = PLANT[model]
    rows = gen.problems(dataset, split)
    base = gen.folder(model, dataset, split)
    reuse = gen.reused(model, dataset)
    if not reuse and not (base / "samples/summary.json").exists():
        out = []
        for i, row in enumerate(rows):
            r = _rng("s", model, dataset, split, row["id"])
            for k in range(1, config.SAMPLES + 1):
                right = r.random() < acc
                value = str(row["reference_answer"]) if right else str(int(r.integers(1, 4)) * 1000 + 7)
                text = _answer_text(value, bool(right and r.random() < sentence and value.isdigit()))
                cap = r.random() < 0.03
                out.append({"task": [row["id"], k], "index": len(out), "request_seed": 0,
                            "generated_tokens": 1536 if cap else 300, "hit_token_cap": cap,
                            "output_full": text, "output": text, "hit_primary_cap": cap})
        _write_stage(base / "samples", out, gen.sha256)
    samples = gen.load_samples(model, dataset, split)
    tasks = gen.judge_tasks(rows, samples)
    ref = {row["id"]: row["reference_answer"] for row in rows}
    from phase14.scripts.answers import parse, same
    from src.data.verifiers.math import _to_sympy
    gold = {pid: _to_sympy(str(v)) for pid, v in ref.items()}
    for judge in ("base", "base_c"):
        path = base / f"judge_{judge}"
        if (path / "summary.json").exists():
            continue
        out = []
        for index, (pid, k, order) in enumerate(tasks):
            r = _rng("j", judge, model, dataset, split, pid, k, order)
            first, other = samples[(pid, 1)]["output"], samples[(pid, k)]["output"]
            shown_a, shown_b = (first, other) if order == "ab" else (other, first)
            a_ok, b_ok = same(parse(shown_a), gold[pid]), same(parse(shown_b), gold[pid])
            signal = skill * ((1 if a_ok else 0) - (1 if b_ok else 0))
            m = bias + signal + r.normal(0, 1.0)
            pick = "A" if m > 0 else "B"
            if judge == "base" and r.random() < invent:
                text = "Kiểm tra.\nĐáp số: 424242"
            else:
                chosen = shown_a if pick == "A" else shown_b
                text = "Kiểm tra.\nĐáp số: " + str(parse(chosen))
            row = {"task": [pid, k, order], "index": index, "request_seed": 0,
                   "generated_tokens": 200, "hit_token_cap": False}
            if judge == "base_c":
                text += f"\nVerdict: Solution {pick}"
                lp = None if r.random() < 0.02 else {
                    "position": 40, "A": float(-np.log1p(np.exp(-m))), "B": float(-np.log1p(np.exp(m)))}
                row["verdict_logprobs"] = lp
            row["output"] = text
            out.append(row)
        _write_stage(path, out, gen.sha256)
    print(json.dumps({"fake_cell": f"{model}/{dataset}/{split}", "judged_tasks": len(tasks)}))


def fake_hidden(model: str, dataset: str, split: str) -> None:
    import numpy as np
    config, gen = _imports()
    from phase14.scripts.extract_hidden import destination
    from phase14.scripts.answers import parse, same
    from src.data.verifiers.math import _to_sympy
    rows = gen.problems(dataset, split)
    samples = gen.load_samples(model, dataset, split)
    base = gen.folder(model, dataset, split)
    gold = {row["id"]: _to_sympy(str(row["reference_answer"])) for row in rows}
    ok = lambda pid, k: same(parse(samples[(pid, k)]["output"]), gold[pid])  # noqa: E731
    r = _rng("h", model, dataset, split)
    direction = _rng("dir", model).normal(size=WIDTH)

    def states(sign: float, band) -> np.ndarray:
        h = r.normal(size=(LAYERS, WIDTH))
        for layer in band:
            h[layer] += 0.9 * sign * direction
        return h.astype(np.float16)

    tasks, hidden, lp = [], [], []
    for row in rows:
        for k in (1, *config.JUDGED):
            c = ok(row["id"], k)
            tasks.append(f"{row['id']}|{k}")
            hidden.append(states(1.0 if c else -1.0, ANSWER_LAYERS))
            lp.append(-0.5 + 0.2 * c + r.normal(0, 0.3))
    sites = {"answer": {"tasks": np.array(tasks), "hidden": np.stack(hidden), "mean_logprob": np.array(lp),
                        "answer_tokens": np.full(len(tasks), 300), "prefix_ok": np.ones(len(tasks), bool)}}
    judged = {j: gen.completed(base, f"judge_{j}") for j in ("base", "base_c")}
    site = {}
    for j, items in judged.items():
        t, h = [], []
        for pid, k, order in sorted(items):
            a_first = order == "ab"
            first_ok, other_ok = ok(pid, 1), ok(pid, k)
            b_ok = other_ok if a_first else first_ok
            a_ok = first_ok if a_first else other_ok
            t.append(f"{pid}|{k}|{order}")
            h.append(states(float(b_ok) - float(a_ok), JUDGE_LAYERS))
        site[f"{j}_tasks"], site[f"{j}_hidden"] = np.array(t), np.stack(h)
    sites["judge"] = site
    t, ours, theirs, final, lens = [], [], [], [], []
    for (pid, k, order), row in sorted(judged["base_c"].items()):
        lpv = row.get("verdict_logprobs") or {}
        m = (lpv.get("A") or -0.7) - (lpv.get("B") or -0.7)
        if "Verdict: Solution" in row["output"]:
            t.append(f"{pid}|{k}|{order}")
            ours.append([(lpv.get("A") or -0.7) + r.normal(0, 0.01), (lpv.get("B") or -0.7) + r.normal(0, 0.01)])
            theirs.append([lpv.get("A") if lpv.get("A") is not None else np.nan,
                           lpv.get("B") if lpv.get("B") is not None else np.nan])
        final.append([-np.log1p(np.exp(-0.8 * m)), -np.log1p(np.exp(0.8 * m))])
        lens.append([0.0 if layer < LENS_FROM else m * (layer - LENS_FROM + 1) / (LAYERS - LENS_FROM)
                     for layer in range(LAYERS)])
    sites["verdict"] = {"tasks": np.array(t), "transformers_logprobs": np.array(ours, float),
                        "vllm_logprobs": np.array(theirs, float)}
    sites["direct"] = {"tasks": np.array([f"{p}|{k}|{o}" for p, k, o in sorted(judged["base_c"])]),
                       "final_logprobs": np.array(final, float), "lens_margin": np.array(lens, np.float32)}
    for name, data in sites.items():
        out = destination(split, model, dataset, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **data)
        out.with_suffix(".json").write_text(json.dumps({"status": "complete", "fake": True}) + "\n", encoding="utf-8")
    print(json.dumps({"fake_hidden": f"{model}/{dataset}/{split}"}))


def check() -> None:
    config, _ = _imports()
    report = json.loads((config.OUT / "main/analysis/report.json").read_text(encoding="utf-8"))
    lines, problems = [], []
    for key, cell in sorted(report["cells"].items()):
        model = key.split("/")[0]
        if model not in PLANT:
            continue
        b = cell["lenient"]["judges"]["base"]
        bias = PLANT[model][3]
        p2 = b.get("position_2_rate")
        lines.append(f"{key:40s} pairs {b.get('one_right_pairs', cell['lenient'].get('one_right_pairs'))!s:>5} "
                     f"skill {b['skill_index']:+.3f} p2 {p2:.3f} tie {b['tie_break']:.3f}")
        if b["skill_index"] <= 0:
            problems.append(f"{key}: planted positive skill not recovered")
        if bias and p2 is not None and (p2 - 0.5) * (-bias) < 0:
            problems.append(f"{key}: planted bias direction not recovered (p2 {p2:.3f}, bias {bias})")
    for name in ("E1_skill", "E2_debias"):
        s = report["summary"][name]
        lines.append(f"{name}: {s['count']}/{s['of']} cells pass -> {s['verdict']}")
    lines.append("reproduction: " + json.dumps(report.get("report_reproduction"))[:600])
    print("\n".join(lines))
    print("PROBLEMS:\n- " + "\n- ".join(problems) if problems else "planted values recovered: OK")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        cmd, args = sys.argv[1], sys.argv[2:]
        if cmd == "_config":
            config, _ = _imports()
            print(json.dumps({"run_models": list(config.RUN_MODELS),
                              "pilot_models": list(getattr(config, "PILOT_MODELS", config.RUN_MODELS)),
                              "datasets": list(config.DATASETS),
                              "comparison": list(config.COMPARISON), "bridge": config.BRIDGE}))
        elif cmd == "_fake_cell":
            fake_cell(*args)
        elif cmd == "_fake_hidden":
            fake_hidden(*args)
        elif cmd == "_check":
            check()
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--order", choices=("remote", "sequential"), default="remote",
                        help="remote: run_remote.sh order; sequential: run_sequential.sh order")
    parser.add_argument("--endpoints-only", action="store_true",
                        help="main run with the endpoint cells only (no GSM8K/SVAMP/bridge/regression)")
    a = parser.parse_args()
    orchestrate(a.workdir.resolve(), a.order, a.endpoints_only)


if __name__ == "__main__":
    main()
