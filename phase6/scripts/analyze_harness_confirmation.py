"""Analyze the fresh non-protected Phase 6 router-to-repair confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase6/configs/harness_confirmation_v1.yaml"
HOLDOUT = ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl"
HOLDOUT_LOCK = ROOT / "phase6/data/harness_confirmation_holdout_v1_lock.json"
ROUTES = ROOT / "phase6/data/harness_confirmation_routes_v1.jsonl"
ROUTES_LOCK = ROOT / "phase6/data/harness_confirmation_routes_v1_lock.json"
RAW = ROOT / "outputs/phase6_harness_confirmation_v1/recheck"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
DISPLAY = {"original_solver": "Original solver", "warmstart_v2": "Warm-start V2",
           "correction_sft_v3": "Correction SFT V3"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def fmt_rate(value: float | None) -> str:
    """Render an undefined rate explicitly rather than treating it as zero."""
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase6_harness_confirmation_v1")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    holdout_lock = json.loads(HOLDOUT_LOCK.read_text(encoding="utf-8"))
    for relative, expected in holdout_lock["files"].items():
        if sha256_lf(ROOT / relative) != expected["sha256_lf"]:
            raise RuntimeError(f"Frozen holdout input changed: {relative}")
    routes_lock = json.loads(ROUTES_LOCK.read_text(encoding="utf-8"))
    for relative, expected in routes_lock["files"].items():
        if relative in {ROUTES.relative_to(ROOT).as_posix(), CONFIG.relative_to(ROOT).as_posix(),
                        HOLDOUT.relative_to(ROOT).as_posix(), HOLDOUT_LOCK.relative_to(ROOT).as_posix()}:
            if sha256_lf(ROOT / relative) != expected["sha256_lf"]:
                raise RuntimeError(f"Frozen route input changed: {relative}")
    rows = read_jsonl(HOLDOUT)
    by_id = {row["problem_id"]: row for row in rows}
    routes = read_jsonl(ROUTES)
    report: dict[str, Any] = {
        "schema_version": "phase6_harness_confirmation_report_v1",
        "scope": {"rows": len(rows), "initially_correct": sum(r["initial_correct"] for r in rows),
                  "initially_wrong": sum(not r["initial_correct"] for r in rows),
                  "protected_data_used": False},
        "input_hashes": {"config": sha256_lf(CONFIG), "holdout": sha256_lf(HOLDOUT),
                         "routes": sha256_lf(ROUTES)},
        "checkpoints": {},
    }
    audit: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        raw = read_jsonl(RAW / f"recheck_{checkpoint}.jsonl")
        for condition in ("self_confidence", "frozen_probe", "oracle_known_wrong"):
            route_ids = {row["problem_id"] for row in routes
                         if row["checkpoint"] == checkpoint and row["condition"] == condition}
            outputs = {row["problem_id"]: row for row in raw if row["condition"] == condition}
            if set(outputs) != route_ids:
                raise RuntimeError(f"Route/output mismatch {checkpoint}/{condition}")
            routed = [by_id[pid] for pid in route_ids]
            if condition == "oracle_known_wrong":
                if any(row["initial_correct"] for row in routed):
                    raise RuntimeError("Oracle route contains an initially correct row")
                repaired = sum(outputs[pid]["final_correct"] for pid in route_ids)
                result = {
                    "routed_known_wrong": len(route_ids), "verified_repairs": repaired,
                    "verified_repair_rate": ratio(repaired, len(route_ids)),
                }
            else:
                tp = sum(not row["initial_correct"] for row in routed)
                fp = sum(row["initial_correct"] for row in routed)
                fn = sum(not row["initial_correct"] for pid, row in by_id.items() if pid not in route_ids)
                tn = sum(row["initial_correct"] for pid, row in by_id.items() if pid not in route_ids)
                final_by_id = {pid: (outputs[pid]["final_correct"] if pid in outputs else by_id[pid]["initial_correct"])
                               for pid in by_id}
                w2c = sum((not by_id[pid]["initial_correct"]) and final_by_id[pid] for pid in by_id)
                c2w = sum(by_id[pid]["initial_correct"] and not final_by_id[pid] for pid in by_id)
                tp_repaired = sum(outputs[pid]["final_correct"] for pid in route_ids
                                  if not by_id[pid]["initial_correct"])
                result = {
                    "routed": len(route_ids), "true_positive": tp, "false_positive": fp,
                    "false_negative": fn, "true_negative": tn,
                    "wrong_recall": ratio(tp, tp + fn), "precision_wrong": ratio(tp, tp + fp),
                    "correct_preservation": ratio(tn, tn + fp),
                    "verified_repairs_among_true_positive_routes": tp_repaired,
                    "repair_rate_given_true_positive_route": ratio(tp_repaired, tp),
                    "wrong_to_correct": w2c, "correct_to_wrong": c2w,
                    "final_accuracy": ratio(sum(final_by_id.values()), len(final_by_id)),
                }
            report.setdefault("checkpoints", {}).setdefault(checkpoint, {})[condition] = result
            for pid in route_ids:
                audit.append({"checkpoint": checkpoint, "condition": condition, "problem_id": pid,
                              "initial_correct": bool(by_id[pid]["initial_correct"]),
                              "final_correct": bool(outputs[pid]["final_correct"]),
                              "route_score_probability_wrong": outputs[pid]["route_score_probability_wrong"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "analysis_audit.jsonl"
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in audit), encoding="utf-8", newline="\n")
    report["analysis_audit_sha256_lf"] = sha256_lf(audit_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = ["# Phase 6 fresh harness confirmation", "",
             "Fresh 40-source balanced holdout, disjoint from every Phase 5 split and all earlier Phase 6 pilots. "
             "The source pool, final holdout, thresholds, and routes were frozen before recheck generation. "
             "No protected data was used.", "",
             "## End-to-end non-oracle routers", "",
             "| Checkpoint | Router | Routed | Wrong recall | Precision | Correct preservation | W->C | C->W | Final accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        for condition, label in (("self_confidence", "Self confidence"), ("frozen_probe", "Frozen probe")):
            m = report["checkpoints"][checkpoint][condition]
            lines.append(f"| {DISPLAY[checkpoint]} | {label} | {m['routed']}/40 | {fmt_rate(m['wrong_recall'])} | "
                         f"{fmt_rate(m['precision_wrong'])} | {fmt_rate(m['correct_preservation'])} | "
                         f"{m['wrong_to_correct']}/20 | {m['correct_to_wrong']}/20 | {100*m['final_accuracy']:.1f}% |")
    lines += ["", "## Oracle-known-wrong repair", "",
              "| Checkpoint | Oracle routes | Verified repairs | Repair rate |",
              "|---|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        m = report["checkpoints"][checkpoint]["oracle_known_wrong"]
        lines.append(f"| {DISPLAY[checkpoint]} | {m['routed_known_wrong']} | {m['verified_repairs']} | {100*m['verified_repair_rate']:.1f}% |")
    lines += ["", "## Boundary", "",
              "Self-confidence and frozen-probe routes use the same non-oracle recheck pattern. "
              "Oracle status is analyzed separately to isolate repair ability. This is a fresh holdout "
              "experiment but remains a harness-controlled system, not autonomous model-only correction."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["checkpoints"], indent=2))


if __name__ == "__main__":
    main()
