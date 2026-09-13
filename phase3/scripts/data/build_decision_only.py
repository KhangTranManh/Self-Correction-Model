"""Build the balanced Phase 3 decision-only diagnostic dataset.

Every label comes from an existing deterministic verification of the
Self_Correction_v1 attempt.  Targets contain only one KEEP/REVISE tag; answer
generation and explanatory prose are deliberately excluded from SFT targets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import (
    ProvenanceResolver,
    VerificationEngine,
    build_prompt,
    problem_for_prompt,
)


ROOT = Path(__file__).resolve().parents[2]
SEED = 314159
DEFAULT_INVENTORY = ROOT / "data" / "behavior" / "unified_source_inventory.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "decision_only"
DECISION_RE = re.compile(r"<decision>(KEEP|REVISE)</decision>")
TEMPLATES = (
    (
        "neutral_review_01",
        "Review your previous answer carefully and decide whether it should be kept or revised.",
    ),
    (
        "neutral_review_02",
        "Re-examine your previous answer and decide whether to keep it or revise it.",
    ),
    (
        "neutral_review_03",
        "Check your previous response carefully, then decide if it should be kept or revised.",
    ),
)

# Identical quotas are used for KEEP and REVISE.  MBPP has only 29 V1-correct
# sources, so 24/36 is the largest comfortably source-disjoint MBPP/APPS mix.
QUOTAS = {
    ("math", "gsm8k"): 60,
    ("code", "mbpp"): 24,
    ("code", "apps"): 36,
}
SPLIT_QUOTAS = {
    ("math", "gsm8k"): {"train": 42, "dev": 9, "test": 9},
    ("code", "mbpp"): {"train": 17, "dev": 4, "test": 3},
    ("code", "apps"): {"train": 25, "dev": 5, "test": 6},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(context: str, source_id: str) -> str:
    return hashlib.sha256(f"{SEED}|decision-only|{context}|{source_id}".encode()).hexdigest()


def eligible(row: dict[str, Any], decision: str) -> bool:
    if decision == "KEEP":
        return bool(row["v1_correct"]) and row["bucket"] in {"CC", "WC"}
    return not bool(row["v1_correct"]) and row["bucket"] == "WW"


def assign_templates(rows: list[dict[str, Any]]) -> None:
    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["domain"], row["decision"])].append(row)
    for key, members in groups.items():
        members.sort(key=lambda row: (stable_rank(f"template:{key}", row["source_id"]), row["source_id"]))
        for index, row in enumerate(members):
            template_id, template = TEMPLATES[index % len(TEMPLATES)]
            row["neutral_template_id"] = template_id
            row["neutral_review_text"] = template
            row["messages"] = [
                {"role": "user", "content": row.pop("_task_prompt")},
                {"role": "assistant", "content": row.pop("_v1_output")},
                {"role": "user", "content": template},
                {"role": "assistant", "content": f"<decision>{row['decision']}</decision>"},
            ]


def nested_counts(rows: list[dict[str, Any]], *fields: str) -> dict[str, Any]:
    if len(fields) == 1:
        return dict(sorted(Counter(str(row[fields[0]]) for row in rows).items()))
    output: dict[str, Any] = {}
    for value in sorted({str(row[fields[0]]) for row in rows}):
        output[value] = nested_counts(
            [row for row in rows if str(row[fields[0]]) == value], *fields[1:]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    inventory_path = Path(args.inventory).resolve()
    output_dir = Path(args.output_dir).resolve()
    inventory = read_jsonl(inventory_path)
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    fresh_verification_exclusions: list[dict[str, Any]] = []

    for decision in ("KEEP", "REVISE"):
        for (domain, dataset), requested in QUOTAS.items():
            candidates = [
                row
                for row in inventory
                if row["domain"] == domain
                and row["dataset"] == dataset
                and eligible(row, decision)
            ]
            candidates.sort(
                key=lambda row: (
                    stable_rank(f"select:{decision}:{domain}:{dataset}", row["id"]),
                    row["id"],
                )
            )
            candidates = [row for row in candidates if row["id"] not in used_sources]
            if len(candidates) < requested:
                raise RuntimeError(
                    f"Insufficient {decision}/{domain}/{dataset}: {len(candidates)} < {requested}"
                )
            chosen: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            expected_correct = decision == "KEEP"
            for source_meta in candidates:
                source_id = str(source_meta["id"])
                source = resolver.resolve(source_meta["source_ref"], source_id)
                attempt = resolver.resolve(source_meta["v1_attempt_ref"], source_id)
                result = verifier.verify(source, str(attempt["initial_output"]))
                if bool(result["passed"]) != expected_correct:
                    fresh_verification_exclusions.append(
                        {
                            "source_id": source_id,
                            "decision_cell": decision,
                            "domain": domain,
                            "dataset": dataset,
                            "historical_v1_correct": bool(source_meta["v1_correct"]),
                            "fresh_passed": bool(result["passed"]),
                            "verifier_detail": str(result["detail"]),
                        }
                    )
                    continue
                chosen.append((source_meta, source, attempt, result))
                if len(chosen) == requested:
                    break
            if len(chosen) < requested:
                raise RuntimeError(
                    f"Only {len(chosen)} freshly verified rows for "
                    f"{decision}/{domain}/{dataset}; need {requested}"
                )
            split_cursor = 0
            for split, count in SPLIT_QUOTAS[(domain, dataset)].items():
                split_rows = chosen[split_cursor : split_cursor + count]
                split_cursor += count
                for source_meta, source, attempt, result in split_rows:
                    source_id = str(source_meta["id"])
                    task_prompt = build_prompt(problem_for_prompt(source))
                    selected.append(
                        {
                            "construction_id": f"decision_only_v1::{source_id}",
                            "source_id": source_id,
                            "dataset": dataset,
                            "domain": domain,
                            "source_split": source_meta["source_split"],
                            "split": split,
                            "bucket": source_meta["bucket"],
                            "decision": decision,
                            "v1_correct": bool(source_meta["v1_correct"]),
                            "label_source": "existing_deterministic_verifier_reconfirmed",
                            "verification_method": verifier.method(source),
                            "verifier_detail": str(result["detail"]),
                            "teacher_used": False,
                            "source_ref": source_meta["source_ref"],
                            "v1_attempt_ref": source_meta["v1_attempt_ref"],
                            "selection_seed": SEED,
                            "training_format": "messages_final_assistant_decision_only",
                            "_task_prompt": task_prompt,
                            "_v1_output": str(attempt["initial_output"]),
                        }
                    )
                    used_sources.add(source_id)

    assign_templates(selected)
    selected.sort(key=lambda row: (row["split"], row["domain"], row["decision"], row["dataset"], row["source_id"]))

    errors: list[str] = []
    if len(selected) != 240 or len(used_sources) != 240:
        errors.append(f"Expected 240 unique rows, got rows={len(selected)}, sources={len(used_sources)}")
    if any(row["bucket"] == "CW" for row in selected):
        errors.append("CW contamination")
    if any("humaneval" in row["dataset"].casefold() for row in selected):
        errors.append("HumanEval contamination")
    if any(row["teacher_used"] for row in selected):
        errors.append("Teacher-generated labels detected")
    for row in selected:
        target = str(row["messages"][-1]["content"])
        match = DECISION_RE.fullmatch(target)
        if not match or match.group(1) != row["decision"] or "<answer>" in target:
            errors.append(f"Invalid target contract: {row['construction_id']}")
    split_sources = {
        split: {row["source_id"] for row in selected if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_sources[left] & split_sources[right]
        if overlap:
            errors.append(f"Source overlap {left}/{right}: {sorted(overlap)}")
    template_by_split_domain_decision = nested_counts(
        selected, "split", "domain", "decision", "neutral_template_id"
    )
    for split in ("train", "dev", "test"):
        for domain in ("math", "code"):
            if template_by_split_domain_decision[split][domain]["KEEP"] != template_by_split_domain_decision[split][domain]["REVISE"]:
                errors.append(f"Template-label imbalance: {split}/{domain}")
    if errors:
        raise RuntimeError("; ".join(errors))

    paths = {}
    write_jsonl(output_dir / "decision_only_dataset.jsonl", selected)
    paths["all"] = output_dir / "decision_only_dataset.jsonl"
    for split in ("train", "dev", "test"):
        path = output_dir / f"decision_only_{split}.jsonl"
        write_jsonl(path, [row for row in selected if row["split"] == split])
        paths[split] = path
    summary = {
        "schema_version": "phase3_decision_only_v1",
        "seed": SEED,
        "total_rows": len(selected),
        "unique_sources": len(used_sources),
        "counts_by_split": nested_counts(selected, "split"),
        "counts_by_split_domain_decision": nested_counts(selected, "split", "domain", "decision"),
        "counts_by_split_dataset_decision": nested_counts(selected, "split", "dataset", "decision"),
        "template_distribution": template_by_split_domain_decision,
        "bucket_distribution": nested_counts(selected, "decision", "bucket"),
        "fresh_verification_exclusions": fresh_verification_exclusions,
        "fresh_verification_exclusion_count": len(fresh_verification_exclusions),
        "source_overlap": {"train_dev": [], "train_test": [], "dev_test": []},
        "validation": {
            "all_keep_v1_correct_cc_or_wc": all(
                row["v1_correct"] and row["bucket"] in {"CC", "WC"}
                for row in selected if row["decision"] == "KEEP"
            ),
            "all_revise_v1_wrong_ww": all(
                not row["v1_correct"] and row["bucket"] == "WW"
                for row in selected if row["decision"] == "REVISE"
            ),
            "no_cw": True,
            "no_humaneval": True,
            "no_frozen_eval": all(
                (row["dataset"] == "gsm8k" and row["source_split"] == "train")
                or (row["dataset"] == "mbpp" and row["source_split"] != "test")
                or (row["dataset"] == "apps" and row["source_split"] == "train")
                for row in selected
            ),
            "one_row_per_source": True,
            "decision_only_targets": True,
            "no_teacher_labels": True,
            "template_distribution_label_balanced": True,
            "all_passed": True,
        },
        "input": {"path": str(inventory_path), "sha256": file_hash(inventory_path)},
        "outputs": {
            name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()
        },
    }
    write_json(output_dir / "decision_only_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
