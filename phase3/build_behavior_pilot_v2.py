"""Build the 200-row Phase 3 contract-focused pilot.

The original 30-row dev set is frozen byte-for-byte. The 170-row training set
contains 128 review-behavior rows, 26 fresh-task anchors, and 16 format-only
warmups whose KEEP/REVISE labels are explicitly supplied and perfectly
balanced. All source selection is deterministic and model-independent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from construct_behavior_pilot import (
    ANSWER_CONTRACT_RE,
    CONDITIONS,
    behavior_contract,
    construct_rows,
    extract_target_answer,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parent
SEED = 314159
DEFAULT_MANIFEST = ROOT / "data" / "behavior" / "selection" / "behavior_selection_manifest.jsonl"
DEFAULT_FROZEN_DEV = ROOT / "data" / "behavior" / "mini_train" / "dev.jsonl"
DEFAULT_CONSTRUCTION_DIR = ROOT / "data" / "behavior" / "construction_pilot_v2"
DEFAULT_SPLIT_DIR = ROOT / "data" / "behavior" / "mini_train_v2"
REVIEW_CONDITIONS = (
    "preserve_neutral",
    "preserve_false_feedback",
    "repair_neutral",
    "repair_true_feedback",
)
FRESH_CONDITIONS = ("normal_solve", "regression_recovery")
LEGACY_REVIEW_PROSE_RE = re.compile(
    r"(?:the previous (?:response|answer) is correct|after reviewing the (?:previous|earlier)|"
    r"###\s*(?:original answer|revised answer|error|correction|revision))",
    flags=re.IGNORECASE,
)
# Reverification under the current executable sandbox disagrees with the
# historical bucket for these sources. They are deterministically excluded.
UNSTABLE_SOURCE_IDS = {"apps_train_3492"}


def stable_rank(context: str, identifier: str) -> str:
    return hashlib.sha256(f"{SEED}|v2|{context}|{identifier}".encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(jsonl_bytes(rows))
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


class Selector:
    def __init__(
        self, manifest: list[dict[str, Any]], forbidden_sources: set[str]
    ) -> None:
        self.manifest = manifest
        self.used_sources = set(forbidden_sources)
        self.selected_ids: set[str] = set()
        self.rows: list[dict[str, Any]] = []

    def _add(self, row: dict[str, Any]) -> None:
        source_id = str(row["source_id"])
        selection_id = str(row["selection_id"])
        if source_id in self.used_sources or selection_id in self.selected_ids:
            raise RuntimeError(f"Duplicate/forbidden source selection: {source_id}")
        self.used_sources.add(source_id)
        self.selected_ids.add(selection_id)
        self.rows.append(row)

    def add_pairs(
        self,
        *,
        context: str,
        first_condition: str,
        second_condition: str,
        domain: str,
        dataset: str | None,
        count: int,
    ) -> None:
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.manifest:
            if (
                row["condition"] in {first_condition, second_condition}
                and row["domain"] == domain
                and (dataset is None or row["dataset"] == dataset)
                and row.get("pair_id")
                and row["source_id"] not in self.used_sources
            ):
                groups[str(row["pair_id"])].append(row)
        candidates = []
        for pair_id, members in groups.items():
            by_condition = {row["condition"]: row for row in members}
            if {first_condition, second_condition} <= by_condition.keys():
                candidates.append(
                    (pair_id, by_condition[first_condition], by_condition[second_condition])
                )
        candidates.sort(key=lambda item: (stable_rank(context, item[0]), item[0]))
        if len(candidates) < count:
            raise RuntimeError(f"Insufficient pairs for {context}: {len(candidates)} < {count}")
        for _, first, second in candidates[:count]:
            # A counterfactual pair intentionally uses one source twice.
            source_id = str(first["source_id"])
            if source_id != str(second["source_id"]):
                raise RuntimeError("Pair members do not share a source")
            if source_id in self.used_sources:
                raise RuntimeError(f"Pair source already used: {source_id}")
            self.used_sources.add(source_id)
            for row in (first, second):
                selection_id = str(row["selection_id"])
                if selection_id in self.selected_ids:
                    raise RuntimeError(f"Duplicate selection: {selection_id}")
                self.selected_ids.add(selection_id)
                self.rows.append(row)

    def add_singles(
        self,
        *,
        context: str,
        condition: str,
        domain: str,
        dataset: str | None,
        count: int,
    ) -> list[dict[str, Any]]:
        candidates = [
            row
            for row in self.manifest
            if row["condition"] == condition
            and row["domain"] == domain
            and (dataset is None or row["dataset"] == dataset)
            and not row.get("is_counterfactual_pair")
            and row["source_id"] not in self.used_sources
        ]
        candidates.sort(
            key=lambda row: (
                stable_rank(context, str(row["selection_id"])),
                str(row["selection_id"]),
            )
        )
        if len(candidates) < count:
            raise RuntimeError(f"Insufficient singles for {context}: {len(candidates)} < {count}")
        selected = candidates[:count]
        for row in selected:
            self._add(row)
        return selected


def select_training_rows(
    manifest: list[dict[str, Any]], dev_sources: set[str]
) -> tuple[list[dict[str, Any]], set[str]]:
    selector = Selector(manifest, dev_sources)

    # Eight complete counterfactual pairs per family. Each review condition then
    # receives another 24 source-unique singles for 32 behavioral rows.
    for family, first, second in (
        ("preserve", "preserve_neutral", "preserve_false_feedback"),
        ("repair", "repair_neutral", "repair_true_feedback"),
    ):
        for domain, dataset, count in (
            ("math", None, 4),
            ("code", "mbpp", 2),
            ("code", "apps", 2),
        ):
            selector.add_pairs(
                context=f"{family}_pair_{domain}_{dataset or 'all'}",
                first_condition=first,
                second_condition=second,
                domain=domain,
                dataset=dataset,
                count=count,
            )
        single_cells = (
            (("math", None, 12), ("code", "mbpp", 4), ("code", "apps", 8))
            if family == "preserve"
            else (("math", None, 12), ("code", "mbpp", 6), ("code", "apps", 6))
        )
        for condition in (first, second):
            for domain, dataset, count in single_cells:
                selector.add_singles(
                    context=f"{condition}_{domain}_{dataset or 'all'}",
                    condition=condition,
                    domain=domain,
                    dataset=dataset,
                    count=count,
                )

    # Fresh-task anchors remain represented so correction scaffolding cannot
    # become the universal response format.
    for condition in FRESH_CONDITIONS:
        for domain, dataset, count in (
            ("math", None, 6),
            ("code", "mbpp", 3),
            ("code", "apps", 4),
        ):
            selector.add_singles(
                context=f"{condition}_{domain}_{dataset or 'all'}",
                condition=condition,
                domain=domain,
                dataset=dataset,
                count=count,
            )

    # Format-only rows carry both decisions equally. The requested decision is
    # supplied in the input, so correctness cannot leak the behavior label.
    warmup_ids: set[str] = set()
    for decision, condition in (("KEEP", "preserve_neutral"), ("REVISE", "repair_neutral")):
        for domain, dataset, count in (
            ("math", None, 4),
            ("code", "apps", 4),
        ):
            selected = selector.add_singles(
                context=f"format_{decision}_{domain}_{dataset or 'all'}",
                condition=condition,
                domain=domain,
                dataset=dataset,
                count=count,
            )
            warmup_ids.update(str(row["selection_id"]) for row in selected)

    if len(selector.rows) != 170 or len(warmup_ids) != 16:
        raise RuntimeError(
            f"Selection size mismatch: rows={len(selector.rows)}, warmups={len(warmup_ids)}"
        )
    return selector.rows, warmup_ids


def make_format_warmup(row: dict[str, Any]) -> dict[str, Any]:
    decision = str(row["decision"])
    answer = extract_target_answer(row)
    prompt = (
        "Format-only task. Do not judge whether the answer is correct. The decision "
        "is supplied explicitly and is unrelated to correctness. Return only the two "
        "tags shown by the schema, with no commentary.\n\n"
        "Schema:\n<decision>VALUE</decision>\n<answer>ANSWER</answer>\n\n"
        f"VALUE: {decision}\nANSWER:\n{answer}"
    )
    converted = dict(row)
    converted.update(
        {
            "construction_id": f"pilot_v2::format::{row['selection_id']}",
            "condition": "format_warmup",
            "feedback_type": "format_only",
            "feedback_template_id": "format_only_balanced_v1",
            "feedback_text": None,
            "is_counterfactual_pair": False,
            "pair_id": None,
            "training_format": "messages_final_assistant_target_format_only",
            "format_only": True,
            "format_label_source": "explicit_user_supplied",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": behavior_contract(decision, answer)},
            ],
        }
    )
    return converted


def validate_and_summarize(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    frozen_dev_path: Path,
    output_dev_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    all_rows = train + dev
    train_sources = {str(row["source_id"]) for row in train}
    dev_sources = {str(row["source_id"]) for row in dev}
    if train_sources & dev_sources:
        errors.append(f"train/dev source overlap: {sorted(train_sources & dev_sources)}")
    if len(train) != 170 or len(dev) != 30 or len(all_rows) != 200:
        errors.append(f"size mismatch: train={len(train)}, dev={len(dev)}")
    train_counts = Counter(str(row["condition"]) for row in train)
    expected_train = {
        "preserve_neutral": 32,
        "preserve_false_feedback": 32,
        "repair_neutral": 32,
        "repair_true_feedback": 32,
        "normal_solve": 13,
        "regression_recovery": 13,
        "format_warmup": 16,
    }
    if dict(train_counts) != expected_train:
        errors.append(f"train condition mismatch: {dict(train_counts)}")

    exact_rows = []
    legacy_answer_rows = []
    for row in all_rows:
        if row["decision"] in {"KEEP", "REVISE"}:
            target = str(row["messages"][-1]["content"])
            match = ANSWER_CONTRACT_RE.fullmatch(target)
            if not match or match.group(1) != row["decision"]:
                errors.append(f"invalid exact contract: {row['construction_id']}")
                continue
            exact_rows.append(row)
            if LEGACY_REVIEW_PROSE_RE.search(match.group(2)):
                legacy_answer_rows.append(str(row["construction_id"]))
    if legacy_answer_rows:
        errors.append(f"legacy correction prose inside answer targets: {legacy_answer_rows}")

    warmups = [row for row in train if row["condition"] == "format_warmup"]
    warmup_decisions = Counter(str(row["decision"]) for row in warmups)
    if warmup_decisions != Counter({"KEEP": 8, "REVISE": 8}):
        errors.append(f"format warmup label imbalance: {dict(warmup_decisions)}")
    if any(row.get("format_label_source") != "explicit_user_supplied" for row in warmups):
        errors.append("format-only labels are not explicitly supplied")
    if file_hash(frozen_dev_path) != file_hash(output_dev_path):
        errors.append("frozen dev bytes changed")

    pair_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        if row.get("pair_id"):
            pair_splits[str(row["pair_id"])].add(str(row["pilot_split"]))
    crossing_pairs = sorted(key for key, values in pair_splits.items() if len(values) > 1)
    if crossing_pairs:
        errors.append(f"counterfactual pairs cross split: {crossing_pairs}")

    summary = {
        "schema_version": "phase3_contract_pilot_v2",
        "seed": SEED,
        "total_rows": len(all_rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "dev_frozen_byte_identical": file_hash(frozen_dev_path) == file_hash(output_dev_path),
        "dev_sha256": file_hash(output_dev_path),
        "train_sha256": None,
        "counts_by_split": dict(sorted(Counter(row["pilot_split"] for row in all_rows).items())),
        "counts_by_condition": dict(sorted(Counter(row["condition"] for row in all_rows).items())),
        "train_counts_by_condition": dict(sorted(train_counts.items())),
        "counts_by_decision": dict(sorted(Counter(row["decision"] for row in all_rows).items())),
        "counts_by_domain": dict(sorted(Counter(row["domain"] for row in all_rows).items())),
        "counts_by_dataset": dict(sorted(Counter(row["dataset"] for row in all_rows).items())),
        "exact_contract_rows": len(exact_rows),
        "exact_contract_rate": round(len(exact_rows) / len(all_rows), 6),
        "train_exact_contract_rows": sum(row in exact_rows for row in train),
        "format_warmup_rows": len(warmups),
        "format_warmup_decisions": dict(sorted(warmup_decisions.items())),
        "format_label_is_behavior_independent": True,
        "legacy_review_prose_target_count": len(legacy_answer_rows),
        "train_dev_source_overlap": sorted(train_sources & dev_sources),
        "counterfactual_pairs_crossing_split": crossing_pairs,
        "validation_errors": errors,
        "all_validation_passed": not errors,
    }
    if errors:
        raise RuntimeError("; ".join(errors))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--frozen-dev", default=str(DEFAULT_FROZEN_DEV))
    parser.add_argument("--construction-dir", default=str(DEFAULT_CONSTRUCTION_DIR))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    frozen_dev_path = Path(args.frozen_dev).resolve()
    construction_dir = Path(args.construction_dir).resolve()
    split_dir = Path(args.split_dir).resolve()
    manifest = read_jsonl(manifest_path)
    dev = read_jsonl(frozen_dev_path)
    dev_sources = {str(row["source_id"]) for row in dev}

    selections, warmup_ids = select_training_rows(
        manifest, dev_sources | UNSTABLE_SOURCE_IDS
    )
    constructed, failures = construct_rows(selections)
    if failures or len(constructed) != 170:
        raise RuntimeError(
            f"Construction failed: constructed={len(constructed)}, failures="
            + json.dumps(failures, ensure_ascii=False)
        )

    train = []
    for row in constructed:
        converted = make_format_warmup(row) if row["selection_id"] in warmup_ids else dict(row)
        converted["pilot_split"] = "train"
        converted["pilot_version"] = "contract_v2"
        train.append(converted)
    train.sort(key=lambda row: (row["condition"], row["construction_id"]))

    # Preserve every byte of the frozen dev file in the new split directory.
    train_path = split_dir / "train.jsonl"
    dev_path = split_dir / "dev.jsonl"
    write_jsonl(train_path, train)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path.write_bytes(frozen_dev_path.read_bytes())

    summary = validate_and_summarize(train, dev, frozen_dev_path, dev_path)
    summary["train_sha256"] = file_hash(train_path)
    summary["inputs"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_hash(manifest_path),
        "frozen_dev": str(frozen_dev_path),
        "frozen_dev_sha256": file_hash(frozen_dev_path),
    }
    summary["outputs"] = {
        "train": str(train_path),
        "dev": str(dev_path),
        "combined": str(construction_dir / "pilot_behavior_rows.jsonl"),
    }
    combined = sorted(train + dev, key=lambda row: (row["pilot_split"], row["construction_id"]))
    write_jsonl(construction_dir / "pilot_behavior_rows.jsonl", combined)
    write_jsonl(construction_dir / "pilot_failures.jsonl", [])
    write_json(construction_dir / "pilot_summary.json", summary)
    write_json(split_dir / "split_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
