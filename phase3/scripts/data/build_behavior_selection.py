"""Build the deterministic Phase 3 v1 source/condition selection manifest.

This step selects metadata references only. It deliberately does not construct
prompts, feedback, corrected answers, teacher outputs, or chat/SFT examples.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "behavior" / "unified_source_inventory.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "behavior" / "selection"
SEED = 314159
MAX_ROWS_PER_SOURCE = 2

CONDITION_TO_DECISION = {
    "preserve_neutral": "KEEP",
    "preserve_false_feedback": "KEEP",
    "repair_neutral": "REVISE",
    "repair_true_feedback": "REVISE",
    "normal_solve": "NORMAL",
    "regression_recovery": "RECOVER",
}
CONDITION_TO_BEHAVIOR = {
    "preserve_neutral": "preserve",
    "preserve_false_feedback": "preserve",
    "repair_neutral": "repair",
    "repair_true_feedback": "repair",
    "normal_solve": "normal_solve",
    "regression_recovery": "regression_recovery",
}
ELIGIBLE_BUCKETS = {
    "preserve_neutral": {"CC", "WC"},
    "preserve_false_feedback": {"CC", "WC"},
    "repair_neutral": {"WW"},
    "repair_true_feedback": {"WW"},
    "normal_solve": {"WC", "CW"},
    "regression_recovery": {"CW"},
}
EXPECTED_BUCKET_STATE = {
    "CC": (True, True),
    "WW": (False, False),
    "WC": (False, True),
    "CW": (True, False),
}
PAIR_CONDITION_SETS = {
    frozenset(("preserve_neutral", "preserve_false_feedback")),
    frozenset(("repair_neutral", "repair_true_feedback")),
}
TARGET_BY_DOMAIN_CONDITION = {
    ("math", "preserve_neutral"): 100,
    ("math", "preserve_false_feedback"): 100,
    ("code", "preserve_neutral"): 85,
    ("code", "preserve_false_feedback"): 85,
    ("math", "repair_neutral"): 60,
    ("math", "repair_true_feedback"): 60,
    ("code", "repair_neutral"): 85,
    ("code", "repair_true_feedback"): 85,
    ("math", "normal_solve"): 60,
    ("code", "normal_solve"): 60,
    ("math", "regression_recovery"): 42,
    ("code", "regression_recovery"): 38,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return text.encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_source_key(row: dict[str, Any]) -> tuple[str, int, str]:
    source_index = row.get("source_index")
    normalized_index = int(source_index) if source_index is not None else 2**63 - 1
    return str(row["dataset"]), normalized_index, str(row["id"])


def stable_rank(context: str, source_id: str) -> str:
    material = f"{SEED}|{context}|{source_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def nested_counts(
    rows: Sequence[dict[str, Any]], outer: str, inner: str
) -> dict[str, dict[str, int]]:
    outer_values = sorted({str(row[outer]) for row in rows})
    inner_values = sorted({str(row[inner]) for row in rows})
    return {
        outer_value: {
            inner_value: sum(
                str(row[outer]) == outer_value and str(row[inner]) == inner_value
                for row in rows
            )
            for inner_value in inner_values
        }
        for outer_value in outer_values
    }


class Selector:
    def __init__(self, source_rows: list[dict[str, Any]]) -> None:
        self.sources = sorted(source_rows, key=stable_source_key)
        self.source_by_id = {str(row["id"]): row for row in self.sources}
        self.source_position = {
            str(row["id"]): position for position, row in enumerate(self.sources)
        }
        self.selected: list[dict[str, Any]] = []
        self.usage: Counter[str] = Counter()
        self.conditions_by_source: defaultdict[str, list[str]] = defaultdict(list)

    def candidates(
        self,
        *,
        context: str,
        domain: str,
        bucket: str,
        dataset: str | None = None,
        require_unused: bool = True,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.sources
            if row["domain"] == domain
            and row["bucket"] == bucket
            and (dataset is None or row["dataset"] == dataset)
            and (not require_unused or self.usage[str(row["id"])] == 0)
        ]
        return sorted(
            rows,
            key=lambda row: (
                stable_rank(context, str(row["id"])),
                self.source_position[str(row["id"])],
            ),
        )

    def choose(
        self,
        *,
        context: str,
        count: int,
        domain: str,
        bucket: str,
        dataset: str | None = None,
        require_unused: bool = True,
    ) -> list[dict[str, Any]]:
        candidates = self.candidates(
            context=context,
            domain=domain,
            bucket=bucket,
            dataset=dataset,
            require_unused=require_unused,
        )
        if len(candidates) < count:
            raise RuntimeError(
                f"Insufficient candidates for {context}: requested={count}, "
                f"available={len(candidates)}, domain={domain}, bucket={bucket}, "
                f"dataset={dataset}"
            )
        return candidates[:count]

    def add(
        self,
        source: dict[str, Any],
        condition: str,
        *,
        pair_id: str | None = None,
        usage_justification: str | None = None,
        target_requirement: str | None = None,
    ) -> None:
        source_id = str(source["id"])
        if self.usage[source_id] >= MAX_ROWS_PER_SOURCE:
            raise RuntimeError(f"Source cap exceeded while adding {source_id}")
        if condition in self.conditions_by_source[source_id]:
            raise RuntimeError(f"Duplicate condition {condition} for {source_id}")
        if source["bucket"] not in ELIGIBLE_BUCKETS[condition]:
            raise RuntimeError(
                f"Ineligible source {source_id}: bucket={source['bucket']} condition={condition}"
            )

        selection_id = f"phase3v1::{source_id}::{condition}"
        row = {
            "selection_id": selection_id,
            "source_id": source_id,
            "dataset": source["dataset"],
            "domain": source["domain"],
            "bucket": source["bucket"],
            "behavior": CONDITION_TO_BEHAVIOR[condition],
            "condition": condition,
            "decision": CONDITION_TO_DECISION[condition],
            "initial_correct": bool(source["v1_correct"]),
            "base_correct": bool(source["base_correct"]),
            "v1_correct": bool(source["v1_correct"]),
            "source_origin": source["source_origin"],
            "source_split": source["source_split"],
            "source_index": source.get("source_index"),
            "source_task_id": source.get("source_task_id"),
            "source_ref": source["source_ref"],
            "base_attempt_ref": source["base_attempt_ref"],
            "v1_attempt_ref": source["v1_attempt_ref"],
            "transition_type": source["transition_type"],
            "is_counterfactual_pair": pair_id is not None,
            "pair_id": pair_id,
            "source_usage_justification": usage_justification,
            "target_requirement": target_requirement,
            "selection_seed": SEED,
        }
        if source["dataset"] == "apps":
            row.update(
                difficulty=source.get("difficulty"),
                source_host=source.get("source_host"),
                test_mode=source.get("test_mode"),
                test_count=source.get("test_count"),
            )
        self.selected.append(row)
        self.usage[source_id] += 1
        self.conditions_by_source[source_id].append(condition)

    def add_paired_stratum(
        self,
        *,
        context: str,
        domain: str,
        bucket: str,
        pair_count: int,
        first_condition: str,
        second_condition: str,
        first_single_count: int,
        second_single_count: int,
        dataset: str | None = None,
    ) -> None:
        pair_sources = self.choose(
            context=f"{context}:pairs",
            count=pair_count,
            domain=domain,
            bucket=bucket,
            dataset=dataset,
        )
        for source in pair_sources:
            pair_id = f"pair::{context}::{source['id']}"
            self.add(source, first_condition, pair_id=pair_id)
            self.add(source, second_condition, pair_id=pair_id)

        first_sources = self.choose(
            context=f"{context}:{first_condition}:single",
            count=first_single_count,
            domain=domain,
            bucket=bucket,
            dataset=dataset,
        )
        for source in first_sources:
            self.add(source, first_condition)

        second_sources = self.choose(
            context=f"{context}:{second_condition}:single",
            count=second_single_count,
            domain=domain,
            bucket=bucket,
            dataset=dataset,
        )
        for source in second_sources:
            self.add(source, second_condition)


def build_selection(source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Selector]:
    selector = Selector(source_rows)

    # Math KEEP: exactly 70 CC + 30 WC per condition, with 30 shared pairs.
    selector.add_paired_stratum(
        context="math_keep_cc",
        domain="math",
        bucket="CC",
        pair_count=20,
        first_condition="preserve_neutral",
        second_condition="preserve_false_feedback",
        first_single_count=50,
        second_single_count=50,
    )
    selector.add_paired_stratum(
        context="math_keep_wc",
        domain="math",
        bucket="WC",
        pair_count=10,
        first_condition="preserve_neutral",
        second_condition="preserve_false_feedback",
        first_single_count=20,
        second_single_count=20,
    )

    # Math REVISE: 40 shared WW sources plus 20 condition-specific sources each.
    selector.add_paired_stratum(
        context="math_revise_ww",
        domain="math",
        bucket="WW",
        pair_count=40,
        first_condition="repair_neutral",
        second_condition="repair_true_feedback",
        first_single_count=20,
        second_single_count=20,
    )

    # Code KEEP uses only CC here so all scarce WC sources remain available for
    # NORMAL anchors. Twenty-four MBPP CC sources yield 36 rows (12 pairs),
    # while APPS yields 134 rows (18 pairs): 140 unique code sources in total.
    selector.add_paired_stratum(
        context="code_keep_mbpp_cc",
        domain="code",
        bucket="CC",
        dataset="mbpp",
        pair_count=12,
        first_condition="preserve_neutral",
        second_condition="preserve_false_feedback",
        first_single_count=6,
        second_single_count=6,
    )
    selector.add_paired_stratum(
        context="code_keep_apps_cc",
        domain="code",
        bucket="CC",
        dataset="apps",
        pair_count=18,
        first_condition="preserve_neutral",
        second_condition="preserve_false_feedback",
        first_single_count=49,
        second_single_count=49,
    )

    # Code REVISE remains mixed in both conditions. The 35/50 MBPP/APPS split
    # reduces dataset->decision correlation without exhausting scarce MBPP KEEP.
    selector.add_paired_stratum(
        context="code_revise_mbpp_ww",
        domain="code",
        bucket="WW",
        dataset="mbpp",
        pair_count=8,
        first_condition="repair_neutral",
        second_condition="repair_true_feedback",
        first_single_count=27,
        second_single_count=27,
    )
    selector.add_paired_stratum(
        context="code_revise_apps_ww",
        domain="code",
        bucket="WW",
        dataset="apps",
        pair_count=12,
        first_condition="repair_neutral",
        second_condition="repair_true_feedback",
        first_single_count=38,
        second_single_count=38,
    )

    # NORMAL math sources are WC and disjoint from the WC sources used for KEEP.
    math_normal = selector.choose(
        context="math_normal_wc",
        count=60,
        domain="math",
        bucket="WC",
    )
    for source in math_normal:
        selector.add(
            source,
            "normal_solve",
            target_requirement="verified_correct_target_only",
        )

    # All 26 code WC rows are preferred NORMAL anchors. The remaining 34 rows
    # use CW (all 11 MBPP + 23 APPS) and require a reference/Base-correct target.
    for dataset, count in (("mbpp", 5), ("apps", 21)):
        code_wc = selector.choose(
            context=f"code_normal_{dataset}_wc",
            count=count,
            domain="code",
            bucket="WC",
            dataset=dataset,
        )
        for source in code_wc:
            selector.add(
                source,
                "normal_solve",
                target_requirement="verified_correct_target_only",
            )

    cw_dual_justification = (
        "Code NORMAL has only 26 WC sources; this CW source supplies a fresh-solve "
        "anchor and regression-recovery row within the two-row cap. Later targets "
        "must use a verified reference or Base-correct solution, never V1-wrong output."
    )
    for dataset, count in (("mbpp", 11), ("apps", 23)):
        code_cw = selector.choose(
            context=f"code_normal_{dataset}_cw",
            count=count,
            domain="code",
            bucket="CW",
            dataset=dataset,
        )
        for source in code_cw:
            selector.add(
                source,
                "normal_solve",
                usage_justification=cw_dual_justification,
                target_requirement="reference_or_base_correct_target_required_never_v1_wrong",
            )

    # Recovery retains every CW source. Code CW sources already used for NORMAL
    # remain inside the two-row cap and carry the same explicit justification.
    recovery_sources = sorted(
        (row for row in selector.sources if row["bucket"] == "CW"),
        key=lambda row: (
            row["domain"],
            row["dataset"],
            stable_rank("all_cw_recovery", str(row["id"])),
        ),
    )
    for source in recovery_sources:
        is_dual_code_cw = selector.usage[str(source["id"])] == 1
        selector.add(
            source,
            "regression_recovery",
            usage_justification=cw_dual_justification if is_dual_code_cw else None,
            target_requirement="reference_or_base_correct_target_required_never_v1_wrong",
        )

    rows = sorted(selector.selected, key=lambda row: row["selection_id"])
    return rows, selector


def validate_source_inventory(source_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(row.get("id")) for row in source_rows]
    duplicates = sorted(source_id for source_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate unified source IDs: {duplicates[:10]}")
    for row in source_rows:
        source_id = str(row.get("id"))
        required = (
            "id",
            "dataset",
            "domain",
            "bucket",
            "base_correct",
            "v1_correct",
            "source_origin",
            "source_ref",
            "base_attempt_ref",
            "v1_attempt_ref",
        )
        missing = [field for field in required if field not in row or row[field] is None]
        if missing:
            errors.append(f"{source_id}: missing fields {missing}")
            continue
        state = (bool(row["base_correct"]), bool(row["v1_correct"]))
        if state != EXPECTED_BUCKET_STATE.get(str(row["bucket"])):
            errors.append(f"{source_id}: bucket/correctness mismatch")
    return errors


def validate_selection(
    selected: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> tuple[list[str], dict[str, bool]]:
    errors: list[str] = []
    source_by_id = {str(row["id"]): row for row in source_rows}
    selection_ids = [row["selection_id"] for row in selected]
    duplicate_selection_ids = sorted(
        selection_id for selection_id, count in Counter(selection_ids).items() if count > 1
    )
    if duplicate_selection_ids:
        errors.append(f"duplicate selection IDs: {duplicate_selection_ids[:10]}")

    usage = Counter(row["source_id"] for row in selected)
    over_cap = sorted(source_id for source_id, count in usage.items() if count > MAX_ROWS_PER_SOURCE)
    if over_cap:
        errors.append(f"sources over cap: {over_cap[:10]}")

    paired_by_id: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    conditions_by_source: defaultdict[str, list[str]] = defaultdict(list)
    for row in selected:
        source_id = row["source_id"]
        source = source_by_id.get(source_id)
        if source is None:
            errors.append(f"unknown selected source: {source_id}")
            continue
        for field in ("dataset", "domain", "bucket", "base_correct", "v1_correct", "source_origin"):
            if row[field] != source[field]:
                errors.append(f"{row['selection_id']}: provenance mismatch in {field}")
        if row["initial_correct"] != source["v1_correct"]:
            errors.append(f"{row['selection_id']}: initial_correct is not V1 correctness")
        condition = row["condition"]
        conditions_by_source[source_id].append(condition)
        if row["bucket"] not in ELIGIBLE_BUCKETS.get(condition, set()):
            errors.append(f"{row['selection_id']}: condition/bucket eligibility violation")
        if row["decision"] == "KEEP" and not row["v1_correct"]:
            errors.append(f"{row['selection_id']}: KEEP from V1-wrong source")
        if row["decision"] == "REVISE" and row["bucket"] != "WW":
            errors.append(f"{row['selection_id']}: ordinary REVISE is not WW")
        if row["bucket"] == "CW" and row["decision"] == "REVISE":
            errors.append(f"{row['selection_id']}: CW used as ordinary repair")
        if condition == "regression_recovery" and row["bucket"] != "CW":
            errors.append(f"{row['selection_id']}: recovery source is not CW")
        if "humaneval" in str(row["dataset"]).casefold():
            errors.append(f"{row['selection_id']}: HumanEval row selected")
        allowed_train_split = (
            (row["dataset"] == "gsm8k" and row["source_split"] == "train")
            or (row["dataset"] == "mbpp" and row["source_split"] in {"train", "validation", "prompt"})
            or (row["dataset"] == "apps" and row["source_split"] == "train")
        )
        if not allowed_train_split:
            errors.append(f"{row['selection_id']}: possible frozen-eval source")
        if row["is_counterfactual_pair"]:
            if not row["pair_id"]:
                errors.append(f"{row['selection_id']}: paired row has no pair_id")
            else:
                paired_by_id[row["pair_id"]].append(row)
        elif row["pair_id"] is not None:
            errors.append(f"{row['selection_id']}: unpaired row has pair_id")

    for pair_id, members in paired_by_id.items():
        member_sources = {row["source_id"] for row in members}
        member_conditions = frozenset(row["condition"] for row in members)
        if len(members) != 2:
            errors.append(f"{pair_id}: expected two members, got {len(members)}")
        if len(member_sources) != 1:
            errors.append(f"{pair_id}: members do not share a source")
        if member_conditions not in PAIR_CONDITION_SETS:
            errors.append(f"{pair_id}: incompatible pair conditions {sorted(member_conditions)}")

    for source_id, conditions in conditions_by_source.items():
        condition_set = frozenset(conditions)
        if len(conditions) <= 1:
            continue
        if condition_set in PAIR_CONDITION_SETS:
            continue
        if condition_set == frozenset(("normal_solve", "regression_recovery")):
            rows = [row for row in selected if row["source_id"] == source_id]
            if all(row["bucket"] == "CW" and row["source_usage_justification"] for row in rows):
                continue
        errors.append(f"{source_id}: incompatible multi-condition usage {sorted(conditions)}")

    checks = {
        "all_selected_sources_exist": all(row["source_id"] in source_by_id for row in selected),
        "no_frozen_evaluation_rows": not any("possible frozen-eval" in error for error in errors),
        "no_humaneval_rows": not any("HumanEval" in error for error in errors),
        "no_duplicate_selection_ids": not duplicate_selection_ids,
        "source_usage_cap_respected": not over_cap,
        "keep_only_from_v1_correct": not any("KEEP from V1-wrong" in error for error in errors),
        "ordinary_revise_only_from_ww": not any("ordinary REVISE is not WW" in error for error in errors),
        "cw_not_used_as_ordinary_repair": not any("CW used as ordinary repair" in error for error in errors),
        "recovery_only_from_cw": not any("recovery source is not CW" in error for error in errors),
        "provenance_metadata_preserved": not any("provenance mismatch" in error for error in errors),
        "pair_ids_internally_consistent": not any(error.startswith("pair::") for error in errors),
        "paired_rows_share_source": not any("members do not share" in error for error in errors),
        "no_incompatible_condition_leakage": not any("incompatible multi-condition" in error for error in errors),
    }
    checks["all_passed"] = all(checks.values()) and not errors
    return errors, checks


def probability(
    rows: Sequence[dict[str, Any]], decision: str, key: str, value: str
) -> dict[str, Any]:
    denominator = sum(str(row[key]) == value for row in rows)
    numerator = sum(str(row[key]) == value and row["decision"] == decision for row in rows)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "probability": round(numerator / denominator, 6) if denominator else None,
    }


def build_shortcut_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    requested = {
        "P(KEEP|math)": probability(rows, "KEEP", "domain", "math"),
        "P(KEEP|code)": probability(rows, "KEEP", "domain", "code"),
        "P(REVISE|math)": probability(rows, "REVISE", "domain", "math"),
        "P(REVISE|code)": probability(rows, "REVISE", "domain", "code"),
        "P(KEEP|mbpp)": probability(rows, "KEEP", "dataset", "mbpp"),
        "P(KEEP|apps)": probability(rows, "KEEP", "dataset", "apps"),
        "P(REVISE|mbpp)": probability(rows, "REVISE", "dataset", "mbpp"),
        "P(REVISE|apps)": probability(rows, "REVISE", "dataset", "apps"),
    }
    core = [row for row in rows if row["decision"] in {"KEEP", "REVISE"}]
    core_requested = {
        "P(KEEP|math,core)": probability(core, "KEEP", "domain", "math"),
        "P(KEEP|code,core)": probability(core, "KEEP", "domain", "code"),
        "P(REVISE|math,core)": probability(core, "REVISE", "domain", "math"),
        "P(REVISE|code,core)": probability(core, "REVISE", "domain", "code"),
        "P(KEEP|mbpp,core)": probability(core, "KEEP", "dataset", "mbpp"),
        "P(KEEP|apps,core)": probability(core, "KEEP", "dataset", "apps"),
        "P(REVISE|mbpp,core)": probability(core, "REVISE", "dataset", "mbpp"),
        "P(REVISE|apps,core)": probability(core, "REVISE", "dataset", "apps"),
    }
    gaps = {
        "all_rows_keep_math_vs_code": round(
            abs(requested["P(KEEP|math)"]["probability"] - requested["P(KEEP|code)"]["probability"]), 6
        ),
        "all_rows_revise_math_vs_code": round(
            abs(requested["P(REVISE|math)"]["probability"] - requested["P(REVISE|code)"]["probability"]), 6
        ),
        "core_keep_mbpp_vs_apps": round(
            abs(core_requested["P(KEEP|mbpp,core)"]["probability"] - core_requested["P(KEEP|apps,core)"]["probability"]), 6
        ),
        "core_revise_mbpp_vs_apps": round(
            abs(core_requested["P(REVISE|mbpp,core)"]["probability"] - core_requested["P(REVISE|apps,core)"]["probability"]), 6
        ),
    }
    severe_threshold = 0.30
    flags = [
        {"metric": metric, "gap": gap, "threshold": severe_threshold}
        for metric, gap in gaps.items()
        if gap >= severe_threshold
    ]
    return {
        "denominator_note": "Requested probabilities use every selected row in the conditioning group.",
        "requested_probabilities": requested,
        "core_denominator_note": "Core probabilities exclude NORMAL and RECOVER rows.",
        "core_probabilities": core_requested,
        "absolute_probability_gaps": gaps,
        "severe_imbalance_threshold": severe_threshold,
        "severe_imbalance_flags": flags,
    }


def target_achievement(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    achieved: dict[str, Any] = {}
    shortages: dict[str, int] = {}
    for (domain, condition), target in sorted(TARGET_BY_DOMAIN_CONDITION.items()):
        actual = sum(row["domain"] == domain and row["condition"] == condition for row in rows)
        key = f"{domain}::{condition}"
        achieved[key] = {"target": target, "actual": actual, "shortage": max(0, target - actual)}
        if actual < target:
            shortages[key] = target - actual
    return achieved, shortages


def make_summary(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    input_path: Path,
    validation_errors: list[str],
    validation_checks: dict[str, bool],
    deterministic_rebuild_matched: bool,
) -> dict[str, Any]:
    usage = Counter(row["source_id"] for row in rows)
    pair_ids = sorted({row["pair_id"] for row in rows if row["pair_id"] is not None})
    rows_in_pairs = sum(row["is_counterfactual_pair"] for row in rows)
    core_rows = sum(row["decision"] in {"KEEP", "REVISE"} for row in rows)
    achieved, shortages = target_achievement(rows)
    return {
        "schema_version": "phase3_behavior_selection_v1",
        "selection_seed": SEED,
        "input": {
            "path": input_path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": sha256_file(input_path),
            "total_rows": len(source_rows),
        },
        "determinism": {
            "stable_pre_sampling_order": "dataset ASC, normalized source_index ASC, source_id ASC",
            "sampling_algorithm": "SHA-256 rank of '<seed>|<selection_context>|<source_id>', then stable source position",
            "output_order": "selection_id ASC; JSON keys sorted; UTF-8 with LF newlines",
            "seed": SEED,
            "independent_in_memory_rebuild_byte_identical": deterministic_rebuild_matched,
        },
        "total_selected_rows": len(rows),
        "unique_source_ids": len(usage),
        "rows_per_unique_source": round(len(rows) / len(usage), 6),
        "counts_by_behavior": dict(sorted(Counter(row["behavior"] for row in rows).items())),
        "counts_by_condition": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "counts_by_decision": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "counts_by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "counts_by_dataset": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "counts_by_bucket": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "cross_tabs": {
            "behavior_x_domain": nested_counts(rows, "behavior", "domain"),
            "behavior_x_dataset": nested_counts(rows, "behavior", "dataset"),
            "condition_x_domain": nested_counts(rows, "condition", "domain"),
            "condition_x_dataset": nested_counts(rows, "condition", "dataset"),
            "bucket_x_condition": nested_counts(rows, "bucket", "condition"),
            "decision_x_domain": nested_counts(rows, "decision", "domain"),
            "decision_x_dataset": nested_counts(rows, "decision", "dataset"),
        },
        "target_cells": achieved,
        "shortage_by_target_cell": shortages,
        "counterfactual_pair_count": len(pair_ids),
        "rows_in_counterfactual_pairs": rows_in_pairs,
        "percentage_rows_paired": round(100 * rows_in_pairs / len(rows), 4),
        "percentage_core_behavior_rows_paired": round(100 * rows_in_pairs / core_rows, 4),
        "max_source_usage": max(usage.values(), default=0),
        "sources_used_once": sum(count == 1 for count in usage.values()),
        "sources_used_twice": sum(count == 2 for count in usage.values()),
        "sources_used_more_than_twice": sorted(source_id for source_id, count in usage.items() if count > 2),
        "shortcut_diagnostics": build_shortcut_diagnostics(rows),
        "validation_checks": validation_checks,
        "validation_errors": validation_errors,
        "design_notes": {
            "code_keep": "Uses 140 unique CC sources (24 MBPP, 116 APPS); all 26 code WC sources are reserved for NORMAL anchors.",
            "code_revise": "Each condition uses 35 MBPP and 50 APPS WW sources to keep both datasets represented and reduce dataset/decision shortcut pressure.",
            "normal_cw_exception": "Code NORMAL needs 60 rows but only 26 code WC sources exist, so 34 CW sources are also selected with reference/Base-correct target requirements and an explicit two-row-cap justification.",
            "cw_semantics": "CW is never ordinary REVISE; every CW source is retained as regression_recovery.",
            "content_boundary": "Manifest contains metadata and references only; no prompts, tests, solutions, teacher outputs, or full model outputs are copied.",
        },
    }


def make_source_usage_summary(
    rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    selected_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        selected_by_source[row["source_id"]].append(row)
    usage = Counter(
        {source_id: len(selected) for source_id, selected in selected_by_source.items()}
    )

    def usage_cross(dimension: str) -> dict[str, dict[str, int]]:
        values = sorted({str(row[dimension]) for row in source_rows})
        return {
            value: {
                "available_sources": sum(str(row[dimension]) == value for row in source_rows),
                "unused_sources": sum(
                    str(row[dimension]) == value and str(row["id"]) not in selected_by_source
                    for row in source_rows
                ),
                "used_once": sum(
                    str(row[dimension]) == value and usage[str(row["id"])] == 1
                    for row in source_rows
                ),
                "used_twice": sum(
                    str(row[dimension]) == value and usage[str(row["id"])] == 2
                    for row in source_rows
                ),
            }
            for value in values
        }

    dual_non_pair = []
    for source_id, selected in sorted(selected_by_source.items()):
        if len(selected) != 2 or all(row["is_counterfactual_pair"] for row in selected):
            continue
        dual_non_pair.append(
            {
                "source_id": source_id,
                "conditions": sorted(row["condition"] for row in selected),
                "justifications": sorted(
                    {row["source_usage_justification"] for row in selected if row["source_usage_justification"]}
                ),
            }
        )
    return {
        "selection_seed": SEED,
        "available_source_count": len(source_rows),
        "selected_unique_source_count": len(selected_by_source),
        "unselected_source_count": len(source_rows) - len(selected_by_source),
        "selected_behavior_row_count": len(rows),
        "max_behavior_rows_per_source": MAX_ROWS_PER_SOURCE,
        "actual_max_source_usage": max(usage.values(), default=0),
        "usage_histogram": {
            "0": len(source_rows) - len(selected_by_source),
            "1": sum(count == 1 for count in usage.values()),
            "2": sum(count == 2 for count in usage.values()),
            ">2": sum(count > 2 for count in usage.values()),
        },
        "usage_by_dataset": usage_cross("dataset"),
        "usage_by_domain": usage_cross("domain"),
        "usage_by_bucket": usage_cross("bucket"),
        "counterfactual_pair_source_count": len(
            {row["source_id"] for row in rows if row["is_counterfactual_pair"]}
        ),
        "non_pair_sources_used_twice_with_explicit_justification": dual_non_pair,
        "sources_used_more_than_twice": sorted(source_id for source_id, count in usage.items() if count > 2),
    }


def make_unused_rows(
    rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_ids = {row["source_id"] for row in rows}
    return [
        {
            "source_id": source["id"],
            "dataset": source["dataset"],
            "domain": source["domain"],
            "bucket": source["bucket"],
            "source_origin": source["source_origin"],
            "eligible_behaviors": source["eligible_behaviors"],
            "unused_reason": "not_selected_by_fixed_target_matrix_or_reserved_by_source_cap",
            "selection_seed": SEED,
        }
        for source in sorted(source_rows, key=stable_source_key)
        if str(source["id"]) not in selected_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_rows = read_jsonl(input_path)
    inventory_errors = validate_source_inventory(source_rows)
    if inventory_errors:
        raise RuntimeError("Unified inventory validation failed: " + json.dumps(inventory_errors[:20]))

    selected, _selector = build_selection(source_rows)
    selected_repeat, _repeat_selector = build_selection(source_rows)
    deterministic_rebuild_matched = jsonl_bytes(selected) == jsonl_bytes(selected_repeat)
    validation_errors, validation_checks = validate_selection(selected, source_rows)
    validation_checks["deterministic_rebuild_matches"] = deterministic_rebuild_matched
    validation_checks["all_passed"] = (
        validation_checks["all_passed"] and deterministic_rebuild_matched
    )
    if not deterministic_rebuild_matched:
        validation_errors.append("Independent deterministic rebuild did not match byte-for-byte")
    if validation_errors:
        raise RuntimeError("Selection validation failed: " + json.dumps(validation_errors[:20]))

    manifest_path = output_dir / "behavior_selection_manifest.jsonl"
    summary_path = output_dir / "behavior_selection_summary.json"
    usage_path = output_dir / "source_usage_summary.json"
    unused_path = output_dir / "rejected_or_unused.jsonl"

    write_jsonl(manifest_path, selected)
    summary = make_summary(
        selected,
        source_rows,
        input_path,
        validation_errors,
        validation_checks,
        deterministic_rebuild_matched,
    )
    write_json(summary_path, summary)
    write_json(usage_path, make_source_usage_summary(selected, source_rows))
    write_jsonl(unused_path, make_unused_rows(selected, source_rows))

    concise = {
        "output_files": {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
            "source_usage": str(usage_path),
            "rejected_or_unused": str(unused_path),
        },
        "total_selected_rows": summary["total_selected_rows"],
        "unique_source_problems": summary["unique_source_ids"],
        "KEEP_REVISE_NORMAL_RECOVERY": summary["counts_by_decision"],
        "math_vs_code": summary["counts_by_domain"],
        "GSM8K_MBPP_APPS": summary["counts_by_dataset"],
        "counterfactual_pair_percentage_all_rows": summary["percentage_rows_paired"],
        "counterfactual_pair_percentage_core_rows": summary["percentage_core_behavior_rows_paired"],
        "maximum_source_usage_count": summary["max_source_usage"],
        "target_shortages": summary["shortage_by_target_cell"],
        "validation_errors": summary["validation_errors"],
        "deterministic_in_memory_rebuild_matched": deterministic_rebuild_matched,
        "output_sha256": {
            "manifest": sha256_file(manifest_path),
            "summary": sha256_file(summary_path),
            "source_usage": sha256_file(usage_path),
            "rejected_or_unused": sha256_file(unused_path),
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
