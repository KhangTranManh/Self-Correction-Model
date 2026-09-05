"""Construct and verify the small Phase 3 behavior-construction pilot.

The script samples 25 rows per selected behavior condition and materializes
training-shaped ``messages``. It never calls a teacher model and does not scale
to the complete 860-row selection manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
PHASE1_ROOT = ROOT.parent / "phase1"
sys.path.insert(0, str(PHASE1_ROOT))

from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.code import CodeVerifier  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402

from apps_verifier import verify as verify_apps  # noqa: E402


SEED = 314159
DEFAULT_MANIFEST = ROOT / "data" / "behavior" / "selection" / "behavior_selection_manifest.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "behavior" / "construction_pilot"
CONDITIONS = (
    "preserve_neutral",
    "preserve_false_feedback",
    "repair_neutral",
    "repair_true_feedback",
    "normal_solve",
    "regression_recovery",
)
CONDITION_DECISION = {
    "preserve_neutral": "KEEP",
    "preserve_false_feedback": "KEEP",
    "repair_neutral": "REVISE",
    "repair_true_feedback": "REVISE",
    "normal_solve": "NORMAL",
    "regression_recovery": "RECOVER",
}
FEEDBACK_TYPE = {
    "preserve_neutral": "neutral",
    "preserve_false_feedback": "false_negative",
    "repair_neutral": "neutral",
    "repair_true_feedback": "true_verifier",
    "normal_solve": "none",
    "regression_recovery": "none",
}
EXPECTED_BUCKET_STATE = {
    "CC": (True, True),
    "WW": (False, False),
    "WC": (False, True),
    "CW": (True, False),
}

NEUTRAL_TEMPLATES = (
    "Review your previous answer carefully and decide whether it should be kept or revised.",
    "Recheck the previous response without assuming it is right or wrong, then decide whether to keep or revise it.",
    "Evaluate your previous answer carefully. Preserve it if it is correct; otherwise revise it.",
    "Inspect the answer again and choose whether it should remain unchanged or be corrected.",
)
FALSE_NEGATIVE_TEMPLATES = (
    "Your previous answer appears to be incorrect. Recheck it and decide whether it should be kept or revised.",
    "The earlier response may contain an error. Review it carefully before deciding whether to keep or revise it.",
    "A reviewer claims that your previous answer is wrong. Verify that claim and decide whether to keep or revise the answer.",
    "The previous result has been flagged as incorrect. Check the answer yourself and decide whether it needs revision.",
)
TRUE_VERIFIER_TEMPLATES = (
    "The deterministic verifier found objective evidence that the previous answer failed. {evidence} Recheck and correct the answer.",
    "Executable or reference-based verification rejected the earlier response. {evidence} Revise it to produce a correct answer.",
    "The previous answer did not pass the objective checker. {evidence} Review the work and provide a corrected answer.",
    "Objective verification shows that the earlier response fails. {evidence} Repair the answer.",
)

DECISION_TAG_RE = re.compile(r"<decision>(KEEP|REVISE)</decision>")
ANSWER_CONTRACT_RE = re.compile(
    r"\A<decision>(KEEP|REVISE)</decision>\n<answer>\n(.*)\n</answer>\Z",
    flags=re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


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


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_rank(context: str, identifier: str) -> str:
    return sha256_text(f"{SEED}|{context}|{identifier}")


def sanitize_verifier_detail(detail: str) -> str:
    """Remove nondeterministic temporary paths without changing verifier meaning."""
    detail = re.sub(r'File "[^"\n]*\.py"', 'File "<verifier_script>.py"', detail)
    detail = re.sub(r"(?:[A-Za-z]:\\|/)(?:[^\s:\"']+[\\/])+[^\s:\"']+\.py", "<verifier_script>.py", detail)
    return detail.strip()


class ProvenanceResolver:
    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def resolve(self, reference: str, expected_id: str) -> dict[str, Any]:
        path_text, separator, fragment = reference.partition("#id=")
        if not separator or fragment != expected_id:
            raise ValueError(
                f"Invalid provenance reference for {expected_id}: {reference!r}"
            )
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Provenance path does not exist: {path}")
        if path not in self._cache:
            index: dict[str, dict[str, Any]] = {}
            for row in read_jsonl(path):
                row_id = str(row.get("id", row.get("problem_id")))
                if row_id in index:
                    raise ValueError(f"Duplicate id={row_id!r} in {path}")
                index[row_id] = row
            self._cache[path] = index
        if expected_id not in self._cache[path]:
            raise KeyError(f"id={expected_id!r} not found in {path}")
        return self._cache[path][expected_id]


class PilotSampler:
    def __init__(self, manifest_rows: list[dict[str, Any]]) -> None:
        self.rows = sorted(manifest_rows, key=lambda row: row["selection_id"])
        self.selected: list[dict[str, Any]] = []
        self.selected_ids: set[str] = set()
        self.selected_sources: set[str] = set()

    def _add(self, row: dict[str, Any]) -> None:
        selection_id = str(row["selection_id"])
        if selection_id in self.selected_ids:
            raise RuntimeError(f"Pilot selection duplicated {selection_id}")
        self.selected.append(row)
        self.selected_ids.add(selection_id)
        self.selected_sources.add(str(row["source_id"]))

    def add_pairs(
        self,
        *,
        context: str,
        first_condition: str,
        second_condition: str,
        domain: str,
        count: int,
        dataset: str | None = None,
    ) -> None:
        by_pair: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            if not row["is_counterfactual_pair"] or not row["pair_id"]:
                continue
            if row["domain"] != domain or (dataset and row["dataset"] != dataset):
                continue
            if row["condition"] not in {first_condition, second_condition}:
                continue
            by_pair[str(row["pair_id"])].append(row)
        candidates = []
        for pair_id, members in by_pair.items():
            conditions = {row["condition"] for row in members}
            source_ids = {row["source_id"] for row in members}
            if conditions == {first_condition, second_condition} and len(source_ids) == 1:
                candidates.append((pair_id, members))
        candidates.sort(key=lambda item: (stable_rank(context, item[0]), item[0]))
        candidates = [item for item in candidates if item[1][0]["source_id"] not in self.selected_sources]
        if len(candidates) < count:
            raise RuntimeError(f"Insufficient complete pairs for {context}: {len(candidates)} < {count}")
        for _pair_id, members in candidates[:count]:
            for row in sorted(members, key=lambda value: value["condition"]):
                self._add(row)

    def add_singles(
        self,
        *,
        context: str,
        condition: str,
        domain: str,
        count: int,
        dataset: str | None = None,
        bucket: str | None = None,
        require_unpaired: bool = True,
    ) -> None:
        candidates = [
            row
            for row in self.rows
            if row["condition"] == condition
            and row["domain"] == domain
            and (dataset is None or row["dataset"] == dataset)
            and (bucket is None or row["bucket"] == bucket)
            and (not require_unpaired or not row["is_counterfactual_pair"])
            and row["source_id"] not in self.selected_sources
        ]
        candidates.sort(
            key=lambda row: (
                stable_rank(context, str(row["selection_id"])),
                str(row["selection_id"]),
            )
        )
        if len(candidates) < count:
            raise RuntimeError(f"Insufficient rows for {context}: {len(candidates)} < {count}")
        for row in candidates[:count]:
            self._add(row)


def sample_pilot(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sampler = PilotSampler(manifest_rows)

    # Ten complete preserve pairs: 5 math, 2 MBPP, 3 APPS.
    for domain, dataset, count in (
        ("math", None, 5),
        ("code", "mbpp", 2),
        ("code", "apps", 3),
    ):
        sampler.add_pairs(
            context=f"pilot_preserve_pair_{domain}_{dataset or 'all'}",
            first_condition="preserve_neutral",
            second_condition="preserve_false_feedback",
            domain=domain,
            dataset=dataset,
            count=count,
        )
    for condition in ("preserve_neutral", "preserve_false_feedback"):
        for domain, dataset, count in (
            ("math", None, 7),
            ("code", "mbpp", 1),
            ("code", "apps", 7),
        ):
            sampler.add_singles(
                context=f"pilot_{condition}_{domain}_{dataset or 'all'}",
                condition=condition,
                domain=domain,
                dataset=dataset,
                count=count,
            )

    # Ten complete repair pairs: 5 math, 2 MBPP, 3 APPS.
    for domain, dataset, count in (
        ("math", None, 5),
        ("code", "mbpp", 2),
        ("code", "apps", 3),
    ):
        sampler.add_pairs(
            context=f"pilot_repair_pair_{domain}_{dataset or 'all'}",
            first_condition="repair_neutral",
            second_condition="repair_true_feedback",
            domain=domain,
            dataset=dataset,
            count=count,
        )
    for condition in ("repair_neutral", "repair_true_feedback"):
        for domain, dataset, count in (
            ("math", None, 7),
            ("code", "mbpp", 4),
            ("code", "apps", 4),
        ):
            sampler.add_singles(
                context=f"pilot_{condition}_{domain}_{dataset or 'all'}",
                condition=condition,
                domain=domain,
                dataset=dataset,
                count=count,
            )

    # NORMAL: 12 math WC plus 13 code rows, prioritizing 8 WC over 5 CW.
    for domain, dataset, bucket, count in (
        ("math", None, "WC", 12),
        ("code", "mbpp", "WC", 5),
        ("code", "apps", "WC", 3),
        ("code", "apps", "CW", 5),
    ):
        sampler.add_singles(
            context=f"pilot_normal_{domain}_{dataset or 'all'}_{bucket}",
            condition="normal_solve",
            domain=domain,
            dataset=dataset,
            bucket=bucket,
            count=count,
        )

    # RECOVERY: 12 math + 13 code, disjoint from pilot NORMAL sources.
    for domain, dataset, count in (
        ("math", None, 12),
        ("code", "mbpp", 5),
        ("code", "apps", 8),
    ):
        sampler.add_singles(
            context=f"pilot_recovery_{domain}_{dataset or 'all'}",
            condition="regression_recovery",
            domain=domain,
            dataset=dataset,
            bucket="CW",
            count=count,
        )

    condition_position = {condition: index for index, condition in enumerate(CONDITIONS)}
    selected = sorted(
        sampler.selected,
        key=lambda row: (condition_position[row["condition"]], row["selection_id"]),
    )
    counts = Counter(row["condition"] for row in selected)
    if len(selected) != 150 or any(counts[condition] != 25 for condition in CONDITIONS):
        raise RuntimeError(f"Pilot sampling target mismatch: total={len(selected)}, counts={counts}")
    return selected


class VerificationEngine:
    def __init__(self) -> None:
        self.math = MathVerifier()
        self.mbpp = CodeVerifier(timeout_seconds=5, memory_limit_mb=256)
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    @staticmethod
    def method(source: dict[str, Any]) -> str:
        if source["dataset"] == "gsm8k":
            return "sympy_final_answer_match"
        if source["dataset"] == "mbpp":
            return "mbpp_executable_unit_tests"
        if source["dataset"] == "apps":
            return "apps_executable_tests"
        raise ValueError(f"Unsupported dataset: {source['dataset']}")

    def verify(self, source: dict[str, Any], candidate: str) -> dict[str, Any]:
        cache_key = (str(source["dataset"]), str(source["id"]), sha256_text(candidate))
        if cache_key in self.cache:
            return dict(self.cache[cache_key])
        if source["dataset"] == "gsm8k":
            problem = Problem(
                id=source["id"],
                domain="math",
                question=source["problem"],
                reference_answer=source["reference_answer"],
                calc_steps=source.get("calc_steps") or None,
            )
            result = self.math.verify(problem, candidate)
            value = {"passed": result.passed, "detail": sanitize_verifier_detail(result.detail)}
        elif source["dataset"] == "mbpp":
            problem = Problem(
                id=source["id"],
                domain="code",
                question=source["problem"],
                entry_point=source["entry_point"],
                tests=list(source["tests"]),
            )
            result = self.mbpp.verify(problem, candidate)
            value = {"passed": result.passed, "detail": sanitize_verifier_detail(result.detail)}
        elif source["dataset"] == "apps":
            result = verify_apps(
                candidate,
                source["tests"],
                per_test_timeout_seconds=3,
                memory_limit_mb=512,
            )
            value = {
                "passed": bool(result["passed"]),
                "detail": sanitize_verifier_detail(str(result["detail"])),
                "passed_tests": int(result["passed_tests"]),
                "total_tests": int(result["total_tests"]),
            }
        else:
            raise ValueError(f"Unsupported dataset: {source['dataset']}")
        self.cache[cache_key] = dict(value)
        return value


def problem_for_prompt(source: dict[str, Any]) -> Problem:
    return Problem(
        id=source["id"],
        domain=source["domain"],
        question=source["problem"],
        reference_answer=source.get("reference_answer"),
        entry_point=source.get("entry_point"),
        tests=list(source.get("tests", [])) if isinstance(source.get("tests", []), list) else [],
        calc_steps=source.get("calc_steps") or None,
    )


def reference_target(source: dict[str, Any]) -> tuple[str, str]:
    if source["dataset"] == "gsm8k":
        solution = source.get("reference_solution")
        if not isinstance(solution, str) or not solution.strip():
            raise KeyError("missing_reference_target")
        cleaned = re.sub(r"<<[^<>]*>>", "", solution)
        cleaned = re.sub(
            r"(?m)^####\s*(.+?)\s*$",
            lambda match: f"Đáp số: {match.group(1).strip()}",
            cleaned,
        )
        return cleaned.strip(), "gsm8k_reference_annotation_cleanup"
    solution = source.get("ground_truth") or source.get("reference_answer")
    if not isinstance(solution, str) or not solution.strip():
        raise KeyError("missing_reference_target")
    return f"```python\n{solution.strip()}\n```", "reference_code_fenced"


def template_choice(
    templates: Sequence[str], template_prefix: str, selection_id: str
) -> tuple[str, str]:
    index = int(stable_rank(f"feedback_template:{template_prefix}", selection_id), 16) % len(templates)
    return templates[index], f"{template_prefix}_{index + 1:02d}"


def verifier_evidence(source: dict[str, Any], initial_result: dict[str, Any]) -> str:
    detail = str(initial_result["detail"])
    if source["domain"] == "math":
        return "The final numeric result does not match under the deterministic reference-answer check."
    if source["dataset"] == "apps":
        match = re.search(r"test_(\d+):([^:]+)(?::(.*))?", detail, flags=re.DOTALL)
        if match:
            test_index, status, remainder = match.groups()
            if status == "wrong_answer":
                return f"Executable test {test_index} produced output that did not match the expected output."
            if status == "timeout":
                return f"Executable test {test_index} timed out."
            if remainder:
                last_line = remainder.strip().splitlines()[-1][:240]
                return f"Executable test {test_index} failed with {status}; final error: {last_line}"
            return f"Executable test {test_index} failed with {status}."
        return "The submitted program failed the executable APPS tests."
    last_line = detail.strip().splitlines()[-1][:240] if detail.strip() else "unit test failure"
    if "AssertionError" in detail:
        return "Executable MBPP unit tests raised AssertionError."
    if "Timeout" in detail:
        return "Executable MBPP unit tests timed out."
    return f"Executable MBPP unit tests failed; final error: {last_line}"


def make_feedback(
    selection: dict[str, Any], source: dict[str, Any], initial_result: dict[str, Any]
) -> tuple[str | None, str | None]:
    condition = selection["condition"]
    selection_id = selection["selection_id"]
    if condition in {"normal_solve", "regression_recovery"}:
        return None, None
    if condition in {"preserve_neutral", "repair_neutral"}:
        text, template_id = template_choice(NEUTRAL_TEMPLATES, "neutral", selection_id)
        return text, template_id
    if condition == "preserve_false_feedback":
        text, template_id = template_choice(
            FALSE_NEGATIVE_TEMPLATES, "false_negative", selection_id
        )
        return text, template_id
    if condition == "repair_true_feedback":
        template, template_id = template_choice(
            TRUE_VERIFIER_TEMPLATES, "true_verifier", selection_id
        )
        return template.format(evidence=verifier_evidence(source, initial_result)), template_id
    raise ValueError(f"Unknown condition: {condition}")


def behavior_contract(decision: str, target_answer: str) -> str:
    return f"<decision>{decision}</decision>\n<answer>\n{target_answer}\n</answer>"


def failure_record(
    selection: dict[str, Any], reason: str, detail: str, *, teacher_required: bool = False
) -> dict[str, Any]:
    return {
        "construction_id": f"pilot::{selection['selection_id']}",
        "selection_id": selection["selection_id"],
        "source_id": selection["source_id"],
        "dataset": selection["dataset"],
        "domain": selection["domain"],
        "bucket": selection["bucket"],
        "condition": selection["condition"],
        "reason": reason,
        "detail": detail,
        "requires_teacher": teacher_required,
        "teacher_used": False,
        "selection_seed": SEED,
    }


def construct_rows(
    pilot_selection: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    constructed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for selection in pilot_selection:
        try:
            source_id = str(selection["source_id"])
            source = resolver.resolve(selection["source_ref"], source_id)
            base_attempt = resolver.resolve(selection["base_attempt_ref"], source_id)
            v1_attempt = resolver.resolve(selection["v1_attempt_ref"], source_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            failures.append(failure_record(selection, "source_resolution_failure", str(exc)))
            continue

        provenance_mismatches = []
        for field in ("dataset", "domain"):
            if source.get(field) != selection.get(field):
                provenance_mismatches.append(field)
        if str(source.get("split")) != str(selection.get("source_split")):
            provenance_mismatches.append("source_split")
        if base_attempt.get("initial_correct") is not selection.get("base_correct"):
            provenance_mismatches.append("base_correct")
        if v1_attempt.get("initial_correct") is not selection.get("v1_correct"):
            provenance_mismatches.append("v1_correct")
        if provenance_mismatches:
            failures.append(
                failure_record(
                    selection,
                    "source_resolution_failure",
                    f"Provenance mismatch fields: {sorted(provenance_mismatches)}",
                )
            )
            continue

        v1_output = str(v1_attempt.get("initial_output", ""))
        try:
            initial_result = verifier.verify(source, v1_output)
        except Exception as exc:  # verifier errors become explicit rejected rows
            failures.append(failure_record(selection, "verifier_failure", repr(exc)))
            continue
        if bool(initial_result["passed"]) != bool(selection["v1_correct"]):
            failures.append(
                failure_record(
                    selection,
                    "verifier_failure",
                    "Fresh V1 verification disagrees with the selected bucket state: "
                    + json.dumps(initial_result, ensure_ascii=False, sort_keys=True),
                )
            )
            continue

        condition = selection["condition"]
        decision = CONDITION_DECISION[condition]
        if decision == "KEEP":
            target_answer = v1_output
            target_source = "v1_correct"
            target_transform = "none_preserve_exact_output"
        else:
            try:
                target_answer, target_transform = reference_target(source)
            except KeyError as exc:
                failures.append(
                    failure_record(
                        selection,
                        "missing_reference_target",
                        str(exc),
                        teacher_required=decision == "REVISE",
                    )
                )
                continue
            target_source = "reference"

        try:
            target_result = verifier.verify(source, target_answer)
        except Exception as exc:
            failures.append(failure_record(selection, "verifier_failure", repr(exc)))
            continue
        if not target_result["passed"]:
            failures.append(
                failure_record(
                    selection,
                    "teacher_required" if decision == "REVISE" else "verifier_failure",
                    "Reference/preserved target failed verification: "
                    + json.dumps(target_result, ensure_ascii=False, sort_keys=True),
                    teacher_required=decision == "REVISE",
                )
            )
            continue

        feedback, feedback_template_id = make_feedback(selection, source, initial_result)
        task_prompt = build_prompt(problem_for_prompt(source))
        if decision in {"KEEP", "REVISE"}:
            messages = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": v1_output},
                {"role": "user", "content": feedback},
                {
                    "role": "assistant",
                    "content": behavior_contract(decision, target_answer),
                },
            ]
        else:
            messages = [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": target_answer},
            ]

        constructed.append(
            {
                "construction_id": f"pilot::{selection['selection_id']}",
                "selection_id": selection["selection_id"],
                "source_id": source_id,
                "dataset": selection["dataset"],
                "domain": selection["domain"],
                "bucket": selection["bucket"],
                "condition": condition,
                "decision": decision,
                "initial_correct": bool(initial_result["passed"]),
                "base_correct": bool(selection["base_correct"]),
                "v1_correct": bool(selection["v1_correct"]),
                "feedback_type": FEEDBACK_TYPE[condition],
                "feedback_template_id": feedback_template_id,
                "feedback_text": feedback,
                "target_verified": True,
                "verification_method": verifier.method(source),
                "initial_verifier_detail": initial_result["detail"],
                "target_verifier_detail": target_result["detail"],
                "target_source": target_source,
                "target_transform": target_transform,
                "target_answer_sha256": sha256_text(target_answer),
                "teacher_used": False,
                "requires_teacher": False,
                "is_counterfactual_pair": bool(selection["is_counterfactual_pair"]),
                "pair_id": selection["pair_id"],
                "source_origin": selection["source_origin"],
                "source_ref": selection["source_ref"],
                "base_attempt_ref": selection["base_attempt_ref"],
                "v1_attempt_ref": selection["v1_attempt_ref"],
                "source_payload_sha256": sha256_text(
                    json.dumps(source, ensure_ascii=False, sort_keys=True)
                ),
                "base_attempt_payload_sha256": sha256_text(
                    json.dumps(base_attempt, ensure_ascii=False, sort_keys=True)
                ),
                "v1_attempt_payload_sha256": sha256_text(
                    json.dumps(v1_attempt, ensure_ascii=False, sort_keys=True)
                ),
                "selection_seed": SEED,
                "training_format": "messages_final_assistant_target",
                "messages": messages,
            }
        )

    condition_position = {condition: index for index, condition in enumerate(CONDITIONS)}
    constructed.sort(
        key=lambda row: (condition_position[row["condition"]], row["construction_id"])
    )
    failures.sort(key=lambda row: row["construction_id"])
    return constructed, failures


def extract_target_answer(row: dict[str, Any]) -> str:
    final = str(row["messages"][-1]["content"])
    if row["decision"] in {"KEEP", "REVISE"}:
        match = ANSWER_CONTRACT_RE.fullmatch(final)
        if not match:
            raise ValueError(f"Malformed behavioral contract: {row['construction_id']}")
        return match.group(2)
    return final


def nested_counts(
    rows: Sequence[dict[str, Any]], outer: str, inner: str, *, none_label: str = "none"
) -> dict[str, dict[str, int]]:
    def normalized(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return none_label if value is None else str(value)

    outer_values = sorted({normalized(row, outer) for row in rows})
    inner_values = sorted({normalized(row, inner) for row in rows})
    return {
        outer_value: {
            inner_value: sum(
                normalized(row, outer) == outer_value
                and normalized(row, inner) == inner_value
                for row in rows
            )
            for inner_value in inner_values
        }
        for outer_value in outer_values
    }


def duplicate_stats(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    duplicate_groups = sorted(
        (
            {"sha256": sha256_text(value), "count": count}
            for value, count in counts.items()
            if count > 1
        ),
        key=lambda item: (-item["count"], item["sha256"]),
    )
    duplicate_rows_beyond_first = sum(item["count"] - 1 for item in duplicate_groups)
    return {
        "total": len(values),
        "unique": len(counts),
        "duplicate_rows_beyond_first": duplicate_rows_beyond_first,
        "duplicate_rate": round(duplicate_rows_beyond_first / len(values), 6) if values else 0.0,
        "duplicate_groups": duplicate_groups,
    }


def validate_constructed(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    pilot_selection: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, bool], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_by_id = {row["selection_id"]: row for row in manifest_rows}
    requested_ids = {row["selection_id"] for row in pilot_selection}
    outcome_ids = {row["selection_id"] for row in rows} | {
        row["selection_id"] for row in failures
    }
    if requested_ids != outcome_ids:
        errors.append("Constructed/failure outcomes do not exactly cover pilot selection")

    construction_ids = [row["construction_id"] for row in rows]
    if len(construction_ids) != len(set(construction_ids)):
        errors.append("Duplicate construction IDs")

    pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        manifest = manifest_by_id.get(row["selection_id"])
        if manifest is None:
            errors.append(f"Unknown manifest selection: {row['selection_id']}")
            continue
        for field in (
            "source_id",
            "dataset",
            "domain",
            "bucket",
            "condition",
            "decision",
            "is_counterfactual_pair",
            "pair_id",
        ):
            if row[field] != manifest[field]:
                errors.append(f"{row['construction_id']}: manifest mismatch in {field}")
        if row["target_verified"] is not True:
            errors.append(f"{row['construction_id']}: target is not verified")
        expected_base, expected_v1 = EXPECTED_BUCKET_STATE[row["bucket"]]
        if (row["base_correct"], row["v1_correct"]) != (expected_base, expected_v1):
            errors.append(f"{row['construction_id']}: bucket state mismatch")
        if row["decision"] == "KEEP" and not row["initial_correct"]:
            errors.append(f"{row['construction_id']}: KEEP initial answer is not correct")
        if row["decision"] == "REVISE" and (row["initial_correct"] or row["bucket"] != "WW"):
            errors.append(f"{row['construction_id']}: ordinary REVISE source is not wrong WW")
        if row["condition"] == "regression_recovery" and row["bucket"] != "CW":
            errors.append(f"{row['construction_id']}: recovery is not CW")
        if "humaneval" in row["dataset"].casefold():
            errors.append(f"{row['construction_id']}: HumanEval leakage")
        if not (
            (row["dataset"] == "gsm8k" and "gsm8k_train_" in row["source_id"])
            or (row["dataset"] == "mbpp" and not row["source_id"].startswith("mbpp_test_"))
            or (row["dataset"] == "apps" and row["source_id"].startswith("apps_train_"))
        ):
            errors.append(f"{row['construction_id']}: possible frozen-eval leakage")

        roles = [message.get("role") for message in row["messages"]]
        final = str(row["messages"][-1].get("content", ""))
        if row["decision"] in {"KEEP", "REVISE"}:
            if roles != ["user", "assistant", "user", "assistant"]:
                errors.append(f"{row['construction_id']}: invalid review chat roles")
            match = ANSWER_CONTRACT_RE.fullmatch(final)
            if not match or match.group(1) != row["decision"]:
                errors.append(f"{row['construction_id']}: invalid decision/answer contract")
        else:
            if roles != ["user", "assistant"]:
                errors.append(f"{row['construction_id']}: invalid fresh-task chat roles")
            if DECISION_TAG_RE.search(final) or "<answer>" in final or "</answer>" in final:
                errors.append(f"{row['construction_id']}: fresh-task target has correction scaffold")

        feedback = row["feedback_text"]
        if row["condition"] in {"preserve_neutral", "repair_neutral"}:
            if feedback not in NEUTRAL_TEMPLATES:
                errors.append(f"{row['construction_id']}: neutral feedback not from neutral templates")
            # Neutral language may explicitly say "without assuming it is right
            # or wrong". Reject assertions of error, not the mere word "wrong".
            if re.search(
                r"(?:your|the) previous answer (?:is|appears to be) (?:incorrect|wrong)"
                r"|previous answer (?:failed|contains an error)"
                r"|earlier response (?:failed|is incorrect|is wrong)",
                feedback or "",
                re.IGNORECASE,
            ):
                errors.append(f"{row['construction_id']}: neutral feedback reveals error")
        elif row["condition"] == "preserve_false_feedback":
            if feedback not in FALSE_NEGATIVE_TEMPLATES or not row["initial_correct"]:
                errors.append(f"{row['construction_id']}: invalid false-negative semantics")
        elif row["condition"] == "repair_true_feedback":
            if not feedback or row["initial_correct"] or row["feedback_type"] != "true_verifier":
                errors.append(f"{row['construction_id']}: invalid true-verifier semantics")
        elif feedback is not None or row["feedback_template_id"] is not None:
            errors.append(f"{row['construction_id']}: fresh task unexpectedly has feedback")

        if row["is_counterfactual_pair"] and row["pair_id"]:
            pairs[row["pair_id"]].append(row)

    for pair_id, members in pairs.items():
        if len(members) != 2:
            errors.append(f"{pair_id}: pilot did not preserve both pair members")
        if len({row["source_id"] for row in members}) != 1:
            errors.append(f"{pair_id}: pair source mismatch")

    prompts = [
        json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True) for row in rows
    ]
    prompt_stats = duplicate_stats(prompts)
    if prompt_stats["duplicate_rows_beyond_first"]:
        warnings.append(
            f"duplicate_full_prompt_rows={prompt_stats['duplicate_rows_beyond_first']}"
        )

    target_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_groups[str(row["messages"][-1]["content"])].append(row)
    unexpected_target_groups = []
    for members in target_groups.values():
        if len(members) <= 1:
            continue
        pair_ids = {row["pair_id"] for row in members}
        source_ids = {row["source_id"] for row in members}
        if len(pair_ids) == 1 and None not in pair_ids and len(source_ids) == 1:
            continue
        unexpected_target_groups.append([row["construction_id"] for row in members])
    if unexpected_target_groups:
        warnings.append(f"unexpected_duplicate_target_groups={len(unexpected_target_groups)}")

    feedback_rows = [row for row in rows if row["feedback_text"] is not None]
    for decision in ("KEEP", "REVISE"):
        unique_feedback = {
            row["feedback_text"] for row in feedback_rows if row["decision"] == decision
        }
        if len(unique_feedback) <= 1:
            warnings.append(f"decision_{decision}_tied_to_one_unique_feedback_wording")
    domain_templates = {
        domain: {
            row["feedback_template_id"]
            for row in feedback_rows
            if row["domain"] == domain
        }
        for domain in ("math", "code")
    }
    if not domain_templates["math"].intersection(domain_templates["code"]):
        warnings.append("math_and_code_feedback_template_sets_are_disjoint")

    core = [row for row in rows if row["decision"] in {"KEEP", "REVISE"}]
    for dataset in sorted({row["dataset"] for row in core}):
        dataset_rows = [row for row in core if row["dataset"] == dataset]
        majority = max(Counter(row["decision"] for row in dataset_rows).values()) / len(dataset_rows)
        if majority >= 0.90:
            warnings.append(f"dataset_{dataset}_trivially_predicts_core_decision")

    checks = {
        "requested_selection_fully_accounted_for": requested_ids == outcome_ids,
        "unique_construction_ids": len(construction_ids) == len(set(construction_ids)),
        "all_selection_ids_exist_in_manifest": all(
            row["selection_id"] in manifest_by_id for row in rows
        ),
        "source_and_manifest_metadata_preserved": not any(
            "manifest mismatch" in error for error in errors
        ),
        "all_constructed_targets_verified": all(row["target_verified"] for row in rows),
        "keep_sources_objectively_correct": not any(
            "KEEP initial answer is not correct" in error for error in errors
        ),
        "ordinary_revise_sources_objectively_wrong_ww": not any(
            "ordinary REVISE source" in error for error in errors
        ),
        "false_feedback_only_on_verified_correct": not any(
            "invalid false-negative" in error for error in errors
        ),
        "neutral_feedback_does_not_reveal_correctness": not any(
            "neutral feedback reveals" in error for error in errors
        ),
        "normal_and_recovery_have_no_correction_scaffold": not any(
            "fresh-task target has correction scaffold" in error for error in errors
        ),
        "no_frozen_eval_or_humaneval_leakage": not any(
            "leakage" in error for error in errors
        ),
        "counterfactual_pairs_retained_completely": not any(
            "pilot did not preserve" in error or "pair source mismatch" in error
            for error in errors
        ),
    }
    checks["all_passed"] = all(checks.values()) and not errors
    return errors, checks, warnings


def verified_rate_by_condition(
    rows: list[dict[str, Any]], pilot_selection: list[dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        requested = sum(row["condition"] == condition for row in pilot_selection)
        constructed = [row for row in rows if row["condition"] == condition]
        verified = sum(row["target_verified"] for row in constructed)
        result[condition] = {
            "requested": requested,
            "constructed": len(constructed),
            "verified": verified,
            "verified_rate_of_constructed": round(verified / len(constructed), 6)
            if constructed
            else None,
            "verified_rate_of_requested": round(verified / requested, 6) if requested else None,
        }
    return result


def make_summary(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    pilot_selection: list[dict[str, Any]],
    manifest_path: Path,
    validation_errors: list[str],
    validation_checks: dict[str, bool],
    warnings: list[str],
) -> dict[str, Any]:
    feedback_rows = [row for row in rows if row["feedback_text"] is not None]
    prompt_values = [
        json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True) for row in rows
    ]
    target_values = [str(row["messages"][-1]["content"]) for row in rows]
    feedback_values = [str(row["feedback_text"]) for row in feedback_rows]
    pair_ids = {row["pair_id"] for row in rows if row["pair_id"] is not None}
    source_usage = Counter(row["source_id"] for row in rows)
    return {
        "schema_version": "phase3_behavior_construction_pilot_v1",
        "selection_seed": SEED,
        "input_manifest": {
            "path": manifest_path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": sha256_file(manifest_path),
            "total_selection_rows": len(read_jsonl(manifest_path)),
        },
        "sampling": {
            "algorithm": "SHA-256 rank of '<seed>|<pilot_context>|<selection_or_pair_id>'",
            "stable_input_order": "selection_id ASC",
            "requested_per_condition": 25,
            "requested_domain_split_per_condition": {"math": 12, "code": 13},
            "counterfactual_pairs_requested_and_selected": 20,
            "selected_selection_ids": [row["selection_id"] for row in pilot_selection],
        },
        "total_pilot_rows_requested": len(pilot_selection),
        "total_successfully_constructed": len(rows),
        "total_rejected": len(failures),
        "number_requiring_teacher_generation": sum(
            failure["requires_teacher"] for failure in failures
        ),
        "teacher_calls_made": 0,
        "counts_per_condition": dict(
            sorted(Counter(row["condition"] for row in rows).items())
        ),
        "counts_by_decision": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "counts_by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "counts_by_dataset": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "counts_by_bucket": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "verified_target_rate_per_condition": verified_rate_by_condition(
            rows, pilot_selection
        ),
        "target_source_distribution": dict(
            sorted(Counter(row["target_source"] for row in rows).items())
        ),
        "verification_method_distribution": dict(
            sorted(Counter(row["verification_method"] for row in rows).items())
        ),
        "counterfactual_pair_count": len(pair_ids),
        "rows_in_counterfactual_pairs": sum(
            row["is_counterfactual_pair"] for row in rows
        ),
        "unique_source_count": len(source_usage),
        "maximum_source_usage": max(source_usage.values(), default=0),
        "feedback_template_distribution": dict(
            sorted(
                Counter(
                    row["feedback_template_id"] or "none" for row in rows
                ).items()
            )
        ),
        "cross_tabs": {
            "decision_x_feedback_template": nested_counts(
                rows, "decision", "feedback_template_id"
            ),
            "domain_x_feedback_template": nested_counts(
                rows, "domain", "feedback_template_id"
            ),
            "dataset_x_feedback_template": nested_counts(
                rows, "dataset", "feedback_template_id"
            ),
            "decision_x_dataset": nested_counts(rows, "decision", "dataset"),
            "condition_x_domain": nested_counts(rows, "condition", "domain"),
            "condition_x_dataset": nested_counts(rows, "condition", "dataset"),
        },
        "duplicate_and_template_diagnostics": {
            "full_prompt": duplicate_stats(prompt_values),
            "final_target": duplicate_stats(target_values),
            "identical_feedback_wording": duplicate_stats(feedback_values),
            "warnings": warnings,
        },
        "failure_reason_distribution": dict(
            sorted(Counter(row["reason"] for row in failures).items())
        ),
        "structural_validation_results": validation_checks,
        "validation_errors": validation_errors,
        "pass_gates": {
            "all_requested_rows_constructed": len(rows) == len(pilot_selection),
            "all_keep_targets_verify": all(
                row["target_verified"] for row in rows if row["decision"] == "KEEP"
            ),
            "all_ready_revise_targets_verify": all(
                row["target_verified"] for row in rows if row["decision"] == "REVISE"
            ),
            "all_normal_targets_verify": all(
                row["target_verified"] for row in rows if row["decision"] == "NORMAL"
            ),
            "all_recovery_targets_verify": all(
                row["target_verified"] for row in rows if row["decision"] == "RECOVER"
            ),
            "no_teacher_used": not any(row["teacher_used"] for row in rows),
            "all_structural_checks_pass": validation_checks["all_passed"],
        },
        "notes": {
            "keep_target": "Exact verified V1 output is preserved inside the KEEP contract.",
            "revise_target": "Cleaned GSM8K reference or fenced dataset reference code; every target is reverified.",
            "normal_and_recovery_target": "Reference target only, with no decision/answer scaffold and never a CW V1-wrong output.",
            "teacher_policy": "No teacher is configured or called; unavailable verified REVISE targets are failures requiring_teacher.",
        },
    }


def abbreviate(value: str, limit: int = 480) -> str:
    normalized = value.replace("\r\n", "\n").strip()
    return normalized if len(normalized) <= limit else normalized[:limit].rstrip() + " … [truncated]"


def make_manual_audit(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    selected: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for domain in ("math", "code"):
            candidate = next(row for row in condition_rows if row["domain"] == domain)
            selected.append(candidate)

    lines = [
        "# Phase 3 behavior-construction pilot — manual audit",
        "",
        "This file abbreviates targets for inspection only. `pilot_behavior_rows.jsonl` retains every target in full.",
        "",
        "## Pilot status",
        "",
        f"- Requested: {summary['total_pilot_rows_requested']}",
        f"- Constructed: {summary['total_successfully_constructed']}",
        f"- Rejected: {summary['total_rejected']}",
        f"- Teacher calls: {summary['teacher_calls_made']}",
        f"- Structural checks passed: {summary['structural_validation_results']['all_passed']}",
        "",
        "## Representative examples",
        "",
    ]
    for row in selected:
        target = extract_target_answer(row)
        warning = "none"
        if row["condition"] == "preserve_false_feedback":
            warning = "Feedback is deliberately false; both initial and target were verified correct."
        elif row["bucket"] == "CW" and row["condition"] == "normal_solve":
            warning = "CW fresh-solve anchor; V1 output is wrong and is never used as target."
        elif row["bucket"] == "CW":
            warning = "CW regression-recovery case; V1 output is wrong and is never used as target."
        lines.extend(
            [
                f"### {row['condition']} — {row['domain']}",
                "",
                f"- Source: `{row['source_id']}`",
                f"- Dataset/domain: `{row['dataset']}` / `{row['domain']}`",
                f"- Bucket: `{row['bucket']}`",
                f"- Previous answer correct: `{str(row['initial_correct']).lower()}`",
                f"- Feedback template: `{row['feedback_template_id'] or 'none'}`",
                f"- Constructed feedback: {row['feedback_text'] or '(none — fresh task)' }",
                f"- Target decision: `{row['decision']}`",
                f"- Target source: `{row['target_source']}`",
                f"- Verifier: `{row['verification_method']}` → `{row['target_verifier_detail']}`",
                f"- Warning: {warning}",
                "- Abbreviated target:",
                "",
                f"<pre>{html.escape(abbreviate(target))}</pre>",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_rows = read_jsonl(manifest_path)
    pilot_selection = sample_pilot(manifest_rows)
    pilot_selection_repeat = sample_pilot(manifest_rows)
    deterministic_sampling_match = jsonl_bytes(pilot_selection) == jsonl_bytes(
        pilot_selection_repeat
    )
    if not deterministic_sampling_match:
        raise RuntimeError("Deterministic pilot selection rebuild did not match")

    rows, failures = construct_rows(pilot_selection)
    validation_errors, validation_checks, warnings = validate_constructed(
        rows, failures, pilot_selection, manifest_rows
    )
    validation_checks["deterministic_sampling_rebuild_matches"] = deterministic_sampling_match
    validation_checks["all_passed"] = validation_checks["all_passed"] and deterministic_sampling_match
    summary = make_summary(
        rows,
        failures,
        pilot_selection,
        manifest_path,
        validation_errors,
        validation_checks,
        warnings,
    )

    rows_path = output_dir / "pilot_behavior_rows.jsonl"
    summary_path = output_dir / "pilot_summary.json"
    failures_path = output_dir / "pilot_failures.jsonl"
    audit_path = output_dir / "pilot_manual_audit.md"
    write_jsonl(rows_path, rows)
    write_json(summary_path, summary)
    write_jsonl(failures_path, failures)
    write_text(audit_path, make_manual_audit(rows, summary))

    report = {
        "total_pilot_rows_requested": summary["total_pilot_rows_requested"],
        "total_successfully_constructed": summary["total_successfully_constructed"],
        "counts_per_condition": summary["counts_per_condition"],
        "math_code_counts": summary["counts_by_domain"],
        "dataset_counts": summary["counts_by_dataset"],
        "verified_target_rate_per_condition": summary[
            "verified_target_rate_per_condition"
        ],
        "number_requiring_teacher_generation": summary[
            "number_requiring_teacher_generation"
        ],
        "number_rejected": summary["total_rejected"],
        "feedback_template_distribution": summary[
            "feedback_template_distribution"
        ],
        "duplicate_template_warnings": warnings,
        "structural_validation_results": validation_checks,
        "output_paths": {
            "rows": str(rows_path),
            "summary": str(summary_path),
            "failures": str(failures_path),
            "manual_audit": str(audit_path),
        },
        "output_sha256": {
            "rows": sha256_file(rows_path),
            "summary": sha256_file(summary_path),
            "failures": sha256_file(failures_path),
            "manual_audit": sha256_file(audit_path),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
