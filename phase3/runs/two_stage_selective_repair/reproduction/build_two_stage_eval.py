"""Freeze a balanced evaluation manifest for two-stage selective repair.

The manifest reuses cached Self_Correction_v1 initial generations from the
canonical P0 log and the existing APPS pilot.  Selection is deterministic and
excludes every APPS source used by the Phase 3 behavior or decision datasets.
No model generation or training occurs here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SEED = 314159
P0_LOG = PROJECT_ROOT / "outputs" / "p0_trained_100_log.jsonl"
APPS_ATTEMPTS = ROOT / "data" / "apps_pilot" / "attempts" / "self_correction_v1" / "raw_attempts.jsonl"
APPS_CANDIDATES = ROOT / "data" / "apps_pilot" / "candidates" / "code_candidates.jsonl"
OUTPUT_DIR = ROOT / "data" / "two_stage_selective_repair"

QUOTAS = {
    "gsm8k": {"correct": 30, "wrong": 30},
    "mbpp": {"correct": 20, "wrong": 20},
    "svamp": {"correct": 20, "wrong": 20},
    "humaneval": {"correct": 10, "wrong": 10},
    "apps": {"correct": 20, "wrong": 20},
}

MATH_PROMPT = (
    "Giải bài toán sau từng bước, sau đó ghi rõ đáp số cuối cùng theo định dạng "
    "'Đáp số: <giá trị>'.\n\nBài toán: {question}"
)
CODE_PROMPT = (
    "{question}\n\nChỉ trả về code Python hoàn chỉnh trong 1 code block "
    "(```python ... ```), không giải thích thêm."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(dataset: str, state: str, source_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{dataset}|{state}|{source_id}".encode()).hexdigest()


def gsm_answer(raw: str) -> str:
    match = re.search(r"####\s*(.+)$", raw, flags=re.MULTILINE)
    if not match:
        raise ValueError("GSM8K answer lacks #### marker")
    return match.group(1).replace(",", "").strip()


def entry_point(code: str) -> str:
    matches = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", code, flags=re.MULTILINE)
    if not matches:
        raise ValueError("Cannot extract MBPP entry point")
    return matches[0]


def used_apps_sources() -> set[str]:
    used: set[str] = set()
    patterns = [
        ROOT / "data" / "decision_only" / "*.jsonl",
        ROOT / "data" / "behavior" / "construction_pilot" / "*.jsonl",
        ROOT / "data" / "behavior" / "construction_pilot_v2" / "*.jsonl",
        ROOT / "data" / "behavior" / "mini_train" / "*.jsonl",
        ROOT / "data" / "behavior" / "mini_train_v2" / "*.jsonl",
    ]
    for pattern in patterns:
        for path in pattern.parent.glob(pattern.name):
            for row in read_jsonl(path):
                source_id = row.get("source_id")
                if isinstance(source_id, str) and source_id.startswith("apps_"):
                    used.add(source_id)
    return used


def select_cached_rows() -> list[dict[str, Any]]:
    p0_map = {"GSM8K": "gsm8k", "MBPP": "mbpp", "SVAMP": "svamp", "HumanEval": "humaneval"}
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(read_jsonl(P0_LOG), start=1):
        if row.get("scenario") != "initial" or row.get("source") not in p0_map:
            continue
        dataset = p0_map[row["source"]]
        state = "correct" if row["initial_correct"] else "wrong"
        pools[(dataset, state)].append(
            {
                "dataset": dataset,
                "source_id": row["problem_id"],
                "domain": row["domain"],
                "question": row["question"],
                "initial_answer": row["initial_answer"],
                "initial_correct": row["initial_correct"],
                "initial_verifier_detail_cached": row["initial_detail"],
                "initial_finish_reason_cached": row.get("initial_finish_reason"),
                "initial_generation_source": "canonical_p0_self_correction_v1_initial",
                "source_ref": f"outputs/p0_trained_100_log.jsonl#line={line_number}",
            }
        )

    excluded = used_apps_sources()
    candidates = {row["id"]: row for row in read_jsonl(APPS_CANDIDATES)}
    for line_number, row in enumerate(read_jsonl(APPS_ATTEMPTS), start=1):
        if row["id"] in excluded:
            continue
        candidate = candidates[row["id"]]
        state = "correct" if row["initial_correct"] else "wrong"
        pools[("apps", state)].append(
            {
                "dataset": "apps",
                "source_id": row["id"],
                "domain": "code",
                "question": candidate["problem"],
                "initial_answer": row["initial_output"],
                "initial_correct": row["initial_correct"],
                "initial_verifier_detail_cached": row["verifier_detail"],
                "initial_finish_reason_cached": row.get("finish_reason"),
                "initial_generation_source": "phase3_apps_pilot_self_correction_v1_initial",
                "source_ref": f"data/apps_pilot/attempts/self_correction_v1/raw_attempts.jsonl#line={line_number}",
                "apps_candidate": candidate,
            }
        )

    selected: list[dict[str, Any]] = []
    for dataset, states in QUOTAS.items():
        for state, count in states.items():
            available = sorted(
                pools[(dataset, state)],
                key=lambda row: rank(dataset, state, row["source_id"]),
            )
            if len(available) < count:
                raise RuntimeError(f"Insufficient {dataset}/{state}: {len(available)} < {count}")
            selected.extend(available[:count])
    return sorted(selected, key=lambda row: rank(row["dataset"], "final", row["source_id"]))


def attach_verifier_payloads(rows: list[dict[str, Any]]) -> None:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    mbpp = load_dataset(
        "parquet",
        data_files="hf://datasets/google-research-datasets/mbpp@refs%2Fconvert%2Fparquet/full/test/0000.parquet",
        split="train",
    )
    mbpp_by_id = {int(row["task_id"]): row for row in mbpp}
    svamp = load_dataset("ChilleD/SVAMP", split="test")
    svamp_by_id = {str(row["ID"]): row for row in svamp}
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    humaneval_by_id = {str(row["task_id"]): row for row in humaneval}

    for row in rows:
        dataset = row["dataset"]
        if dataset == "gsm8k":
            index = int(row["source_id"].rsplit("_", 1)[1])
            source = gsm[index]
            row["verifier"] = {"type": "math", "reference_answer": gsm_answer(source["answer"])}
        elif dataset == "mbpp":
            task_id = int(row["source_id"].rsplit("_", 1)[1])
            source = mbpp_by_id[task_id]
            tests = list(source["test_list"])
            if source.get("test_setup_code"):
                tests.insert(0, source["test_setup_code"])
            row["verifier"] = {
                "type": "python_asserts",
                "entry_point": entry_point(source["code"]),
                "tests": tests,
            }
        elif dataset == "svamp":
            source = svamp_by_id[row["source_id"].removeprefix("ood_svamp_")]
            row["verifier"] = {"type": "math", "reference_answer": str(source["Answer"])}
        elif dataset == "humaneval":
            task_id = row["source_id"].removeprefix("ood_").replace("_", "/", 1)
            source = humaneval_by_id[task_id]
            row["verifier"] = {
                "type": "python_asserts",
                "entry_point": source["entry_point"],
                "tests": [source["test"], f"check({source['entry_point']})"],
            }
        elif dataset == "apps":
            candidate = row.pop("apps_candidate")
            row["verifier"] = {"type": "apps", "input_output": candidate["tests"]}
        else:
            raise ValueError(dataset)
        template = MATH_PROMPT if row["domain"] == "math" else CODE_PROMPT
        row["task_prompt"] = template.format(question=row["question"])
        row["expected_decision"] = "KEEP" if row["initial_correct"] else "REVISE"
        row["selection_seed"] = SEED
        row["schema_version"] = "phase3_two_stage_frozen_eval_v1"


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["source_id"] for row in rows]
    by_dataset_state = Counter(
        (row["dataset"], "correct" if row["initial_correct"] else "wrong") for row in rows
    )
    expected = {
        (dataset, state): count
        for dataset, states in QUOTAS.items()
        for state, count in states.items()
    }
    used = used_apps_sources()
    checks = {
        "total_200": len(rows) == 200,
        "unique_sources": len(ids) == len(set(ids)),
        "quota_match": dict(by_dataset_state) == expected,
        "initial_state_balanced": sum(row["initial_correct"] for row in rows) == 100,
        "domain_balanced": Counter(row["domain"] for row in rows) == {"math": 100, "code": 100},
        "domain_state_balanced": Counter(
            (row["domain"], row["initial_correct"]) for row in rows
        ) == {("math", True): 50, ("math", False): 50, ("code", True): 50, ("code", False): 50},
        "no_apps_training_or_prior_eval_overlap": not any(
            row["dataset"] == "apps" and row["source_id"] in used for row in rows
        ),
        "all_v1_cached": all("self_correction_v1" in row["initial_generation_source"] for row in rows),
        "all_have_verifiers": all(row.get("verifier", {}).get("type") for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))
    return {
        "schema_version": "phase3_two_stage_frozen_eval_summary_v1",
        "seed": SEED,
        "total_rows": len(rows),
        "counts_by_dataset_state": {
            dataset: {
                state: by_dataset_state[(dataset, state)] for state in ("correct", "wrong")
            }
            for dataset in QUOTAS
        },
        "counts_by_domain": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "counts_by_domain_state": {
            domain: {
                state: sum(
                    row["domain"] == domain
                    and row["initial_correct"] == (state == "correct")
                    for row in rows
                )
                for state in ("correct", "wrong")
            }
            for domain in ("math", "code")
        },
        "validation": checks,
        "selection_note": (
            "Cached V1 initial generations are reused to preserve the frozen P0/APPS state; "
            "no ground-truth or verifier result is supplied to either inference stage."
        ),
    }


def main() -> None:
    rows = select_cached_rows()
    attach_verifier_payloads(rows)
    summary = validate(rows)
    manifest = OUTPUT_DIR / "frozen_eval.jsonl"
    write_jsonl(manifest, rows)
    summary["manifest_sha256"] = sha256(manifest)
    summary_path = OUTPUT_DIR / "frozen_eval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
