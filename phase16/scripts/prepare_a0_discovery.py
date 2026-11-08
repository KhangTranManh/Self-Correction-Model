"""Build and validate the CPU candidate bank for P16 A0 residual discovery.

The bank is source-screening material, not a certified correction/evaluation
pool.  It keeps raw third-party text and tests in a gitignored private pack and
writes a compact, non-text public summary.  Dataset solutions are parsed but
never executed by this script; the GPU handoff runner must execute them in its
networkless sandbox before freezing an A0 screen lock.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


TACO_REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MIN_TESTS = 5
MIN_PARSEABLE_SOLUTIONS = 2
MAX_REFERENCE_SOLUTIONS = 3
MAX_QUESTION_CHARS = 30_000
MAX_IO_CHARS = 500_000
MAX_SOLUTION_CHARS = 100_000
NEAR_DUPLICATE_THRESHOLD = 0.80
A0_TARGET = 400
ORDER_SEED = "p16-a0-discovery-v1-2026-11-07"
WORD = re.compile(r"[A-Za-z0-9_]+")
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_hash(value: dict, field: str) -> dict:
    return {key: item for key, item in value.items() if key != field}


def strict_json(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def render_prompt(question: str, function_name: str, starter_code: str) -> str:
    parts = [
        "Implement the requested Python callable.",
        "",
        "Problem:",
        question.strip(),
        "",
        f"Required callable name: {function_name}",
    ]
    if starter_code.strip():
        parts.extend(["", "Starter interface:", starter_code.strip()])
    parts.extend([
        "",
        "Return only Python code. Define either the required top-level callable "
        "or the requested method on class Solution. Do not read stdin and do not print.",
    ])
    return "\n".join(parts)


def parse_call_tests(raw: object) -> tuple[list, list] | None:
    if not isinstance(raw, str) or len(raw) > MAX_IO_CHARS:
        return None
    try:
        value = strict_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("fn_name"), str):
        return None
    inputs, outputs = value.get("inputs"), value.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if len(inputs) != len(outputs) or len(inputs) < MIN_TESTS:
        return None
    if not all(isinstance(arguments, list) for arguments in inputs):
        return None
    try:
        canonical_bytes({"inputs": inputs, "outputs": outputs})
    except (TypeError, ValueError):
        return None
    return inputs, outputs


def solution_form(code: str, function_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return None
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
        for node in tree.body
    ):
        return "top_level"
    if any(
        isinstance(node, ast.ClassDef)
        and node.name == "Solution"
        and any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == function_name
            for child in node.body
        )
        for node in tree.body
    ):
        return "Solution_method"
    return None


def parse_reference_solutions(raw: object, function_name: str) -> list[dict]:
    if not isinstance(raw, str) or len(raw) > 10 * MAX_SOLUTION_CHARS:
        return []
    try:
        values = strict_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    records, seen = [], set()
    for position, code in enumerate(values):
        if not isinstance(code, str) or not code.strip() or len(code) > MAX_SOLUTION_CHARS:
            continue
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        form = solution_form(code, function_name)
        if form is None:
            continue
        seen.add(digest)
        records.append({
            "position": position,
            "form": form,
            "code": code,
            "code_sha256": digest,
        })
        if len(records) == MAX_REFERENCE_SOLUTIONS:
            break
    return records


def five_grams(text: str) -> frozenset[tuple[str, ...]]:
    tokens = WORD.findall(text.casefold())
    if len(tokens) < 5:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[index:index + 5])
                     for index in range(len(tokens) - 4))


def near_duplicate_components(tasks: list[dict]) -> list[list[int]]:
    """Return connected components under token 5-gram Jaccard >= 0.80."""
    grams = [five_grams(task["question"]) for task in tasks]
    parent = list(range(len(tasks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    inverted: dict[tuple[str, ...], list[int]] = {}
    pair_intersections: Counter[tuple[int, int]] = Counter()
    for index, item_grams in enumerate(grams):
        for gram in item_grams:
            prior = inverted.setdefault(gram, [])
            for other in prior:
                pair_intersections[(other, index)] += 1
            prior.append(index)
    for (left, right), intersection in pair_intersections.items():
        union_size = len(grams[left]) + len(grams[right]) - intersection
        if union_size and intersection / union_size >= NEAR_DUPLICATE_THRESHOLD:
            union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(tasks)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda group: tasks[group[0]]["item_id"])


def _load_exclusions(overlap: dict) -> tuple[set[str], set[str]]:
    historical = {row["id"] for row in overlap["historical_exact_hits"]}
    duplicate_later = {
        row_id
        for group in overlap["within_taco_exact_text_groups"]
        for row_id in group[1:]
    }
    return historical, duplicate_later


def _ordered(tasks: Iterable[dict]) -> list[dict]:
    return sorted(tasks, key=lambda task: (
        hashlib.sha256(f"{ORDER_SEED}:{task['item_id']}".encode()).hexdigest(),
        task["item_id"],
    ))


def build_bank(raw_dir: Path, metadata_path: Path, overlap_path: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to build the A0 candidate bank") from exc

    metadata = [json.loads(line) for line in metadata_path.open(encoding="utf-8")]
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if overlap["metadata_sha256"] != sha256_file(metadata_path):
        raise ValueError("overlap/metadata hash mismatch")
    historical, duplicate_later = _load_exclusions(overlap)
    rejections: Counter[str] = Counter()
    tasks: list[dict] = []
    row_index = 0
    shards = sorted((raw_dir / "ALL").glob("train-*.parquet"))
    if len(shards) != 9:
        raise ValueError(f"expected nine TACO train shards, found {len(shards)}")
    columns = ["question", "input_output", "starter_code", "solutions"]
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                if row_index >= len(metadata):
                    raise ValueError("raw data has more rows than metadata")
                meta = metadata[row_index]
                expected_id = f"taco_train_{row_index:05d}"
                if meta["id"] != expected_id:
                    raise ValueError(f"metadata order mismatch at {row_index}")
                row_index += 1
                if meta["io_mode"] != "call_based":
                    rejections["not_call_based"] += 1
                    continue
                if meta["picture_flag"]:
                    rejections["picture_flag"] += 1
                    continue
                if meta["id"] in historical:
                    rejections["historical_exact_overlap"] += 1
                    continue
                if meta["id"] in duplicate_later:
                    rejections["later_exact_duplicate"] += 1
                    continue
                function_name = meta.get("fn_name")
                if not isinstance(function_name, str) or not IDENTIFIER.fullmatch(function_name):
                    rejections["invalid_function_name"] += 1
                    continue
                if meta["test_count"] < MIN_TESTS:
                    rejections["too_few_tests"] += 1
                    continue
                if (meta.get("solution_count") or 0) < MIN_PARSEABLE_SOLUTIONS:
                    rejections["too_few_declared_solutions"] += 1
                    continue
                question = row.get("question")
                if (not isinstance(question, str) or not question.strip()
                        or len(question) > MAX_QUESTION_CHARS):
                    rejections["invalid_question"] += 1
                    continue
                parsed_tests = parse_call_tests(row.get("input_output"))
                if parsed_tests is None:
                    rejections["invalid_call_tests"] += 1
                    continue
                inputs, outputs = parsed_tests
                references = parse_reference_solutions(row.get("solutions"), function_name)
                if len(references) < MIN_PARSEABLE_SOLUTIONS:
                    rejections["too_few_parseable_solutions"] += 1
                    continue
                starter = row.get("starter_code")
                starter = starter if isinstance(starter, str) else ""
                prompt = render_prompt(question, function_name, starter)
                test_payload = {"inputs": inputs, "outputs": outputs}
                tasks.append({
                    "item_id": meta["id"],
                    "row_index": row_index - 1,
                    "dataset_id": "BAAI/TACO",
                    "source_revision": TACO_REVISION,
                    "source_split": "train",
                    "source_site": meta["source"],
                    "source_id_sha256": canonical_sha256({
                        "url": meta["canonical_url"],
                        "normalized_question_sha256": meta["normalized_question_sha256"],
                    }),
                    "question": question,
                    "normalized_question_sha256": meta["normalized_question_sha256"],
                    "function_name": function_name,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "test_inputs": inputs,
                    "test_outputs": outputs,
                    "tests_sha256": canonical_sha256(test_payload),
                    "reference_solutions": references,
                    "history_exact_clean": True,
                    "exact_duplicate_representative": True,
                    "dataset_oracle_only": True,
                    "independent_oracle_validated": False,
                })
    if row_index != len(metadata):
        raise ValueError(f"raw/metadata row mismatch: {row_index} vs {len(metadata)}")

    components = near_duplicate_components(tasks)
    for group in components:
        member_ids = sorted(tasks[index]["item_id"] for index in group)
        family_id = "taco-nd-" + canonical_sha256(member_ids)[:20]
        for index in group:
            tasks[index]["family_id"] = family_id
            tasks[index]["near_duplicate_component_size"] = len(group)
    tasks = _ordered(tasks)
    bank = {
        "schema_version": "p16_a0_candidate_bank_v1",
        "purpose": "inference_only_residual_discovery",
        "dataset_id": "BAAI/TACO",
        "source_revision": TACO_REVISION,
        "source_split": "train",
        "test_split_opened": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selection": {
            "minimum_tests": MIN_TESTS,
            "minimum_parseable_reference_solutions": MIN_PARSEABLE_SOLUTIONS,
            "maximum_saved_reference_solutions": MAX_REFERENCE_SOLUTIONS,
            "near_duplicate_method": "token_5gram_jaccard_connected_components",
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "order_seed": ORDER_SEED,
            "a0_target": A0_TARGET,
        },
        "source_artifacts": {
            "metadata_sha256": sha256_file(metadata_path),
            "overlap_sha256": sha256_file(overlap_path),
            "shard_sha256": [
                {"name": shard.name, "sha256": sha256_file(shard)} for shard in shards
            ],
        },
        "rejection_counts": dict(sorted(rejections.items())),
        "tasks": tasks,
        "claims": {
            "source_structural_candidate_only": True,
            "independent_oracle_certified": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
    }
    bank["bank_sha256"] = canonical_sha256(bank)
    return bank


def public_summary(bank: dict) -> dict:
    tasks = bank["tasks"]
    records = [{
        "item_id": task["item_id"],
        "row_index": task["row_index"],
        "source_site": task["source_site"],
        "family_id": task["family_id"],
        "prompt_sha256": task["prompt_sha256"],
        "tests_sha256": task["tests_sha256"],
        "reference_solution_sha256": [
            row["code_sha256"] for row in task["reference_solutions"]
        ],
    } for task in tasks]
    summary = {
        "schema_version": "p16_a0_candidate_bank_summary_v1",
        "purpose": bank["purpose"],
        "bank_sha256": bank["bank_sha256"],
        "source_revision": bank["source_revision"],
        "test_split_opened": False,
        "candidate_count": len(tasks),
        "family_count": len({task["family_id"] for task in tasks}),
        "source_counts": dict(sorted(Counter(task["source_site"] for task in tasks).items())),
        "near_duplicate_multirow_components": len({
            task["family_id"] for task in tasks
            if task["near_duplicate_component_size"] > 1
        }),
        "selection": bank["selection"],
        "rejection_counts": bank["rejection_counts"],
        "records": records,
        "authorization": {
            "a0_inference_authorized": False,
            "condition": "remote sandbox/tokenizer preflight must freeze 400 passing families",
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "limits": [
            "Dataset tests and reference solutions are discovery evidence, not an independent oracle.",
            "Any residual correction or held-out role requires the full CPU admission contract.",
            "The A0 screen is prevalence/discovery only and cannot estimate an update effect.",
        ],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def validate_bank(bank: dict, summary: dict | None = None) -> dict:
    if bank.get("schema_version") != "p16_a0_candidate_bank_v1":
        raise ValueError("unsupported candidate bank schema")
    if bank.get("bank_sha256") != canonical_sha256(without_hash(bank, "bank_sha256")):
        raise ValueError("candidate bank hash mismatch")
    if bank.get("source_revision") != TACO_REVISION or bank.get("test_split_opened") is not False:
        raise ValueError("unexpected source revision or test split state")
    tasks = bank.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < A0_TARGET:
        raise ValueError("candidate bank cannot supply the fixed A0 target")
    ids, prompts = set(), set()
    families = set()
    for task in tasks:
        item_id = task.get("item_id")
        if item_id in ids:
            raise ValueError("duplicate item id")
        ids.add(item_id)
        if task.get("prompt_sha256") != hashlib.sha256(
                task.get("prompt", "").encode("utf-8")).hexdigest():
            raise ValueError(f"prompt hash mismatch: {item_id}")
        if task["prompt_sha256"] in prompts:
            raise ValueError("duplicate prompt hash")
        prompts.add(task["prompt_sha256"])
        if task.get("tests_sha256") != canonical_sha256({
                "inputs": task.get("test_inputs"), "outputs": task.get("test_outputs")}):
            raise ValueError(f"tests hash mismatch: {item_id}")
        references = task.get("reference_solutions")
        if not isinstance(references, list) or len(references) < MIN_PARSEABLE_SOLUTIONS:
            raise ValueError(f"insufficient reference solutions: {item_id}")
        for reference in references:
            if reference["code_sha256"] != hashlib.sha256(
                    reference["code"].encode("utf-8")).hexdigest():
                raise ValueError(f"reference hash mismatch: {item_id}")
        if task.get("independent_oracle_validated") is not False:
            raise ValueError("candidate bank must not claim independent oracle certification")
        families.add(task.get("family_id"))
    if None in families:
        raise ValueError("missing family id")
    if summary is not None:
        if summary.get("summary_sha256") != canonical_sha256(
                without_hash(summary, "summary_sha256")):
            raise ValueError("public summary hash mismatch")
        expected = public_summary(bank)
        if summary != expected:
            raise ValueError("public summary does not match private bank")
    return {
        "valid": True,
        "bank_sha256": bank["bank_sha256"],
        "candidate_count": len(tasks),
        "family_count": len(families),
        "a0_inference_authorized": False,
        "gpu_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("overlap", type=Path)
    parser.add_argument("private_bank", type=Path)
    parser.add_argument("public_summary", type=Path)
    args = parser.parse_args()
    for output in (args.private_bank, args.public_summary):
        if output.exists():
            parser.error(f"refusing to overwrite {output}")
    bank = build_bank(args.raw_dir, args.metadata, args.overlap)
    summary = public_summary(bank)
    validate_bank(bank, summary)
    args.private_bank.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.private_bank.write_text(
        json.dumps(bank, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.public_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validate_bank(bank, summary), sort_keys=True))


if __name__ == "__main__":
    main()
