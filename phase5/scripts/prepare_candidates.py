"""CPU-only GSM8K candidate selection with conservative source exclusions.

The caller must supply the original train JSONL and every previous-phase
source/evaluation inventory via repeatable --exclude paths. This program does
not infer that an incomplete exclusion list is complete.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import unicodedata


STEP = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
FINAL = re.compile(r"####\s*([-+]?[\d,.]+)\s*$")
PRIOR_ID = re.compile(r"gsm8k[_-]?train[_-]?(\d+)", re.IGNORECASE)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def question_key(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def value(text: str) -> Fraction:
    tree = ast.parse(text.replace(",", ""), mode="eval")

    def calculate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return calculate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            result = calculate(node.operand)
            return -result if isinstance(node.op, ast.USub) else result
        if isinstance(node, ast.BinOp):
            left, right = calculate(node.left), calculate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("Unsupported arithmetic annotation")

    return calculate(tree)


def records(path: Path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    yield json.loads(line), number
    elif path.suffix == ".json":
        yield json.loads(path.read_text(encoding="utf-8")), 1


def nested_objects(node):
    if isinstance(node, dict):
        yield node
        for child in node.values():
            yield from nested_objects(child)
    elif isinstance(node, list):
        for child in node:
            yield from nested_objects(child)


def previous_sources(paths: list[Path]):
    questions, indices, files = set(), set(), {}
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(f"Missing exclusion path: {root}")
        inputs = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")
        )
        if not inputs:
            raise ValueError(f"No JSON/JSONL exclusion evidence at {root}")
        for path in inputs:
            files[str(path.resolve())] = digest(path)
            for record, _ in records(path):
                for obj in nested_objects(record):
                    question = obj.get("question")
                    if isinstance(question, str) and question.strip():
                        questions.add(question_key(question))
                    for key in ("id", "problem_id", "source_id"):
                        identifier = obj.get(key)
                        if isinstance(identifier, str) and (match := PRIOR_ID.search(identifier)):
                            indices.add(int(match.group(1)))
    if not questions and not indices:
        raise ValueError("Exclusion inputs yielded no source identifiers or questions")
    return questions, indices, files


def prepare(args):
    if args.budget <= 0:
        raise ValueError("Budget must be positive")
    excluded_questions, excluded_indices, inventory = previous_sources(args.exclude)
    accepted, reasons, seen = [], Counter(), set()
    raw_rows = list(records(args.train_jsonl))
    for index, (row, _) in enumerate(raw_rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"Unexpected GSM8K train schema at row {index}")
        question, answer = row["question"], row["answer"]
        key = question_key(question)
        if index in excluded_indices or key in excluded_questions:
            reasons["previous_phase_source"] += 1
            continue
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        if len(question.split()) > 120:
            reasons["question_too_long"] += 1
            continue
        steps = STEP.findall(answer)
        if not 2 <= len(steps) <= 6:
            reasons["step_count"] += 1
            continue
        try:
            for expression, declared in steps:
                if value(expression) != value(declared):
                    raise ValueError("Reference step mismatch")
            match = FINAL.search(answer)
            if match is None:
                raise ValueError("No numeric final answer")
            reference = str(value(match.group(1)))
        except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
            reasons["unverifiable_reference"] += 1
            continue
        accepted.append({
            "id": f"phase5_gsm8k_train_{index}", "dataset_index": index,
            "domain": "math", "question": question,
            "reference_answer": reference, "question_sha256": key,
            "reference_step_count": len(steps),
        })
    accepted.sort(key=lambda row: hashlib.sha256(
        f"phase5_source_v1|{row['id']}".encode("utf-8")
    ).hexdigest())
    selected = accepted[: args.budget]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "candidate_problems.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in selected), encoding="utf-8", newline="\n")
    report = {
        "status": "cpu_candidates_only_no_model_answers", "source": "GSM8K original train",
        "raw_train_sha256": digest(args.train_jsonl), "raw_train_rows": len(raw_rows),
        "exclusion_files_sha256": inventory,
        "excluded_question_hashes": len(excluded_questions),
        "excluded_gsm8k_indices": len(excluded_indices),
        "eligible_after_cpu_rules": len(accepted), "selected": len(selected),
        "target_budget": args.budget, "rejected_by_first_reason": dict(sorted(reasons.items())),
        "candidate_problems_sha256": digest(output),
        "selection_key": "SHA256(phase5_source_v1|phase5_gsm8k_train_<index>)",
        "confirmation_opened": False, "gpu_used": False,
    }
    (args.output_dir / "candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True, action="append",
                        help="Repeat for each prior-phase source/evaluation inventory path")
    parser.add_argument("--output-dir", type=Path, default=Path("phase5/data/candidates_v1"))
    parser.add_argument("--budget", type=int, default=1200)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
