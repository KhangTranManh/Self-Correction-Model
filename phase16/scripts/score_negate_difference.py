"""Conservative CPU scorer for an explicit -(a-b) intermediate expansion.

Only the small grammar {a,b, integer, +,-,* , parentheses} is accepted.
Input is a structured transcript; extracting the expansion from free-form
model text is a separate, still-unvalidated step.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import sympy as sp


A, B = sp.symbols("a b")


def parse_expression(source: str) -> sp.Expr:
    if not isinstance(source, str) or not source.strip() or len(source) > 120:
        raise ValueError("missing or oversized expression")
    node = ast.parse(source, mode="eval")
    if sum(1 for _ in ast.walk(node)) > 40:
        raise ValueError("expression too complex")

    def convert(item: ast.AST) -> sp.Expr:
        if isinstance(item, ast.Expression):
            return convert(item.body)
        if isinstance(item, ast.Name) and item.id in {"a", "b"}:
            return {"a": A, "b": B}[item.id]
        if isinstance(item, ast.Constant) and type(item.value) is int and abs(item.value) <= 1000:
            return sp.Integer(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            child = convert(item.operand)
            return child if isinstance(item.op, ast.UAdd) else -child
        if isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = convert(item.left), convert(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            return left * right
        raise ValueError("unsupported expression syntax")

    return convert(node)


def score(record: dict) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get("task_id"), str) or not record["task_id"]:
        raise ValueError("missing task_id")
    a, b = record.get("a"), record.get("b")
    if type(a) is not int or type(b) is not int or b == 0 or max(abs(a), abs(b)) > 1000:
        raise ValueError("a,b must be bounded integers and b nonzero")
    status = record.get("status")
    if status not in {"ok", "missing", "ambiguous"}:
        raise ValueError("invalid extraction status")
    result = {"task_id": record["task_id"], "full_correct": False,
              "label": "unscorable", "expansion_correct": False,
              "final_correct": False}
    if status != "ok":
        return result
    try:
        expansion = parse_expression(record.get("expansion"))
    except (ValueError, SyntaxError, RecursionError):
        return result
    final = record.get("final")
    if type(final) is not int:
        return result
    correct = sp.simplify(expansion - (-A + B)) == 0
    signature = sp.simplify(expansion - (-A - B)) == 0
    final_correct = final == -a + b
    full = bool(correct and final_correct)
    result.update(full_correct=full, expansion_correct=bool(correct),
                  final_correct=final_correct,
                  label="correct" if full else
                  "negate_difference_signature" if signature else "other_failure")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.transcript.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
