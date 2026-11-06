"""Build an excluded 12-task reference smoke pack from pinned TACO train."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def build(raw_dir: Path, preflight_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    chosen = {
        int(row["id"].split("_")[-1]): row
        for row in preflight["rows"]
        if "non_sequence_call_input" not in row["flags"]
    }
    tasks = []
    row_index = 0
    for shard in sorted((raw_dir / "ALL").glob("train-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=256,
                                          columns=["input_output", "solutions"]):
            for row in batch.to_pylist():
                if row_index in chosen:
                    io = json.loads(row["input_output"], parse_int=str)
                    solutions = json.loads(row["solutions"])
                    fn = io["fn_name"]
                    parsed = []
                    for position, code in enumerate(solutions):
                        if not isinstance(code, str):
                            continue
                        try:
                            tree = ast.parse(code)
                        except SyntaxError:
                            continue
                        top = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                  and node.name == fn for node in tree.body)
                        method = any(isinstance(node, ast.ClassDef)
                                     and node.name == "Solution"
                                     and any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                                             and child.name == fn for child in node.body)
                                     for node in tree.body)
                        if top or method:
                            parsed.append((position, code, "top_level" if top else "Solution_method"))
                    if parsed:
                        position, code, form = parsed[0]
                    else:
                        position, code, form = None, None, "no_matching_parseable_solution"
                    tasks.append({
                        "id": chosen[row_index]["id"],
                        "function_name": fn,
                        "test_inputs": io["inputs"],
                        "test_outputs": io["outputs"],
                        "solution_position": position,
                        "solution_form": form,
                        "solution_code": code,
                        "solution_sha256": hashlib.sha256((code or "").encode()).hexdigest(),
                    })
                row_index += 1
    if len(tasks) != len(chosen):
        raise SystemExit(f"found {len(tasks)} of {len(chosen)} tasks")
    pack = {
        "purpose": "excluded_engineering_reference_smoke_only",
        "source_revision": "d593ed0a2becbbc952230bb89be09189bf1056dc",
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "tasks": tasks,
    }
    output_path.write_text(json.dumps(pack, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(tasks),
                      "forms": {form: sum(t["solution_form"] == form for t in tasks)
                                for form in sorted({t["solution_form"] for t in tasks})},
                      "pack_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("preflight_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    build(args.raw_dir, args.preflight_path, args.output_path)
