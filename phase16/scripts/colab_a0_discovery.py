"""P16 A0 residual-discovery handoff: remote preflight and GPU shards.

This script does not train or update weights.  Run ``preflight`` before any
model generation; it freezes exactly 400 source-side tasks whose tokenizer
length fits and for which two dataset reference solutions reproduce all
published tests inside a networkless bubblewrap sandbox.  Then run four
immutable 100-item ``generate`` shards.

The resulting failures are discovery candidates only.  Dataset tests are not
an independent oracle, and no item enters a correction/evaluation pool until
it passes the separate CPU admission contract.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


EXPECTED_BANK_SHA256 = "1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SYSTEM_PROMPT = "Write the requested Python callable. Return only Python code."
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 768
TARGET = 400
SHARD_SIZE = 100
SHARD_COUNT = 4
SANDBOX_TIMEOUT_SECONDS = 8
RESULT_PREFIX = "__P16_A0_RESULT__="
PACKAGES = {
    "transformers": "4.46.3",
    "accelerate": "1.0.1",
    "bitsandbytes": "0.50.2",
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_new_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def ensure_packages() -> dict[str, str]:
    import importlib.metadata as metadata

    present = {}
    for name in PACKAGES:
        try:
            present[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            present[name] = None
    if present != PACKAGES:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
             *[f"{name}=={version}" for name, version in PACKAGES.items()]],
            check=True, timeout=900,
        )
    return {name: metadata.version(name) for name in PACKAGES}


def ensure_bwrap() -> None:
    if shutil.which("bwrap"):
        return
    subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=300)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "bubblewrap"],
        check=True, timeout=300,
    )


def sandbox_preflight() -> None:
    ensure_bwrap()
    result = subprocess.run([
        "bwrap", "--die-with-parent", "--unshare-user", "--uid", "65534",
        "--gid", "65534", "--unshare-net", "--unshare-pid",
        "--ro-bind", "/usr", "/usr", "--tmpfs", "/tmp", "--dev", "/dev",
        "/usr/bin/id", "-u",
    ], capture_output=True, text=True, timeout=SANDBOX_TIMEOUT_SECONDS)
    if result.returncode or result.stdout.strip() != "65534":
        raise RuntimeError("bubblewrap UID/network sandbox preflight failed")


def extract_code(text: str) -> tuple[str, str]:
    """Frozen reference-free recovery for a single Markdown code fence."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped, "unchanged"
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        opener = lines[0].strip().casefold()
        if opener in {"```", "```python", "```py"}:
            return "\n".join(lines[1:-1]).strip(), "accepted"
    return stripped, "rejected"


def callable_shape(code: str, function_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError):
        return None
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name == function_name for node in tree.body):
        return "top_level"
    if any(isinstance(node, ast.ClassDef) and node.name == "Solution"
           and any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and child.name == function_name for child in node.body)
           for node in tree.body):
        return "Solution_method"
    return None


CHILD = r'''
import bisect, collections, copy, functools, heapq, itertools, json, math, re
import resource, string, sys
from typing import Any, Dict, List, Optional, Set, Tuple
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
record = json.load(sys.stdin)
namespace = {
    "__name__": "__candidate__", "Any": Any, "Dict": Dict, "List": List,
    "Optional": Optional, "Set": Set, "Tuple": Tuple,
}
with open("/work/candidate.py", encoding="utf-8") as source:
    exec(compile(source.read(), "candidate.py", "exec"), namespace)
name = record["function_name"]
if name in namespace and callable(namespace[name]):
    function = namespace[name]
elif "Solution" in namespace:
    function = getattr(namespace["Solution"](), name)
else:
    raise RuntimeError("required callable missing")
answers = [function(*copy.deepcopy(args)) for args in record["test_inputs"]]
print("__P16_A0_RESULT__=" + json.dumps(answers, ensure_ascii=False))
'''


def sandbox_execute(code: str, task: dict) -> dict:
    shape = callable_shape(code, task["function_name"])
    if shape is None:
        return {"status": "syntax_or_interface_error"}
    with tempfile.TemporaryDirectory(prefix="p16_a0_") as directory:
        work = Path(directory)
        os.chmod(work, 0o755)
        candidate = work / "candidate.py"
        candidate.write_text(code + "\n", encoding="utf-8")
        os.chmod(candidate, 0o644)
        command = [
            "bwrap", "--die-with-parent", "--unshare-user", "--uid", "65534",
            "--gid", "65534", "--unshare-net", "--unshare-pid",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--tmpfs", "/tmp",
            "--ro-bind", str(work), "/work", "--dev", "/dev",
            "--chdir", "/tmp", "/usr/bin/python3", "-I", "-S", "-c", CHILD,
        ]
        try:
            result = subprocess.run(
                command,
                input=json.dumps({
                    "function_name": task["function_name"],
                    "test_inputs": task["test_inputs"],
                }),
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT_SECONDS,
                env={"PATH": "/usr/bin", "HOME": "/tmp",
                     "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "callable_shape": shape}
    if result.returncode:
        return {
            "status": "runtime_error",
            "callable_shape": shape,
            "returncode": result.returncode,
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        }
    marked = [line[len(RESULT_PREFIX):] for line in result.stdout.splitlines()
              if line.startswith(RESULT_PREFIX)]
    if len(marked) != 1:
        return {"status": "missing_result", "callable_shape": shape}
    try:
        actual = json.loads(marked[0])
    except json.JSONDecodeError:
        return {"status": "invalid_result_json", "callable_shape": shape}
    if not isinstance(actual, list) or len(actual) != len(task["test_outputs"]):
        return {"status": "result_count_mismatch", "callable_shape": shape}
    matches = [got == expected for got, expected in zip(actual, task["test_outputs"])]
    return {
        "status": "completed",
        "callable_shape": shape,
        "test_count": len(matches),
        "match_count": sum(matches),
        "all_correct": all(matches),
        "actual_sha256": canonical_sha256(actual),
    }


def load_inputs(root: Path) -> tuple[dict, dict]:
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from prepare_a0_discovery import validate_bank

    bank_path = root / "candidate_bank.json"
    summary_path = root / "a0_candidate_bank_summary_v1.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_bank(bank, summary)
    if bank["bank_sha256"] != EXPECTED_BANK_SHA256:
        raise RuntimeError("unexpected candidate bank hash")
    return bank, summary


def run_preflight(root: Path, output: Path) -> None:
    lock_path = output / "a0_lock.json"
    audit_path = output / "preflight_audit.jsonl"
    if lock_path.exists() or audit_path.exists():
        raise RuntimeError("refusing to overwrite preflight artifacts")
    versions = ensure_packages()
    sandbox_preflight()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True,
    )
    bank, summary = load_inputs(root)
    append_jsonl(audit_path, {
        "kind": "metadata",
        "bank_sha256": bank["bank_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "packages": versions,
        "target": TARGET,
        "test_split_opened": False,
        "gpu_training_authorized": False,
    })
    selected, seen_families = [], set()
    for task in bank["tasks"]:
        if task["family_id"] in seen_families:
            continue
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ]
        token_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True,
        )
        if len(token_ids) > MAX_INPUT_TOKENS:
            append_jsonl(audit_path, {
                "kind": "candidate_preflight", "item_id": task["item_id"],
                "decision": "reject", "reason": "prompt_too_long",
                "input_tokens": len(token_ids),
            })
            continue
        reference_reports = []
        for reference in task["reference_solutions"]:
            report = sandbox_execute(reference["code"], task)
            reference_reports.append({
                "code_sha256": reference["code_sha256"], **report,
            })
            if sum(row.get("all_correct") is True for row in reference_reports) >= 2:
                break
        passing = [row for row in reference_reports if row.get("all_correct") is True]
        decision = "accept" if len(passing) >= 2 else "reject"
        append_jsonl(audit_path, {
            "kind": "candidate_preflight",
            "item_id": task["item_id"],
            "family_id": task["family_id"],
            "input_tokens": len(token_ids),
            "decision": decision,
            "reason": None if decision == "accept" else "fewer_than_two_references_pass",
            "reference_reports": reference_reports,
        })
        if decision == "accept":
            seen_families.add(task["family_id"])
            selected.append({
                "position": len(selected),
                "item_id": task["item_id"],
                "family_id": task["family_id"],
                "prompt_sha256": task["prompt_sha256"],
                "tests_sha256": task["tests_sha256"],
                "input_tokens": len(token_ids),
                "passing_reference_sha256": [row["code_sha256"] for row in passing[:2]],
            })
        if len(selected) == TARGET:
            break
    if len(selected) != TARGET:
        append_jsonl(audit_path, {
            "kind": "preflight_failure", "selected": len(selected), "required": TARGET,
        })
        raise RuntimeError(f"preflight selected {len(selected)} of required {TARGET}")
    lock = {
        "schema_version": "p16_a0_discovery_lock_v1",
        "purpose": "inference_only_residual_discovery",
        "bank_sha256": bank["bank_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "generation": {
            "decoding": "greedy",
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "item_seed_method": "sha256_item_id_first_8_hex",
        },
        "recovery": {
            "method": "single_markdown_fence_v1",
            "reference_free": True,
        },
        "selected": selected,
        "source_preflight_complete": True,
        "test_split_opened": False,
        "independent_oracle_certified": False,
        "a0_inference_authorized": True,
        "gpu_training_authorized": False,
        "protected_data_opening_authorized": False,
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    write_new_json(lock_path, lock)
    append_jsonl(audit_path, {
        "kind": "a0_lock_frozen",
        "lock_sha256": lock["lock_sha256"],
        "selected": len(selected),
        "audit_prefix_sha256": sha256_file(audit_path),
        "gpu_training_authorized": False,
    })
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"], "selected": len(selected),
        "a0_inference_authorized": True, "gpu_training_authorized": False,
    }, sort_keys=True))


def load_lock(output: Path) -> dict:
    lock = json.loads((output / "a0_lock.json").read_text(encoding="utf-8"))
    claimed = lock.get("lock_sha256")
    actual = canonical_sha256({key: value for key, value in lock.items()
                               if key != "lock_sha256"})
    if claimed != actual or lock.get("bank_sha256") != EXPECTED_BANK_SHA256:
        raise RuntimeError("A0 lock hash mismatch")
    if (lock.get("a0_inference_authorized") is not True
            or lock.get("gpu_training_authorized") is not False
            or len(lock.get("selected", [])) != TARGET):
        raise RuntimeError("A0 lock authorization/state mismatch")
    return lock


def classify(raw_status: str, recovered_status: str, recovery_status: str) -> str:
    if recovered_status == "correct":
        return "presentation_only" if recovery_status == "accepted" else "raw_correct"
    if recovered_status == "wrong_answer":
        return "substantive_content"
    if recovered_status in {
            "syntax_or_interface_error", "runtime_error", "timeout",
            "missing_result", "invalid_result_json", "result_count_mismatch"}:
        return "execution_or_scaffold"
    return "ambiguous_or_unscorable"


def result_label(report: dict) -> str:
    if report["status"] != "completed":
        return report["status"]
    return "correct" if report["all_correct"] else "wrong_answer"


def run_generate(root: Path, output: Path, shard_index: int) -> None:
    if shard_index not in range(SHARD_COUNT):
        raise ValueError(f"shard must be 0..{SHARD_COUNT - 1}")
    audit = output / f"a0_shard_{shard_index:02d}.jsonl"
    summary_path = output / f"a0_shard_{shard_index:02d}_summary.json"
    if audit.exists() or summary_path.exists():
        raise RuntimeError("refusing to overwrite shard artifacts")
    lock = load_lock(output)
    bank, _ = load_inputs(root)
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    start = shard_index * SHARD_SIZE
    selected = lock["selected"][start:start + SHARD_SIZE]
    if len(selected) != SHARD_SIZE:
        raise RuntimeError("lock shard size mismatch")
    versions = ensure_packages()
    sandbox_preflight()
    append_jsonl(audit, {
        "kind": "metadata", "shard_index": shard_index,
        "lock_sha256": lock["lock_sha256"], "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "packages": versions,
        "item_count": len(selected), "gpu_training_authorized": False,
    })
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    append_jsonl(audit, {
        "kind": "stage", "name": "model_loading",
        "cuda_device": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
    })
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map={"": 0},
        torch_dtype=torch.float16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    append_jsonl(audit, {"kind": "stage", "name": "model_ready"})
    counts = Counter()
    try:
        for locked in selected:
            task = by_id[locked["item_id"]]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )
            if inputs.shape[1] != locked["input_tokens"]:
                raise RuntimeError(f"tokenizer drift for {task['item_id']}")
            seed = int(hashlib.sha256(task["item_id"].encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            started = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs.to("cuda:0"),
                    attention_mask=torch.ones_like(inputs).to("cuda:0"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = generated[0, inputs.shape[1]:]
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            recovered, recovery_status = extract_code(raw)
            report = sandbox_execute(recovered, task)
            label = result_label(report)
            raw_shape = callable_shape(raw.strip(), task["function_name"])
            raw_status = "parseable" if raw_shape else "not_directly_parseable"
            failure_class = classify(raw_status, label, recovery_status)
            counts[label] += 1
            counts[f"class:{failure_class}"] += 1
            append_jsonl(audit, {
                "kind": "completion",
                "position": locked["position"],
                "item_id": task["item_id"],
                "family_id": task["family_id"],
                "seed": seed,
                "prompt_sha256": task["prompt_sha256"],
                "tests_sha256": task["tests_sha256"],
                "raw_output": raw,
                "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "recovered_output": recovered,
                "recovered_output_sha256": hashlib.sha256(recovered.encode()).hexdigest(),
                "recovery_status": recovery_status,
                "raw_status": raw_status,
                "verifier_label": label,
                "pre_intervention_class": failure_class,
                "verification": report,
                "generated_tokens": int(tokens.numel()),
                "elapsed_seconds": time.time() - started,
                "dataset_oracle_only": True,
                "independent_oracle_validated": False,
            })
            print(f"{locked['position'] + 1}/{TARGET} {task['item_id']}: {label}", flush=True)
    except Exception:
        append_jsonl(audit, {
            "kind": "failure", "traceback": traceback.format_exc()[-5000:],
        })
        raise
    completed = sum(count for key, count in counts.items() if not key.startswith("class:"))
    if completed != SHARD_SIZE:
        raise RuntimeError("incomplete shard")
    summary = {
        "schema_version": "p16_a0_discovery_shard_summary_v1",
        "shard_index": shard_index,
        "lock_sha256": lock["lock_sha256"],
        "completed": completed,
        "counts": dict(sorted(counts.items())),
        "audit_sha256": sha256_file(audit),
        "dataset_oracle_only": True,
        "independent_oracle_validated": False,
        "gpu_training_authorized": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    write_new_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "generate"))
    parser.add_argument("--root", type=Path, default=Path("/content/phase16"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    output = args.output or args.root / "a0_discovery_run_v1"
    if args.stage == "preflight":
        if args.shard is not None:
            parser.error("--shard is invalid for preflight")
        run_preflight(args.root, output)
    else:
        if args.shard is None:
            parser.error("--shard is required for generate")
        run_generate(args.root, output, args.shard)


if __name__ == "__main__":
    main()
