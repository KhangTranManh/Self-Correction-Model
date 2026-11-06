"""CPU-only Colab sandbox smoke for 12 excluded TACO reference solutions.

Upload this file and the generated reference_smoke_pack.json under
/content/phase16. Never run the pack's code outside bubblewrap.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path("/content/phase16")
PACK = ROOT / "reference_smoke_pack.json"
AUDIT = ROOT / "d0_ref_audit.jsonl"
SUMMARY = ROOT / "d0_ref_summary.json"
PREFIX = "__P16_D0_RESULT__="


def append(record: dict) -> None:
    with AUDIT.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def ensure_bwrap() -> None:
    if shutil.which("bwrap"):
        return
    subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=180)
    subprocess.run(["apt-get", "install", "-y", "-qq", "bubblewrap"],
                   check=True, timeout=180)


def sandbox(task: dict) -> dict:
    code = task["solution_code"]
    if not code:
        return {"status": "no_code"}
    child = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
record = json.load(sys.stdin)
namespace = {"__name__": "__candidate__"}
with open("/work/candidate.py", encoding="utf-8") as stream:
    exec(compile(stream.read(), "candidate.py", "exec"), namespace)
name = record["function_name"]
if name in namespace and callable(namespace[name]):
    function = namespace[name]
else:
    function = getattr(namespace["Solution"](), name)
answers = [function(*args) for args in record["test_inputs"]]
print("__P16_D0_RESULT__=" + json.dumps(answers, ensure_ascii=False))
'''
    with tempfile.TemporaryDirectory(prefix="p16_d0_ref_") as directory:
        work = Path(directory)
        os.chmod(work, 0o755)
        path = work / "candidate.py"
        path.write_text(code, encoding="utf-8")
        os.chmod(path, 0o644)
        command = [
            "bwrap", "--die-with-parent", "--unshare-user", "--uid", "65534",
            "--gid", "65534", "--unshare-net", "--unshare-pid",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--tmpfs", "/tmp",
            "--ro-bind", str(work), "/work", "--dev", "/dev",
            "--chdir", "/tmp", "/usr/bin/python3.13", "-I", "-S", "-c", child,
        ]
        try:
            result = subprocess.run(
                command,
                input=json.dumps({"function_name": task["function_name"],
                                  "test_inputs": task["test_inputs"]}),
                capture_output=True, text=True, timeout=8,
                env={"PATH": "/usr/bin", "HOME": "/tmp",
                     "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
    if result.returncode:
        return {"status": "runtime_error", "returncode": result.returncode,
                "stderr_tail": result.stderr[-300:]}
    marked = [line[len(PREFIX):] for line in result.stdout.splitlines()
              if line.startswith(PREFIX)]
    if len(marked) != 1:
        return {"status": "missing_result_marker"}
    try:
        actual = json.loads(marked[0])
    except (json.JSONDecodeError, ValueError):
        return {"status": "invalid_result_json"}
    expected = task["test_outputs"]
    if not isinstance(actual, list) or len(actual) != len(expected):
        return {"status": "result_count_mismatch"}
    matches = [a == b for a, b in zip(actual, expected)]
    return {"status": "completed", "test_count": len(matches),
            "match_count": sum(matches), "mismatch_count": len(matches) - sum(matches)}


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    if AUDIT.exists() or SUMMARY.exists():
        raise SystemExit("refuse to overwrite existing audit/summary")
    pack_bytes = PACK.read_bytes()
    pack = json.loads(pack_bytes)
    if len(pack["tasks"]) != 12:
        raise SystemExit("expected exactly 12 excluded reference tasks")
    append({"kind": "meta", "pack_sha256": hashlib.sha256(pack_bytes).hexdigest(),
            "task_count": 12, "purpose": pack["purpose"]})
    ensure_bwrap()
    check = subprocess.run([
        "bwrap", "--die-with-parent", "--unshare-user", "--uid", "65534",
        "--gid", "65534", "--unshare-net", "--unshare-pid",
        "--ro-bind", "/usr", "/usr", "--tmpfs", "/tmp",
        "--dev", "/dev", "/usr/bin/id", "-u",
    ], capture_output=True, text=True, timeout=8)
    if check.returncode or check.stdout.strip() != "65534":
        raise SystemExit("sandbox preflight failed")
    append({"kind": "stage", "name": "sandbox_ready", "uid": 65534})
    for task in pack["tasks"]:
        result = sandbox(task)
        row = {"kind": "task", "id": task["id"],
               "solution_form": task["solution_form"],
               "solution_sha256": task["solution_sha256"], **result}
        append(row)
        print(json.dumps({"id": task["id"], **result}), flush=True)
    rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines()]
    counts = Counter(row.get("status") for row in rows if row["kind"] == "task")
    summary = {"pack_sha256": hashlib.sha256(pack_bytes).hexdigest(),
               "task_count": 12, "status_counts": dict(counts),
               "audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest()}
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
