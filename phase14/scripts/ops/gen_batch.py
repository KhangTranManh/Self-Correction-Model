"""Run phase14/scripts/generate.py with a larger vLLM chunk size (amendment 2026-10-13, GPU time).

generate_ordered sends requests to vLLM in chunks of phase8.scripts.batched.BATCH_SIZE and
waits for each chunk's longest output; a larger chunk keeps the GPU busy. Per-request seeds
(attempts) and greedy decoding (judges) are unchanged; only scheduling differs. The chunk
size actually used is written into the stage settings. A stage whose audit was opened with
another batch size but holds no completed rows is restarted.
Usage: gen_batch.py <batch size> <generate.py arguments...>
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
size = int(sys.argv[1])
import phase8.scripts.batched as batched  # noqa: E402
from phase14.scripts import generate  # noqa: E402

batched.BATCH_SIZE = size
generate.BATCH_SIZE = size
args = sys.argv[2:]
spec = dict(zip(args[::2], args[1::2]))
stage = spec["--stage"] if spec["--stage"] == "samples" else f"judge_{spec['--judge']}"
audit = generate.folder(spec["--model"], spec["--dataset"], spec["--split"]) / stage / "audit.jsonl"
if audit.exists():
    lines = [json.loads(l) for l in audit.read_text(encoding="utf-8").split("\n") if l.strip()]
    header = next((l for l in lines if l.get("type") != "completion"), {})
    done = sum(l.get("type") == "completion" for l in lines)
    if done == 0 and json.dumps(header).find(f'"batch_size": {size}') < 0:
        audit.unlink()
        print(json.dumps({"restarted_empty_audit": str(audit)}), flush=True)
sys.argv = ["generate.py", *args]
generate.main()
