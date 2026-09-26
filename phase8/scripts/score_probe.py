"""Score Phase 8 first answers with a locked Phase 5-style hidden-state probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402


MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
SOURCE = ROOT / "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl"
FIRST = ROOT / "outputs/phase8_first_pass_v1/initial/answers.jsonl"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=CHECKPOINTS)
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "outputs/phase8_probe_v1/selection")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/phase8_probe_scores_v1")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import joblib
    import torch
    import yaml
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    source_report = json.loads(SOURCE.with_name("candidate_report.json").read_text(encoding="utf-8"))
    first_report = json.loads(FIRST.with_name("summary.json").read_text(encoding="utf-8"))
    if (sha256(SOURCE) != source_report["candidate_manifest_sha256"] or
            sha256(FIRST) != first_report["answers_sha256"]):
        raise RuntimeError("Frozen source or first answers changed")
    sources = read_jsonl(SOURCE)
    answers = {row["problem_id"]: row["output"] for row in read_jsonl(FIRST)}
    if len(sources) != len(answers) or len(sources) != 400:
        raise RuntimeError("Source and first-answer counts differ")
    selection_path = args.probe_dir / f"{args.checkpoint}_probe_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_path = args.probe_dir / f"{args.checkpoint}_probe.joblib"
    if sha256(model_path) != selection["probe_model_sha256"]:
        raise RuntimeError("Probe model hash mismatch")
    layer = int(selection["selected"]["layer"])
    if layer != 14:
        raise RuntimeError("Unexpected layer; Phase 8 preregistered layer 14")
    if float(selection["selected"]["c"]) != {
            "original_solver": 0.01, "warmstart_v2": 0.01,
            "correction_sft_v3": 1.0}[args.checkpoint]:
        raise RuntimeError("Unexpected C; refuse a retuned probe")
    classifier = joblib.load(model_path)
    protocol = yaml.safe_load((ROOT / "phase5/configs/review_protocol_v1.yaml").read_text(encoding="utf-8"))
    lineage = json.loads((ROOT / "phase8/data/model_lineage_v1.json").read_text(encoding="utf-8"))
    if args.checkpoint == "correction_sft_v3":
        checkpoint = ROOT / "models/phase7_v2_merged_fp16"
        adapter = ROOT / "outputs/phase4_correction_sft_v3/final_adapter"
        revision = None
    else:
        checkpoint = MODEL
        adapter = (ROOT / "outputs/phase4_exploration_warmstart_v2/final_adapter"
                   if args.checkpoint == "warmstart_v2" else None)
        revision = REVISION
    if args.checkpoint == "correction_sft_v3":
        marker = json.loads((checkpoint / "phase7_lineage.json").read_text(encoding="utf-8"))
        if marker != lineage["v2_merged_lineage"]:
            raise RuntimeError("V2 merged parent lineage mismatch")
    if adapter:
        expected_hash = (lineage["v3_adapter_sha256"] if args.checkpoint == "correction_sft_v3"
                         else lineage["v2_merged_lineage"]["adapter_weight_sha256"])
        if sha256(adapter / "adapter_model.safetensors") != expected_hash:
            raise RuntimeError("Adapter weight hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    rendered = []
    for source in sources:
        pid = source["id"]
        problem = Problem(id=pid, domain="math", question=source["question"])
        messages = [
            {"role": "system", "content": protocol["conversation"]["system_prompt"]},
            {"role": "user", "content": build_prompt(problem)},
            {"role": "assistant", "content": answers[pid]},
        ]
        rendered.append(tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=bool(protocol["probe"]["apply_chat_template_add_generation_prompt"])))
    tokenized = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in rendered]
    if max(map(len, tokenized)) > int(protocol["generation"]["max_sequence_length"]):
        raise RuntimeError("Probe context would truncate")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, revision=revision, torch_dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False, device_map={"": 0})
    model.eval()
    model.config.use_cache = False
    output_dir = args.output_root / args.checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = output_dir / "scores.audit.jsonl"
    settings = {"schema_version": "phase8_probe_scores_v1", "checkpoint": args.checkpoint,
                "source_sha256": sha256(SOURCE), "initial_answers_sha256": sha256(FIRST),
                "probe_sha256": sha256(model_path), "selection_sha256": sha256(selection_path),
                "layer": layer, "threshold": 0.5, "dtype": "float16",
                "gold_labels_opened": False}
    if audit.exists():
        entries = read_jsonl(audit)
        if not entries or entries[0] != {"type": "metadata", "settings": settings}:
            raise RuntimeError("Existing score audit settings differ")
        done = entries[1:]
        for index, entry in enumerate(done):
            if entry["index"] != index or entry["row"]["problem_id"] != sources[index]["id"]:
                raise RuntimeError("Existing score audit is not an ordered prefix")
    else:
        with audit.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"type": "metadata", "settings": settings}, sort_keys=True) + "\n")
        done = []
    with torch.inference_mode():
        for index in range(len(done), len(sources)):
            ids = torch.tensor([tokenized[index]], device="cuda")
            output = model(input_ids=ids, output_hidden_states=True,
                           use_cache=False, return_dict=True)
            vector = output.hidden_states[layer][0, -1].float().cpu().numpy()[None, :]
            probability = float(classifier.predict_proba(vector)[0, 1])
            row = {"problem_id": sources[index]["id"], "index": index,
                   "probability_wrong": probability, "flag": probability >= 0.5,
                   "token_length": len(tokenized[index])}
            with audit.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"type": "completion", "index": index, "row": row}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            done.append({"row": row})
            del output, ids
            if len(done) % 20 == 0:
                print(f"{args.checkpoint}: {len(done)}/400", flush=True)
    scores = output_dir / "scores.jsonl"
    scores.write_text("".join(json.dumps(entry["row"], sort_keys=True) + "\n"
                              for entry in done), encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(done),
               "flagged": sum(entry["row"]["flag"] for entry in done),
               "audit_sha256": sha256(audit), "scores_sha256": sha256(scores),
               "settings": settings}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
