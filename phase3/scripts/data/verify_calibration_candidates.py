"""Fresh-verify stochastic calibration generations and retain one wrong answer per source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
NEUTRAL_REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ids(path: Path) -> set[str]:
    return {str(row.get("source_id", row.get("id"))) for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--manifest", default=str(ROOT / "data/router_calibration_v1/generation/generation_manifest.jsonl"))
    parser.add_argument("--probe", default=str(ROOT / "runs/representation_probe/probe_dataset.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "data/two_stage_selective_repair/frozen_eval.jsonl"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "outputs/phase3_decision_only_v1/final_adapter"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/router_calibration_v1/generation"))
    parser.add_argument("--required-wrong", type=int, default=21)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    candidate_path = Path(args.candidates).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = {row["source_id"]: row for row in read_jsonl(manifest_path)}
    candidates = read_jsonl(candidate_path)
    protected = ids(Path(args.probe)) | ids(Path(args.frozen))
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    accepted = []
    audits = []
    used = set()
    for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"])):
        source_id = str(candidate["source_id"])
        if source_id in used or source_id not in manifest or source_id in protected:
            continue
        task = manifest[source_id]
        source = resolver.resolve(task["problem_ref"], source_id)
        answer = str(candidate["raw_answer"])
        result = verifier.verify(source, answer)
        audits.append({
            "candidate_id": candidate["candidate_id"], "source_id": source_id,
            "passed": bool(result["passed"]), "detail": str(result["detail"]),
        })
        if result["passed"]:
            continue
        messages = [
            task["problem_message"],
            {"role": "assistant", "content": answer},
            {"role": "user", "content": NEUTRAL_REVIEW},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if token_length > args.max_length:
            continue
        accepted.append({
            "schema_version": "phase3_calibration_generated_wrong_v1",
            "candidate_id": candidate["candidate_id"], "source_id": source_id,
            "dataset": "gsm8k", "domain": "math", "label": "REVISE", "class_id": 1,
            "bucket": "generated_natural_failure", "messages": messages,
            "feedback_type": "canonical_neutral_review", "v1_correct": False,
            "verification_method": verifier.method(source), "verifier_detail": str(result["detail"]),
            "source_ref": task["problem_ref"], "v1_attempt_ref": f"{candidate_path}#candidate_id={candidate['candidate_id']}",
            "answer_length_chars": len(answer), "answer_length_words": len(answer.split()),
            "input_token_length": token_length, "decision_only_sft_source_overlap": False,
            "generation": {key: candidate.get(key) for key in ("model_origin", "served_model", "seed", "temperature", "top_p")},
        })
        used.add(source_id)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "verified_wrong.jsonl"
    with accepted_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output / "verification_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "phase3_calibration_candidate_verification_v1",
        "raw_candidates": len(candidates), "verified_candidates": len(audits),
        "unique_natural_wrong_sources": len(accepted), "required_wrong_sources": args.required_wrong,
        "ready": len(accepted) >= args.required_wrong,
        "probe_frozen_overlap": len({row["source_id"] for row in accepted} & protected),
        "accepted_sha256": hashlib.sha256(accepted_path.read_bytes()).hexdigest(),
    }
    (output / "verification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
