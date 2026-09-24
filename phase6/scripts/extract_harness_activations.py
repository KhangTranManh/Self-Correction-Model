"""Extract pre-hint selected-layer activations for the new Phase 6 holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402

CONFIG = ROOT / "phase6/configs/harness_confirmation_v1.yaml"
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
HOLDOUT = ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl"
HOLDOUT_LOCK = ROOT / "phase6/data/harness_confirmation_holdout_v1_lock.json"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_lock() -> None:
    lock = json.loads(HOLDOUT_LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256_lf(ROOT / relative) != expected["sha256_lf"]:
            raise RuntimeError(f"Frozen input changed: {relative}")


def resolve_adapter(repo: str, revision: str, expected_sha: str, token: str | None) -> str:
    path = Path(snapshot_download(repo_id=repo, revision=revision, token=token))
    if sha256_file(path / "adapter_model.safetensors") != expected_sha:
        raise RuntimeError(f"Adapter hash mismatch: {repo}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, choices=CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-v2", type=Path, default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    verify_lock()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    repos, revisions, hashes = (registry["checkpoint_hub_repos"],
                                registry["checkpoint_hub_revisions"],
                                registry["checkpoint_adapter_sha256"])
    adapter_path: str | None = None
    revision: str | None = None
    if args.checkpoint == "original_solver":
        model_path, revision = repos["original_solver"], revisions["original_solver"]
    elif args.checkpoint == "warmstart_v2":
        model_path, revision = repos["original_solver"], revisions["original_solver"]
        adapter_path = resolve_adapter(repos["phase4_warmstart_v2"], revisions["phase4_warmstart_v2"],
                                       hashes["phase4_warmstart_v2"], token)
    else:
        if not (args.merged_v2 / "phase5_lineage.json").exists():
            raise RuntimeError("Missing merged V2 parent")
        model_path = str(args.merged_v2)
        adapter_path = resolve_adapter(repos["phase4_correction_sft_v3"], revisions["phase4_correction_sft_v3"],
                                       hashes["phase4_correction_sft_v3"], token)
    rows = read_jsonl(HOLDOUT)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = []
    for row in rows:
        problem = Problem(id=row["problem_id"], domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        messages = [
            {"role": "system", "content": "You are a selective-revision policy."},
            {"role": "user", "content": build_prompt(problem)},
            {"role": "assistant", "content": row["initial_output"]},
        ]
        rendered.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=False))
    tokens = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in rendered]
    if max(map(len, tokens)) > int(config["generation"]["max_model_len"]):
        raise RuntimeError("Refusing truncation")
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, token=token,
                                                  torch_dtype=torch.float16, device_map={"": 0},
                                                  low_cpu_mem_usage=True, attn_implementation="sdpa")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, device_map={"": 0})
    model.eval(); model.config.use_cache = False
    selection = json.loads((ROOT / "outputs/phase5_gpu_vllm/probe_v1/selection" /
                            f"{args.checkpoint}_probe_selection.json").read_text(encoding="utf-8"))["selected"]
    layer = int(selection["layer"])
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            sequences = tokens[start:start + args.batch_size]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(sequences):
                ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
                mask[i, :len(seq)] = 1
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                        use_cache=False, return_dict=True)
            for i, pos in enumerate((mask.sum(dim=1) - 1).tolist()):
                values.append(out.hidden_states[layer][i, pos].float().cpu().numpy())
            del out, ids, mask
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.checkpoint}.npz"
    np.savez_compressed(output, activations=np.stack(values).astype(np.float16),
                        problem_ids=np.asarray([row["problem_id"] for row in rows]),
                        labels=np.asarray([0 if row["initial_correct"] else 1 for row in rows], dtype=np.int8),
                        layer=np.asarray([layer], dtype=np.int16))
    summary = {"checkpoint": args.checkpoint, "rows": len(rows), "layer": layer,
               "output_sha256_lf": sha256_lf(output), "holdout_lock_sha256_lf": sha256_lf(HOLDOUT_LOCK)}
    (args.output_dir / f"{args.checkpoint}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

