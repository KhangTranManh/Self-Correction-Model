"""Extract frozen final-token hidden states before any review hint is visible."""

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


PROTOCOL_PATH = ROOT / "phase5/configs/review_protocol_v1.yaml"
LOCK_PATH = ROOT / "phase5/data/protocol/review_protocol_v1_lock.json"
REGISTRY_PATH = ROOT / "phase5/configs/experiments.yaml"
SPLITS = {
    "train": ROOT / "phase5/data/splits/v1/train.jsonl",
    "development": ROOT / "phase5/data/splits/v1/development.jsonl",
    "protected_test": ROOT / "phase5/data/splits/v1/protected_test.jsonl",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def resolve_adapter(repo: str, revision: str, expected_sha: str, token: str | None) -> str:
    path = Path(snapshot_download(repo_id=repo, revision=revision, token=token))
    observed = file_sha256(path / "adapter_model.safetensors")
    if observed != expected_sha:
        raise RuntimeError(f"Adapter hash mismatch for {repo}: {observed}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        choices=("original_solver", "warmstart_v2", "correction_sft_v3"))
    parser.add_argument("--split", required=True, choices=tuple(SPLITS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-v2", type=Path,
                        default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--open-protected", action="store_true")
    args = parser.parse_args()
    if args.split == "protected_test" and not args.open_protected:
        raise RuntimeError("Protected test remains sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen activation extraction")

    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    layers = [int(value) for value in protocol["probe"]["candidate_layers"]]
    max_length = int(protocol["generation"]["max_sequence_length"])
    repos = registry["checkpoint_hub_repos"]
    revisions = registry["checkpoint_hub_revisions"]
    hashes = registry["checkpoint_adapter_sha256"]

    adapter_path: str | None = None
    revision: str | None = None
    if args.checkpoint == "original_solver":
        model_path = repos["original_solver"]
        revision = revisions["original_solver"]
    elif args.checkpoint == "warmstart_v2":
        model_path = repos["original_solver"]
        revision = revisions["original_solver"]
        adapter_path = resolve_adapter(
            repos["phase4_warmstart_v2"], revisions["phase4_warmstart_v2"],
            hashes["phase4_warmstart_v2"], token,
        )
    else:
        marker = args.merged_v2 / "phase5_lineage.json"
        if not marker.exists():
            raise RuntimeError(f"Missing V2 lineage marker: {marker}")
        model_path = str(args.merged_v2)
        adapter_path = resolve_adapter(
            repos["phase4_correction_sft_v3"], revisions["phase4_correction_sft_v3"],
            hashes["phase4_correction_sft_v3"], token,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows = read_jsonl(SPLITS[args.split])
    system = protocol["conversation"]["system_prompt"]
    rendered: list[str] = []
    for row in rows:
        problem = Problem(id=row["problem_id"], domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(problem)},
            {"role": "assistant", "content": row["initial_output"]},
        ]
        rendered.append(tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=bool(protocol["probe"]["apply_chat_template_add_generation_prompt"]),
        ))
    tokenized = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in rendered]
    lengths = [len(ids) for ids in tokenized]
    if max(lengths) > max_length:
        raise RuntimeError(f"Refusing truncation: {max(lengths)} > {max_length}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, token=token, torch_dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=False, device_map={"": 0},
        )
    model.eval()
    model.config.use_cache = False
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_ids = tokenized[start:start + args.batch_size]
            width = max(map(len, batch_ids))
            ids = torch.full((len(batch_ids), width), tokenizer.pad_token_id,
                             dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for index, sequence in enumerate(batch_ids):
                ids[index, :len(sequence)] = torch.tensor(sequence, device="cuda")
                mask[index, :len(sequence)] = 1
            output = model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True,
                use_cache=False, return_dict=True,
            )
            if any(layer >= len(output.hidden_states) for layer in layers):
                raise RuntimeError(f"Requested {layers}, only {len(output.hidden_states)} states")
            final_positions = mask.sum(dim=1) - 1
            for batch_index, position in enumerate(final_positions.tolist()):
                vector = torch.stack([
                    output.hidden_states[layer][batch_index, position] for layer in layers
                ]).float().cpu().numpy()
                values.append(vector)
            del output, ids, mask
            if len(values) % 20 == 0 or len(values) == len(rows):
                print(f"{args.checkpoint}/{args.split}: {len(values)}/{len(rows)}", flush=True)

    activations = np.stack(values).astype(np.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.checkpoint}_{args.split}.npz"
    np.savez_compressed(
        output_path,
        activations=activations,
        problem_ids=np.asarray([row["problem_id"] for row in rows]),
        labels=np.asarray([0 if row["initial_correct"] else 1 for row in rows], dtype=np.int8),
        layers=np.asarray(layers, dtype=np.int16),
        token_lengths=np.asarray(lengths, dtype=np.int16),
    )
    summary = {
        "schema_version": "phase5_prehint_activations_v1",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "rows": len(rows),
        "correct": sum(bool(row["initial_correct"]) for row in rows),
        "wrong": sum(not bool(row["initial_correct"]) for row in rows),
        "layers": layers,
        "shape": list(activations.shape),
        "dtype": str(activations.dtype),
        "pooling": protocol["probe"]["position"],
        "hint_visible": False,
        "generation_performed": False,
        "weights_changed": False,
        "token_lengths": {"min": min(lengths), "max": max(lengths),
                          "mean": sum(lengths) / len(lengths)},
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "protocol_lock_sha256": file_sha256(LOCK_PATH),
    }
    (args.output_dir / f"{args.checkpoint}_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    del model, activations, values
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
