"""Supersede the incorrect position-398 oracle using its frozen contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from audit_a0_cpu_review_batch6 import _evidence, canonical_sha256
from close_a0_cpu_sandbox_v2 import prepare
from colab_a0_discovery import sandbox_execute, sandbox_preflight
from rescore_a0_discovery import load_lock, verified_completions, verify_manifest
from track_a0_cpu_admission import sha256_file


FRACTION_MESSAGE = "The number has a decimal part. No Results"
CASES = [("composite", (12,)), ("square", (49,)), ("negative", (-18,)),
         ("integral_float", (20.0,)), ("fraction", (20.5,)), ("prime", (13,))]


def oracle(value):
    if isinstance(value, float) and not value.is_integer():
        return FRACTION_MESSAGE
    number = abs(int(value))
    factors = []
    remaining = number
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    if len(factors) < 2:
        return []
    return [max(factors), number // min(factors)]


def reference(value):
    if value != int(value):
        return "The number has a decimal part. No Results"
    number = abs(int(value))
    divisors = [candidate for candidate in range(2, number) if number % candidate == 0]
    if not divisors:
        return []
    primes = [candidate for candidate in range(2, number + 1)
              if number % candidate == 0 and all(candidate % divisor
                  for divisor in range(2, math.isqrt(candidate) + 1))]
    return [max(primes), max(divisors)]


def superseded_mutant(value):
    if isinstance(value, float) and not value.is_integer():
        return []
    result = oracle(value)
    return result[::-1] if result else result


def build(inventory, reviews, completions, previous, bank, *, execute=sandbox_execute,
          preflight=sandbox_preflight, source_files=None):
    tracker, prepared = prepare(inventory, reviews, completions, previous)
    row, code, _, _, old_evidence = next(item for item in prepared if item[0]["position"] == 398)
    task = next(task for task in bank["tasks"] if task["item_id"] == row["item_id"])
    if task["prompt_sha256"] != row["prompt_sha256"]:
        raise RuntimeError("frozen prompt mismatch")
    if hashlib.sha256(task["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
        raise RuntimeError("prompt content hash mismatch")
    evidence = _evidence(CASES, oracle, reference, superseded_mutant)
    inputs = [list(copy.deepcopy(args)) for _, args in CASES]
    expected = [oracle(*copy.deepcopy(args)) for args in inputs]
    preflight()
    observation = execute(code, {"function_name": row["function_name"],
                                 "test_inputs": inputs, "test_outputs": expected},
                          exact_expected=True)
    if observation.get("status") != "completed":
        verdict = "execution_or_scaffold_unresolved"
    elif observation.get("all_correct") is True and observation.get("match_count") == len(inputs):
        verdict = "residual_not_reproduced"
    elif (observation.get("all_correct") is False and
          isinstance(observation.get("match_count"), int) and
          0 <= observation["match_count"] < len(inputs)):
        verdict = "substantive_residual_reproduced"
    else:
        raise RuntimeError("inconsistent sandbox observation")
    report = {"schema_version": "p16_a0_position398_contract_amendment_v2",
              "position": 398, "item_id": row["item_id"],
              "source": {"inventory_sha256": inventory["inventory_sha256"],
                         "review_chain_tip_sha256": tracker["source"]["review_chain_tip_sha256"],
                         "superseded_cases_sha256": old_evidence["cases_sha256"],
                         "source_file_sha256": dict(sorted((source_files or {}).items()))},
              "contract_correction": "return largest prime factor before largest proper divisor; return the exact stated message for fractional inputs",
              "independent_oracle": evidence, "sandbox_observation": observation,
              "verdict": verdict, "correction_admitted": False,
              "claims": {"model_generation": False, "gpu_used": False,
                         "correction_pool_certified": False, "gpu_training_authorized": False}}
    report["amendment_sha256"] = canonical_sha256(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manifest = verify_manifest(args.run, args.manifest)
    lock = load_lock(args.run)
    if manifest.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("manifest and lock differ")
    completions, _ = verified_completions(args.run, lock)
    paths = [args.inventory, args.manifest, args.previous_closure, *args.reviews]
    report = build(read(args.inventory), [read(p) for p in args.reviews], completions,
                   read(args.previous_closure), read(args.bank),
                   source_files={p.name: sha256_file(p) for p in paths})
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"verdict": report["verdict"], "observation": report["sandbox_observation"],
                      "amendment_sha256": report["amendment_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
