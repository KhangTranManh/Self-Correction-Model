from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_a0_discovery import (  # noqa: E402
    A0_TARGET,
    MODEL_ID,
    MODEL_REVISION,
    TACO_REVISION,
    canonical_sha256,
    near_duplicate_components,
    parse_call_tests,
    parse_reference_solutions,
    public_summary,
    render_prompt,
    validate_bank,
)


def reference_code(name: str, suffix: str) -> str:
    return f"def {name}(value):\n    return value  # {suffix}\n"


def task(index: int) -> dict:
    item_id = f"taco_train_{index:05d}"
    function_name = f"solve_{index}"
    question = f"Return the supplied value for independently worded task number {index}."
    prompt = render_prompt(question, function_name, "")
    inputs = [[number] for number in range(5)]
    outputs = list(range(5))
    references = []
    for position, suffix in enumerate(("a", "b")):
        code = reference_code(function_name, suffix)
        references.append({
            "position": position,
            "form": "top_level",
            "code": code,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        })
    return {
        "item_id": item_id,
        "row_index": index,
        "dataset_id": "BAAI/TACO",
        "source_revision": TACO_REVISION,
        "source_split": "train",
        "source_site": "codewars",
        "source_id_sha256": canonical_sha256({"id": item_id}),
        "question": question,
        "normalized_question_sha256": canonical_sha256(question.casefold()),
        "function_name": function_name,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "test_inputs": inputs,
        "test_outputs": outputs,
        "tests_sha256": canonical_sha256({"inputs": inputs, "outputs": outputs}),
        "reference_solutions": references,
        "history_exact_clean": True,
        "exact_duplicate_representative": True,
        "dataset_oracle_only": True,
        "independent_oracle_validated": False,
        "family_id": f"family-{index}",
        "near_duplicate_component_size": 1,
    }


def valid_bank() -> dict:
    tasks = [task(index) for index in range(A0_TARGET)]
    bank = {
        "schema_version": "p16_a0_candidate_bank_v1",
        "purpose": "inference_only_residual_discovery",
        "dataset_id": "BAAI/TACO",
        "source_revision": TACO_REVISION,
        "source_split": "train",
        "test_split_opened": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selection": {
            "minimum_tests": 5,
            "minimum_parseable_reference_solutions": 2,
            "maximum_saved_reference_solutions": 3,
            "near_duplicate_method": "token_5gram_jaccard_connected_components",
            "near_duplicate_threshold": 0.8,
            "order_seed": "fixture",
            "a0_target": A0_TARGET,
        },
        "source_artifacts": {
            "metadata_sha256": "a" * 64,
            "overlap_sha256": "b" * 64,
            "shard_sha256": [],
        },
        "rejection_counts": {},
        "tasks": tasks,
        "claims": {
            "source_structural_candidate_only": True,
            "independent_oracle_certified": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
    }
    bank["bank_sha256"] = canonical_sha256(bank)
    return bank


class ParsingTests(unittest.TestCase):
    def test_call_tests_require_argument_lists(self):
        self.assertIsNone(parse_call_tests(
            '{"fn_name":"f","inputs":[1,2,3,4,5],"outputs":[1,2,3,4,5]}'
        ))
        parsed = parse_call_tests(
            '{"fn_name":"f","inputs":[[1],[2],[3],[4],[5]],'
            '"outputs":[1,2,3,4,5]}'
        )
        self.assertEqual(parsed[0][0], [1])

    def test_reference_parser_requires_named_callable(self):
        raw = '["def other(x):\\n return x", "def wanted(x):\\n return x", '
        raw += '"class Solution:\\n def wanted(self, x):\\n  return x"]'
        parsed = parse_reference_solutions(raw, "wanted")
        self.assertEqual([row["form"] for row in parsed],
                         ["top_level", "Solution_method"])

    def test_near_duplicate_components_are_connected(self):
        rows = [
            {"item_id": "a", "question": "one two three four five six seven eight"},
            {"item_id": "b", "question": "one two three four five six seven eight"},
            {"item_id": "c", "question": "entirely different prompt words live over here"},
        ]
        components = near_duplicate_components(rows)
        self.assertIn([0, 1], components)
        self.assertIn([2], components)


class BankTests(unittest.TestCase):
    def test_valid_bank_is_not_gpu_authorization(self):
        bank = valid_bank()
        summary = public_summary(bank)
        report = validate_bank(bank, summary)
        self.assertTrue(report["valid"])
        self.assertFalse(report["a0_inference_authorized"])
        self.assertFalse(report["gpu_training_authorized"])

    def test_tampered_prompt_is_rejected(self):
        bank = valid_bank()
        bank["tasks"][0]["prompt"] += " hidden change"
        bank["bank_sha256"] = canonical_sha256({
            key: value for key, value in bank.items() if key != "bank_sha256"
        })
        with self.assertRaisesRegex(ValueError, "prompt hash mismatch"):
            validate_bank(bank)

    def test_false_independent_oracle_claim_is_rejected(self):
        bank = valid_bank()
        bank["tasks"][0]["independent_oracle_validated"] = True
        bank["bank_sha256"] = canonical_sha256({
            key: value for key, value in bank.items() if key != "bank_sha256"
        })
        with self.assertRaisesRegex(ValueError, "must not claim"):
            validate_bank(bank)

    def test_public_summary_must_match_private_bank(self):
        bank = valid_bank()
        summary = public_summary(bank)
        summary["candidate_count"] -= 1
        summary["summary_sha256"] = canonical_sha256({
            key: value for key, value in summary.items() if key != "summary_sha256"
        })
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_bank(bank, summary)


if __name__ == "__main__":
    unittest.main()
