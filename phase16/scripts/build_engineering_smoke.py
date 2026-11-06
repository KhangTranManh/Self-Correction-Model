"""Build original, synthetic Phase 16 engineering tasks; never pilot data."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "data" / "engineering_smoke_v1" / "tasks.json"
SPECS = [
    {
        "task_id": "eng_count_integers",
        "variant": "count", "shape": "integers", "function_name": "count_integers",
        "prompt": "Write Python code defining count_integers(values: list[int], lower: int, upper: int) -> int. Return the number of list entries x with lower <= x <= upper. The interval is inclusive at both ends, lower <= upper, and duplicates count separately. Return only the code.",
    },
    {
        "task_id": "eng_count_scores",
        "variant": "count", "shape": "dict_score", "function_name": "count_scores",
        "prompt": "Write Python code defining count_scores(rows: list[dict], low: int, high: int) -> int. Each row has integer key 'score'. Count rows with score in the closed interval low through high, including score == high. Assume low <= high. Return only the code.",
    },
    {
        "task_id": "eng_count_events",
        "variant": "count", "shape": "pairs", "function_name": "count_events",
        "prompt": "Write Python code defining count_events(events: list[tuple[int, str]], start: int, end: int) -> int. Every event is (timestamp, label). Count events whose timestamp lies between start and end, inclusive of both timestamps; start <= end. Return only the code.",
    },
    {
        "task_id": "eng_sum_payments",
        "variant": "sum", "shape": "dict_day_amount", "function_name": "sum_payments",
        "prompt": "Write Python code defining sum_payments(payments: list[dict], first_day: int, last_day: int) -> int. Each payment has integer keys 'day' and 'amount'. Sum amounts when first_day <= day <= last_day; include last_day and assume first_day <= last_day. Return only the code.",
    },
    {
        "task_id": "eng_sum_readings",
        "variant": "sum", "shape": "pairs", "function_name": "sum_readings",
        "prompt": "Write Python code defining sum_readings(readings: list[tuple[int, int]], low: int, high: int) -> int. Each pair is (time, value). Sum values with time in the inclusive interval [low, high]. Assume low <= high. Return only the code.",
    },
    {
        "task_id": "eng_sum_weights",
        "variant": "sum", "shape": "dict_key_weight", "function_name": "sum_weights",
        "prompt": "Write Python code defining sum_weights(items: list[dict], min_key: int, max_key: int) -> int. Every item contains integer fields 'key' and 'weight'. Return the total weight for items with min_key <= key <= max_key; both endpoints are included. Assume min_key <= max_key. Return only the code.",
    },
    {
        "task_id": "eng_select_names",
        "variant": "select", "shape": "dict_price_name", "function_name": "select_names",
        "prompt": "Write Python code defining select_names(products: list[dict], min_price: int, max_price: int) -> list[str]. Each product has integer 'price' and unique string 'name'. Return names in input order for products with min_price <= price <= max_price, including max_price. Assume min_price <= max_price. Return only the code.",
    },
    {
        "task_id": "eng_select_labels",
        "variant": "select", "shape": "pairs", "function_name": "select_labels",
        "prompt": "Write Python code defining select_labels(events: list[tuple[int, str]], start: int, finish: int) -> list[str]. Each event is (timestamp, unique label). Return labels in input order for events in the closed time interval [start, finish], including events at finish. Assume start <= finish. Return only the code.",
    },
    {
        "task_id": "eng_select_ids",
        "variant": "select", "shape": "dict_rank_id", "function_name": "select_ids",
        "prompt": "Write Python code defining select_ids(records: list[dict], smallest: int, largest: int) -> list[str]. Each record has integer 'rank' and unique string 'id'. Preserve input order while selecting IDs whose rank is at least smallest and at most largest. Assume smallest <= largest. Return only the code.",
    },
]


def encode(items: list[tuple[int, object]], shape: str) -> list:
    if shape == "integers":
        return [key for key, _ in items]
    if shape == "pairs":
        return [[key, value] for key, value in items]
    fields = {
        "dict_score": ("score", "tag"),
        "dict_day_amount": ("day", "amount"),
        "dict_key_weight": ("key", "weight"),
        "dict_price_name": ("price", "name"),
        "dict_rank_id": ("rank", "id"),
    }
    key_field, value_field = fields[shape]
    return [{key_field: key, value_field: value} for key, value in items]


def oracle(items: list[tuple[int, object]], lower: int, upper: int, variant: str):
    selected = [value for key, value in items if lower <= key <= upper]
    if variant == "count":
        return len(selected)
    if variant == "sum":
        return sum(selected)
    return selected


def raw_cases(variant: str) -> list[tuple]:
    def item(key, name):
        if variant == "sum":
            return (key, {"mid": 3, "low": 4, "right5": 6,
                          "right7": 5, "other": 9, "rightb": 2}[name])
        return (key, name)

    return [
        ("ordinary", "ordinary", [item(2, "mid")], 1, 5, None, None),
        ("lower", "lower", [item(1, "low")], 1, 5, None, None),
        ("outside", "outside", [item(0, "other"), item(6, "other")], 1, 5, None, None),
        ("empty", "empty", [], 1, 5, None, None),
        ("single_point", "single_point", [item(5, "right5")], 5, 5, None, None),
        ("duplicate", "duplicate", [item(5, "right5"), item(5, "rightb")], 1, 5, None, None),
        ("p1_anchor", "upper_anchor", [item(2, "mid")], 1, 5, "p1", None),
        ("p1_added", "upper_added", [item(2, "mid"), item(5, "right5")], 1, 5, "p1", "right5"),
        ("p2_anchor", "upper_anchor", [item(1, "low"), item(3, "mid")], 1, 7, "p2", None),
        ("p2_added", "upper_added", [item(1, "low"), item(3, "mid"), item(7, "right7")], 1, 7, "p2", "right7"),
    ]


def build() -> dict:
    tasks = []
    for spec in SPECS:
        cases = []
        for case_id, kind, items, lower, upper, pair_id, added in raw_cases(spec["variant"]):
            case = {
                "case_id": case_id, "kind": kind,
                "args": [encode(items, spec["shape"]), lower, upper],
                "expected": oracle(items, lower, upper, spec["variant"]),
            }
            if pair_id is not None:
                case.update(pair_id=pair_id, upper=upper)
            if kind == "upper_added" and spec["variant"] == "sum":
                case["added_value"] = item_value = next(value for key, value in items if key == upper)
                assert item_value != 0
            if kind == "upper_added" and spec["variant"] == "select":
                case["added_item"] = added
            cases.append(case)
        tasks.append({**spec, "cases": cases})
    ids = [task["task_id"] for task in tasks]
    assert len(tasks) == 9 and len(set(ids)) == 9
    assert all(len(task["cases"]) == 10 for task in tasks)
    return {
        "schema_version": "phase16_engineering_smoke_tasks_v1",
        "purpose": "mechanics_only_excluded_from_pilot",
        "source": "original_synthetic_handwritten_specs",
        "tasks": tasks,
    }


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(build(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(OUT)
