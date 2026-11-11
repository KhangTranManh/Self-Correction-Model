"""Create a fail-closed A0 follow-up queue from verified sandbox evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from track_a0_cpu_admission import canonical_sha256, sha256_file, verify_inventory


def verified(document, key, schema):
    body = dict(document)
    claimed = body.pop(key, None)
    if claimed != canonical_sha256(body) or body.get("schema_version") != schema:
        raise RuntimeError(f"invalid {schema} hash or schema")


def build(inventory, tracker, summary, triage, amendment, *, source_files):
    verify_inventory(inventory)
    verified(tracker, "tracker_sha256", "p16_a0_cpu_admission_tracker_v1")
    verified(summary, "summary_sha256", "p16_a0_cpu_sandbox_exact_summary_v1")
    verified(triage, "audit_sha256", "p16_a0_oracle_input_shape_triage_v1")
    verified(amendment, "amendment_sha256", "p16_a0_position398_contract_amendment_v2")
    if (summary["source"]["tracker_sha256"] != tracker["tracker_sha256"] or
            triage["source"]["inventory_sha256"] != inventory["inventory_sha256"] or
            amendment["source"]["inventory_sha256"] != inventory["inventory_sha256"] or
            triage["source"]["review_chain_tip_sha256"] != tracker["source"]["review_chain_tip_sha256"] or
            amendment["source"]["review_chain_tip_sha256"] != tracker["source"]["review_chain_tip_sha256"]):
        raise RuntimeError("source identity mismatch")
    if [flag["position"] for flag in triage["flags"]] != [343, 392]:
        raise RuntimeError("shape flags require a new manual review")
    if amendment["position"] != 398 or amendment["verdict"] != "substantive_residual_reproduced":
        raise RuntimeError("position 398 amendment not resolved as expected")
    if summary["positions_by_verdict"]["execution_or_scaffold_unresolved"] != [392]:
        raise RuntimeError("sandbox unresolved positions require a new review")
    by_position = {row["position"]: row for row in inventory["records"]}
    sandbox = {position: verdict for verdict, positions in summary["positions_by_verdict"].items()
               for position in positions}
    ready = tracker["positions"]["oracle_ready"]
    if set(sandbox) != set(ready):
        raise RuntimeError("sandbox coverage mismatch")
    records = []
    for position in ready:
        frozen = by_position[position]
        contract = ("held_currency_string_format_unspecified" if position == 392 else
                    "amended_pending_independent_review" if position == 398 else
                    "pending_full_contract_audit")
        records.append({"position": position, "item_id": frozen["item_id"],
                        "family_id": frozen["family_id"],
                        "prompt_sha256": frozen["prompt_sha256"],
                        "sandbox_verdict": (amendment["verdict"] if position == 398
                                            else sandbox[position]),
                        "sandbox_evidence": ("position_398_amendment_v2" if position == 398
                                             else "sandbox_exact_summary_v1"),
                        "contract_gate": contract,
                        "history_exact_frozen": ("clean" if frozen["source_history_exact_clean"] is True else "not_clean"),
                        "semantic_history_gate": "pending",
                        "cross_history_near_duplicate_gate": "pending",
                        "corrected_implementation_gate": "pending",
                        "human_approval": "pending",
                        "correction_admitted": False})
    document = {"schema_version": "p16_a0_cpu_followup_worklist_v1",
                "source": {"inventory_sha256": inventory["inventory_sha256"],
                           "tracker_sha256": tracker["tracker_sha256"],
                           "sandbox_summary_sha256": summary["summary_sha256"],
                           "shape_triage_sha256": triage["audit_sha256"],
                           "position398_amendment_sha256": amendment["amendment_sha256"],
                           "source_file_sha256": dict(sorted(source_files.items()))},
                "counts": {"records": len(records),
                           "sandbox_substantive_observations": sum(r["sandbox_verdict"] == "substantive_residual_reproduced" for r in records),
                           "contract_held": sum(r["contract_gate"].startswith("held_") for r in records),
                           "full_contract_audit_pending": sum(r["contract_gate"] == "pending_full_contract_audit" for r in records),
                           "amendment_independent_review_pending": sum(r["contract_gate"] == "amended_pending_independent_review" for r in records),
                           "semantic_history_pending": len(records),
                           "manual_cross_history_pending": len(records),
                           "corrected_implementation_pending": len(records),
                           "human_approval_pending": len(records),
                           "correction_admitted": 0},
                "records": records,
                "claims": {"contract_audit_complete": False, "human_approval_complete": False,
                           "correction_pool_certified": False, "gpu_training_authorized": False}}
    document["worklist_sha256"] = canonical_sha256(document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "tracker", "summary", "triage", "amendment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    paths = [args.inventory, args.tracker, args.summary, args.triage, args.amendment]
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    report = build(*[read(p) for p in paths], source_files={p.name: sha256_file(p) for p in paths})
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report["counts"], indent=2))
    print(f"worklist_sha256: {report['worklist_sha256']}")


if __name__ == "__main__":
    main()
