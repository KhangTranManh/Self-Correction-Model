"""Build P16-R0 A0 CPU contract/oracle review batch 12."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from audit_a0_cpu_review_group_b import build_group_b_batch
POSITIONS = (87, 90, 99, 101, 104)
FUNCTIONS = ("is_balanced", "short_form", "drop_cap", "palin", "binary_cleaner")
def build(inventory, bank, previous):
    return build_group_b_batch(inventory, bank, previous, batch_number=12,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=49,
        cumulative_ready=40, cumulative_rejected=9)
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--inventory",type=Path,required=True); p.add_argument("--bank",type=Path,required=True); p.add_argument("--previous-review",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.output.exists(): p.error(f"refusing to overwrite {a.output}")
    d=build(*[json.loads(x.read_text(encoding="utf-8")) for x in (a.inventory,a.bank,a.previous_review)]); a.output.write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n"); print(json.dumps(d["counts"],indent=2,sort_keys=True)); print(f"next_position: {d['next_position']}")
if __name__ == "__main__": main()
