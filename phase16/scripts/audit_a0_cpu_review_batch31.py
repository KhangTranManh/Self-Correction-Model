"""Build P16-R0 A0 CPU contract/oracle review batch 31."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_f import build_group_f_batch
POSITIONS = (338, 339, 340, 341, 342)
FUNCTIONS = ("triangle", "i_tri", "x", "tacofy", "encode")
def build(inventory, bank, previous):
    return build_group_f_batch(inventory, bank, previous, batch_number=31, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=144, cumulative_ready=120, cumulative_rejected=24)
if __name__ == "__main__": run_review_cli(build)
