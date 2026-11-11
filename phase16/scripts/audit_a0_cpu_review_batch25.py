"""Build P16-R0 A0 CPU contract/oracle review batch 25."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_e import build_group_e_batch
POSITIONS = (260, 263, 268, 271, 276)
FUNCTIONS = ("bus_timer", "is_lucky", "box_capacity", "palindrome", "sum_mul")
def build(inventory, bank, previous):
    return build_group_e_batch(inventory, bank, previous, batch_number=25,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=114,
        cumulative_ready=92, cumulative_rejected=22)
if __name__ == "__main__": run_review_cli(build)
