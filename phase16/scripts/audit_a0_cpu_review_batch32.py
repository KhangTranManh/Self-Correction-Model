"""Build P16-R0 A0 CPU contract/oracle review batch 32."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_f import build_group_f_batch
POSITIONS = (343, 345, 346, 353, 356)
FUNCTIONS = ("bucketize", "fib_rabbits", "isTree", "complete_series", "is_keith_number")
def build(inventory, bank, previous):
    return build_group_f_batch(inventory, bank, previous, batch_number=32, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=149, cumulative_ready=125, cumulative_rejected=24)
if __name__ == "__main__": run_review_cli(build)
