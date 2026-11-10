"""Build P16-R0 A0 CPU contract/oracle review batch 20."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_c import build_group_c_batch
POSITIONS = (187, 192, 193, 194, 198)
FUNCTIONS = ("check_challenge", "toUnderScore", "riders", "mult_triangle", "do_math")
def build(inventory, bank, previous):
    return build_group_c_batch(inventory, bank, previous, batch_number=20,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=89,
        cumulative_ready=69, cumulative_rejected=20)
if __name__ == "__main__":
    run_review_cli(build)
