"""Build P16-R0 A0 CPU contract/oracle review batch 24."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_d import build_group_d_batch
POSITIONS = (243, 244, 245, 252, 255)
FUNCTIONS = ("last_man_standing", "part", "missing_angle", "get_drink_by_profession", "solve")
def build(inventory, bank, previous):
    return build_group_d_batch(inventory, bank, previous, batch_number=24,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=109,
        cumulative_ready=88, cumulative_rejected=21)
if __name__ == "__main__":
    run_review_cli(build)
