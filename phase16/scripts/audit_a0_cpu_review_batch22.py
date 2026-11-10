"""Build P16-R0 A0 CPU contract/oracle review batch 22."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_d import build_group_d_batch
POSITIONS = (212, 215, 222, 225, 226)
FUNCTIONS = ("count_subsequences", "decrypt", "seven_ate9", "union_jack", "build_a_wall")
def build(inventory, bank, previous):
    return build_group_d_batch(inventory, bank, previous, batch_number=22,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=99,
        cumulative_ready=79, cumulative_rejected=20)
if __name__ == "__main__":
    run_review_cli(build)
