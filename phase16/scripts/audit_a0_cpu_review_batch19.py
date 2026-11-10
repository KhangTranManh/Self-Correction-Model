"""Build P16-R0 A0 CPU contract/oracle review batch 19."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_c import build_group_c_batch
POSITIONS = (178, 179, 180, 183, 186)
FUNCTIONS = ("penaltyShots", "solve", "owned_cat_and_dog", "hex_color", "digits_average")
def build(inventory, bank, previous):
    return build_group_c_batch(inventory, bank, previous, batch_number=19,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=84,
        cumulative_ready=66, cumulative_rejected=18)
if __name__ == "__main__":
    run_review_cli(build)
