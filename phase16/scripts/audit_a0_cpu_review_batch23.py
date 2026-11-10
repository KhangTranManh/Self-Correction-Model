"""Build P16-R0 A0 CPU contract/oracle review batch 23."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_d import build_group_d_batch
POSITIONS = (230, 235, 237, 240, 241)
FUNCTIONS = ("group_cities", "average_string", "over_the_road", "abundant", "pattern")
def build(inventory, bank, previous):
    return build_group_d_batch(inventory, bank, previous, batch_number=23,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=104,
        cumulative_ready=83, cumulative_rejected=21)
if __name__ == "__main__":
    run_review_cli(build)
