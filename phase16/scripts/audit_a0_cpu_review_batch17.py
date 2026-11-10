"""Build P16-R0 A0 CPU contract/oracle review batch 17."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_c import build_group_c_batch
POSITIONS = (152, 156, 157, 158, 159)
FUNCTIONS = ("pattern", "band_name_generator", "compare", "how_many_pizzas", "solve")
def build(inventory, bank, previous):
    return build_group_c_batch(inventory, bank, previous, batch_number=17,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=74,
        cumulative_ready=58, cumulative_rejected=16)
if __name__ == "__main__":
    run_review_cli(build)
