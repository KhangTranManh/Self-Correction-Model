"""Build P16-R0 A0 CPU contract/oracle review batch 28."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_e import build_group_e_batch
POSITIONS = (302, 304, 307, 317, 318)
FUNCTIONS = ("score", "sum_nested_numbers", "f", "solve", "reindeer")
def build(inventory, bank, previous):
    return build_group_e_batch(inventory, bank, previous, batch_number=28,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=129,
        cumulative_ready=107, cumulative_rejected=22)
if __name__ == "__main__": run_review_cli(build)
