"""Build P16-R0 A0 CPU contract/oracle review batch 34."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_g import build_group_g_batch
POSITIONS = (366, 373, 374, 377, 378)
FUNCTIONS = ("shift_left", "elections_winners", "find", "separate_liquids", "sort_reindeer")
def build(inventory, bank, previous):
    return build_group_g_batch(inventory, bank, previous, batch_number=34, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=159, cumulative_ready=135, cumulative_rejected=24)
if __name__ == "__main__": run_review_cli(build)
