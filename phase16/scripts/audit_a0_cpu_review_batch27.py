"""Build P16-R0 A0 CPU contract/oracle review batch 27."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_e import build_group_e_batch
POSITIONS = (290, 291, 295, 297, 301)
FUNCTIONS = ("buy_newspaper", "grader", "sum_groups", "sc_perm_comb", "example_sort")
def build(inventory, bank, previous):
    return build_group_e_batch(inventory, bank, previous, batch_number=27,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=124,
        cumulative_ready=102, cumulative_rejected=22)
if __name__ == "__main__": run_review_cli(build)
