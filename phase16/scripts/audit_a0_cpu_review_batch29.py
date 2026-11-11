"""Build P16-R0 A0 CPU contract/oracle review batch 29."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_f import build_group_f_batch
POSITIONS = (321, 322, 323, 324, 326)
FUNCTIONS = ("cooking_time", "validate_number", "withdraw", "langtons_ant", "get_score")
def build(inventory, bank, previous):
    return build_group_f_batch(inventory, bank, previous, batch_number=29, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=134, cumulative_ready=112, cumulative_rejected=22)
if __name__ == "__main__": run_review_cli(build)
