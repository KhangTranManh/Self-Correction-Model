"""Build P16-R0 A0 CPU contract/oracle review batch 33."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_g import build_group_g_batch
POSITIONS = (357, 358, 360, 361, 364)
FUNCTIONS = ("memesorting", "any_odd", "is_odd_heavy", "is_substitution_cipher", "max_ball")
def build(inventory, bank, previous):
    return build_group_g_batch(inventory, bank, previous, batch_number=33, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=154, cumulative_ready=130, cumulative_rejected=24)
if __name__ == "__main__": run_review_cli(build)
