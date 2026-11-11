"""Build P16-R0 A0 CPU contract/oracle review batch 35."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_g import build_group_g_batch
POSITIONS = (381, 382, 387, 392, 395)
FUNCTIONS = ("spinning_rings", "code", "largest_power", "pay_cheese", "calculate_damage")
def build(inventory, bank, previous):
    return build_group_g_batch(inventory, bank, previous, batch_number=35, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=164, cumulative_ready=138, cumulative_rejected=26)
if __name__ == "__main__": run_review_cli(build)
