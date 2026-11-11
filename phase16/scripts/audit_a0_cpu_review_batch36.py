"""Build P16-R0 A0 CPU contract/oracle review batch 36."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_g import build_group_g_batch
POSITIONS = (396, 397, 398, 399)
FUNCTIONS = ("pattern", "operator_insertor", "big_primefac_div", "word_wrap")
def build(inventory, bank, previous):
    return build_group_g_batch(inventory, bank, previous, batch_number=36, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=168, cumulative_ready=142, cumulative_rejected=26)
if __name__ == "__main__": run_review_cli(build)
