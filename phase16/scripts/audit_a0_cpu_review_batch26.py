"""Build P16-R0 A0 CPU contract/oracle review batch 26."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_e import build_group_e_batch
POSITIONS = (278, 281, 282, 285, 288)
FUNCTIONS = ("womens_age", "simplify", "to_chinese_numeral", "to_binary", "split_the_bill")
def build(inventory, bank, previous):
    return build_group_e_batch(inventory, bank, previous, batch_number=26,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=119,
        cumulative_ready=97, cumulative_rejected=22)
if __name__ == "__main__": run_review_cli(build)
