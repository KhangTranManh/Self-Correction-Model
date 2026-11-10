"""Build P16-R0 A0 CPU contract/oracle review batch 18."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_c import build_group_c_batch
POSITIONS = (160, 161, 168, 170, 176)
FUNCTIONS = ("triple_double", "black_or_white_key", "evenator", "is_zero_balanced", "graceful_tipping")
def build(inventory, bank, previous):
    return build_group_c_batch(inventory, bank, previous, batch_number=18,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=79,
        cumulative_ready=61, cumulative_rejected=18)
if __name__ == "__main__":
    run_review_cli(build)
