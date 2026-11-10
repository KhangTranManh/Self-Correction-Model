"""Build P16-R0 A0 CPU contract/oracle review batch 21."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_d import build_group_d_batch
POSITIONS = (202, 204, 207, 209, 211)
FUNCTIONS = ("bad_apples", "step_through_with", "to_pretty", "one_down", "missing_alphabets")
def build(inventory, bank, previous):
    return build_group_d_batch(inventory, bank, previous, batch_number=21,
        positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=94,
        cumulative_ready=74, cumulative_rejected=20)
if __name__ == "__main__":
    run_review_cli(build)
