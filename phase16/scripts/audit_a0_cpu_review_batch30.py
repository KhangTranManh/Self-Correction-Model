"""Build P16-R0 A0 CPU contract/oracle review batch 30."""
from a0_cpu_review_common import run_review_cli
from audit_a0_cpu_review_group_f import build_group_f_batch
POSITIONS = (327, 329, 332, 335, 336)
FUNCTIONS = ("is_in_middle", "bird_code", "longest_palindrome", "word_count", "word_mesh")
def build(inventory, bank, previous):
    return build_group_f_batch(inventory, bank, previous, batch_number=30, positions=POSITIONS, functions=FUNCTIONS, cumulative_examined=139, cumulative_ready=117, cumulative_rejected=22)
if __name__ == "__main__": run_review_cli(build)
