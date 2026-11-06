# P16-S0 AWS GPU cost/run definition (2026-11-05)

**Planning only. No EC2 instance is launched by this document.** The user has
about USD 140 in AWS funds/credits and prefers not to consume all of it. The
initial project spending ceiling is **USD 45 for one domain** and **USD 80
total for both domains**, including estimated ancillary charges. At least
USD 60 remains uncommitted. A later sealed confirmation requires a new
budget decision; it is not included here. These are *self-imposed stop
limits*, not a claim that AWS will automatically enforce them.

## Preconditions: no GPU spending until all pass

1. P16-S0 source manifest: 160 screen tasks, 40 rule-applicable probes, 20
   same-domain near-negatives, 20 unrelated controls, 60 sealed reserves,
   all source/template-family checked as in `SIGNAL_ONLY_CPU_V1.md`.
2. The named-error oracle and full-correct verifier pass CPU fixtures,
   reference-versus-known-wrong checks and a preselected manual audit.
   CodeContests standard-I/O needs its own networkless sandbox; the current
   function-call runner is not sufficient. Mathematics Dataset Q/A alone
   does not establish the `M-negate-difference` intermediate-step label.
3. Freeze exact model revision, prompts, test hashes, source IDs/split,
   decoding settings, per-item seeds, output cap, LoRA recipe, reset test,
   software versions and append-only audit format. The original Phase 16
   40-episode branch is not silently substituted for this 20-episode S0.
4. In the AWS console verify credit **type, expiration and eligible services**;
   region-specific EC2 price, EBS and networking; G-instance quota (often
   initially zero); and usable instance capacity. Confirm that a budget
   guard/notification and a local wall-clock stopper are configured.

If a precondition fails, record `not_ready`; do not rent a GPU to find more
tasks, change the error definition, or troubleshoot a CPU oracle.

## Instance and cost assumption

Planning region: `us-east-1`, Linux On-Demand, one `g6.xlarge` (NVIDIA L4;
AWS lists 22 GiB accelerator memory, 4 vCPU, 16 GiB RAM). Use NF4 QLoRA
as in the Phase 15 T4 pilot; perform a two-hour smoke to validate actual
VRAM/RAM, package versions, output hashes, reset and speed. If L4 does not
fit or is unavailable, stop and price `g5.xlarge` (A10G, also listed with
22 GiB accelerator memory) before any substitution. Do not assume the
larger instance has better economics. An AMI or Marketplace image with
extra software fees is excluded from this estimate.

Illustrative Linux On-Demand rates observed for `us-east-1` on 2026-11-05:
`g6.xlarge` about **USD 0.8048/hour** and `g5.xlarge` about **USD
1.006/hour**. Recheck the AWS console/Price List at launch; region, purchase
option, software and taxes can change the bill. The hourly planning formula
is `EC2 hours * current instance rate + provisioned EBS + public IPv4 +
other billable transfer/snapshots/tax`. For example, 40 hours of g6.xlarge
compute would be about USD 32.19 before ancillary charges. A 100 GB gp3
volume at the commonly published USD 0.08/GB-month is about USD 8/month if
kept the whole month; it continues to cost money when EC2 is stopped. A
public IPv4 address is billed at USD 0.005/hour. Avoid an idle Elastic IP.

## One-domain stage ladder (maximum time, not promised runtime)

| Stage | Max g6.xlarge hours | Compute ceiling | Continue only if |
|---|---:|---:|---|
| G0: environment/runner smoke | 2 | USD 1.61 | Model, sandbox, hashes, 10 generations and 2 temporary updates finish; all resets exact. Measure tokens/s and minutes/update. |
| G1: 160-task A0 screen | 8 | USD 6.44 | All 160 are scored in frozen order; at least 20 distinct named-error failures and 20 initially correct tasks. Otherwise stop that error type. |
| G2: S1 arms and S2 A0/A1/A1-sham | 10 | USD 8.05 | All required items/arms complete with matching IDs/seeds; CPU-rescored output valid. No partial-result gate. |
| G3: S2 A2/A2-sham temporary updates | 16 | USD 12.88 | Only after G2 and a fresh cost/ETA check; every episode starts/ends at the verified adapter state. |
| G4: optional hidden-state readout | 4 | USD 3.22 | Run only if a behavioral hint exists and the source-grouped probe plan is locked. Never use it to rescue a failed behavioral gate. |
| **Maximum** | **40** | **USD 32.19** | Plus ancillary charges, within the USD 45 one-domain project ceiling. |

G2 covers 100 S1 completions (20 wrong x 3, 20 correct x 2) and 240 S2
non-LoRA completions (80 tasks x 3 arms). G3 covers 160 S2 LoRA completions
(80 tasks x 2 arms) and 40 temporary updates (20 episodes x 2 recipes).
Together with G1, the full one-domain pilot is approximately 660 model
completions, plus smoke. Actual runtime must be measured, not inferred from
this count. If any stage cannot finish under its time ceiling, stop, preserve
append-only records and report `incomplete`; do not score a convenient prefix
as the 20-episode result. A later restart needs a dated amendment and a
budget/lock review.

Run the **first feasible domain only**. The second domain is separately
authorized only after reviewing the first domain's cost, verifier quality,
and signal/harm report. Even if both use the full 40-hour ladder, g6.xlarge
compute is approximately USD 64.38, leaving about USD 15.62 under the USD
80 two-domain ceiling for EBS/network/price variance. If this margin is
insufficient at the actual regional price, reduce the planned *number of
domains/stages*, not the frozen sample size inside a stage.

## Cost and data safeguards

- Set AWS Budgets alerts, but do **not** rely on them as a hard real-time
  stop: AWS warns that usage/billing notifications can lag. Before each
  stage, calculate remaining wall time as the lesser of its stage ceiling,
  the USD 45/80 ceiling divided by the actual hourly rate, and the available
  credit after reserved ancillary margin. Stop the process and EC2 instance
  before that clock expires.
- Use a single tagged instance and a single bounded gp3 EBS volume. Save
  audit/checkpoints to EBS, not instance-store NVMe alone: instance-store
  data is lost on stop. Download/hash the audit after every stage. A stopped
  instance stops EC2 compute billing but EBS and any idle Elastic IP keep
  accruing charges.
- Prefer On-Demand for the first complete run. Spot may be considered later
  only if append-only resumption and interruption handling are tested; AWS
  Spot can give only about two minutes' warning.
- Do not use a larger multi-GPU instance, always-on notebook, managed
  endpoint, or paid Marketplace AMI under this pilot budget. Do not open the
  sealed 60-task reserve for tuning or throughput tests.

## Status boundary

This file freezes a **cost-aware execution proposal**, not the scientific
input lock. No AWS account balance, current regional quote, instance quota,
model compatibility or source capacity has been checked in the user's AWS
account. There is no permission here to launch resources or consume credits.

## Pricing and operations sources checked on 2026-11-05

- [AWS accelerated instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html),
  [EC2 G-instance default quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html).
- Public rate snapshots for [g6.xlarge](https://compute.info/vm/aws/g6-xlarge/)
  and [g5.xlarge](https://cloudparity.io/pricing/aws/g5.xlarge), not a quote
  for the user's account; use the AWS console/Price List before launch.
- [AWS EBS pricing](https://aws.amazon.com/ebs/pricing/),
  [public IPv4 pricing](https://aws.amazon.com/vpc/pricing/), and
  [stop/start charges and instance-store loss](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html).
- [AWS Budgets notification delay](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html),
  [credit eligibility and expiration](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/useconsolidatedbilling-credits.html),
  and [Spot interruption notice](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html).
