# Phase 7 data

The CPU audit selected 600 fresh candidate sources in `candidates_v1/` from
1,974 eligible verifier-checked problems. Its report contains source hashes,
exclusion counts, the raw-dataset hash, and the manifest hash. The GPU run
then collected 334 natural initial answers and froze the balanced
development/protected split in `split_v1/`. Those raw answers were later lost
from this workstation; Phase 8 regenerated a separate donor pool with the
frozen protocol (see `../../phase8/RUN_STATUS.md`). Never copy Phase 5 protected or
Phase 6 holdout rows here for reuse.
