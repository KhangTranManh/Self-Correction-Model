#!/usr/bin/env bash
# Rerun the resumable Phase 14 pipeline after transient failures (max 5 tries).
# Usage: supervise.sh smoke | pilot | main | comparison
set -uo pipefail
mode="${1:?usage: supervise.sh smoke|pilot|main|comparison}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
mkdir -p logs
for attempt in 1 2 3 4 5; do
  echo "### $mode attempt $attempt $(date -u +%FT%TZ)"
  if bash phase14/scripts/ops/run_remote.sh "$mode"; then
    echo "### PIPELINE_DONE $(date -u +%FT%TZ)"
    touch "logs/PHASE14_${mode^^}_DONE"
    exit 0
  fi
  echo "### attempt $attempt failed; retrying in 60s"
  sleep 60
done
touch "logs/PHASE14_${mode^^}_FAILED"
exit 1
