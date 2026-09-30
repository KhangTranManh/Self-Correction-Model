#!/usr/bin/env bash
# Rerun the resumable Phase 9 pipeline after transient failures (max 5 tries).
set -uo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
mkdir -p logs
for attempt in 1 2 3 4 5; do
  echo "### attempt $attempt $(date -u +%FT%TZ)"
  if bash phase9/scripts/ops/run_remote.sh; then
    echo "### PIPELINE_DONE $(date -u +%FT%TZ)"
    touch logs/PHASE9_DONE
    exit 0
  fi
  echo "### attempt $attempt failed; retrying in 60s"
  sleep 60
done
touch logs/PHASE9_FAILED
exit 1
