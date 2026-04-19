#!/usr/bin/env bash
# Official SWE-bench submission workflow
# Prerequisites: fit bench proxy running, ANTHROPIC_API_KEY set
# Usage: ./scripts/submit-swe.sh [--dry-run]

set -euo pipefail

DATASET="${SWEBENCH_DATASET:-princeton-nlp/SWE-bench_Verified}"
OUTPUT="${SWEBENCH_OUTPUT:-./predictions.jsonl}"
PORT="${FIT_BENCH_PORT:-8781}"
UPSTREAM="${FIT_BENCH_UPSTREAM:-anthropic}"
ADVISOR="${FIT_ADVISOR:-./advisor.yaml}"
MAX_WORKERS="${SWEBENCH_MAX_WORKERS:-4}"

cleanup() { kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Start proxy
echo "Starting fit bench proxy..." >&2
fit bench serve --mode plan --advisor "$ADVISOR" \
  --upstream "$UPSTREAM" --port "$PORT" &
PROXY_PID=$!
sleep 2

# Run evaluation
echo "Running SWE-bench Verified ($DATASET)..." >&2
fit bench run-swe \
  --endpoint "http://localhost:$PORT" \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --max-workers "$MAX_WORKERS"

# Submit (unless dry-run)
if [[ "${1:-}" != "--dry-run" ]]; then
  echo "Submitting to SWE-bench leaderboard..." >&2
  sb submit --predictions "$OUTPUT"
else
  echo "Dry run — predictions at $OUTPUT" >&2
fi
