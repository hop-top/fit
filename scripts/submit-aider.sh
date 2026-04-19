#!/usr/bin/env bash
# Aider polyglot benchmark official run via fit proxy.
# Usage: ./scripts/submit-aider.sh [extra harness flags...]
set -euo pipefail

PORT="${FIT_BENCH_PORT:-8781}"
UPSTREAM="${FIT_BENCH_UPSTREAM:-anthropic}"
ADVISOR="${FIT_ADVISOR:-./advisor.yaml}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

cleanup() { kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Start proxy
fit bench serve \
  --mode oneshot \
  --advisor "$ADVISOR" \
  --upstream "$UPSTREAM" \
  --port "$PORT" &
PROXY_PID=$!
sleep 2

# Run Aider benchmark
PYTHONPATH=py/src python -m fit.bench.harness aider \
  --endpoint "http://localhost:$PORT" \
  "$@"

echo "Results match Aider leaderboard format (pass_rate_1, pass_rate_2)." >&2
echo "To reproduce: run this script with same env vars." >&2
