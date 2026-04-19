#!/usr/bin/env bash
# TAU-bench official run via fit proxy.
# Usage: ./scripts/submit-tau.sh [--domain retail|airline]
set -euo pipefail

DOMAIN="${1:-retail}"
PORT="${FIT_BENCH_PORT:-8781}"
UPSTREAM="${FIT_BENCH_UPSTREAM:-anthropic}"
ADVISOR="${FIT_ADVISOR:-./advisor.yaml}"
USER_SIM="${TAU_USER_SIM_MODEL:-gpt-4o-mini}"
CONCURRENCY="${TAU_MAX_CONCURRENCY:-4}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

cleanup() { kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Start proxy
fit bench serve \
  --mode session \
  --advisor "$ADVISOR" \
  --upstream "$UPSTREAM" \
  --port "$PORT" &
PROXY_PID=$!
sleep 2

# Run TAU-bench
PYTHONPATH=py/src python -m fit.bench.harness tau \
  --endpoint "http://localhost:$PORT" \
  --domain "$DOMAIN" \
  --user-sim-model "$USER_SIM" \
  --max-concurrency "$CONCURRENCY"

echo "Results above are pass^1 for $DOMAIN domain." >&2
echo "To reproduce: run this script with same env vars." >&2
