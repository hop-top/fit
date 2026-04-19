#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENDPOINT="${FIT_ENDPOINT:-http://localhost:8090}"
DOMAIN="${TAU_DOMAIN:-retail}"
USER_SIM_MODEL="${TAU_USER_SIM_MODEL:-gpt-4o-mini}"
MAX_CONCURRENCY="${TAU_MAX_CONCURRENCY:-4}"

cd "$ROOT_DIR" && PYTHONPATH=py/src exec python -m fit.bench.harness tau \
    --endpoint "$ENDPOINT" \
    --domain "$DOMAIN" \
    --user-sim-model "$USER_SIM_MODEL" \
    --max-concurrency "$MAX_CONCURRENCY" \
    "$@"
