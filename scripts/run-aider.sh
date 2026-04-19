#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for the Aider polyglot benchmark harness.
# Usage: scripts/run-aider.sh --endpoint <url> [options]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"
PYTHONPATH=py/src exec python -m fit.bench.harness aider "$@"
