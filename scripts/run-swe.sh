#!/usr/bin/env bash
# ben CLI adapter for SWE-bench
set -euo pipefail
cd "$(dirname "$0")/../py" && PYTHONPATH=src python -m fit.bench.harness swe "$@"
