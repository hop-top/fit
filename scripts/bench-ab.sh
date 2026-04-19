#!/usr/bin/env bash
# A/B benchmark runner: baseline vs fit-steered
# Usage: ./scripts/bench-ab.sh <suite> [ben-args...]
# Example: ./scripts/bench-ab.sh swe-bench --input model=sonnet
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIT_BENCH_PORT="${FIT_BENCH_PORT:-8781}"

suite="${1:-}"
if [[ -z "$suite" ]]; then
  echo "usage: bench-ab.sh <suite> [ben-args...]" >&2
  echo "  suites: swe-bench | tau-bench | aider" >&2
  exit 2
fi
shift

suite_yaml="$ROOT_DIR/py/src/fit/bench/suite/${suite}.yaml"
if [[ ! -f "$suite_yaml" ]]; then
  echo "error: suite not found: $suite_yaml" >&2
  exit 1
fi

# warn if proxy not reachable
if ! curl -sf "http://localhost:${FIT_BENCH_PORT}/health" >/dev/null 2>&1; then
  echo "warn: fit bench proxy not responding on port $FIT_BENCH_PORT" >&2
fi

run_with_ben() {
  if command -v ben >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

if run_with_ben; then
  echo "==> baseline arm" >&2
  baseline_out=$(ben run "$suite_yaml" --candidate baseline "$@" 2>&1 | tee /dev/stderr) \
    || { echo "ERROR: baseline run failed" >&2; exit 1; }
  baseline_id=$(echo "$baseline_out" | tail -1 | grep -oE '[a-f0-9-]{36}' || true)

  echo "==> fit-steered arm" >&2
  steered_out=$(ben run "$suite_yaml" --candidate fit-steered "$@" 2>&1 | tee /dev/stderr) \
    || { echo "ERROR: fit-steered run failed" >&2; exit 1; }
  steered_id=$(echo "$steered_out" | tail -1 | grep -oE '[a-f0-9-]{36}' || true)

  if [[ -n "$baseline_id" && -n "$steered_id" ]]; then
    echo "==> compare" >&2
    ben compare "$baseline_id" "$steered_id"
  else
    echo "warn: could not extract run IDs; skipping compare" >&2
  fi
else
  echo "warn: ben not on PATH; falling back to fit bench CLI" >&2
  out_dir="${BENCH_AB_OUTDIR:-$ROOT_DIR/results/ab-$(date +%Y%m%d-%H%M%S)}"
  mkdir -p "$out_dir"

  cd "$ROOT_DIR"
  export PYTHONPATH="py/src"

  echo "==> baseline arm" >&2
  python -m fit.bench.cli run-swe \
    --endpoint "${LLM_ENDPOINT:-http://localhost:8090}" \
    --output "$out_dir/baseline.jsonl" "$@" \
    > "$out_dir/baseline.json" 2>&1 || true

  echo "==> fit-steered arm" >&2
  python -m fit.bench.cli run-swe \
    --endpoint "http://localhost:${FIT_BENCH_PORT}" \
    --output "$out_dir/steered.jsonl" "$@" \
    > "$out_dir/steered.json" 2>&1 || true

  echo "==> diff" >&2
  diff --color=auto "$out_dir/baseline.json" "$out_dir/steered.json" || true

  echo "results in $out_dir" >&2
fi
