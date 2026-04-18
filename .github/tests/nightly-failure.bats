#!/usr/bin/env bats
# Tests for nightly.yml "Create or update failure issue" shell logic.
#
# The step decides whether to create a new issue or comment on an
# existing one based on `gh issue list` output.
#
# Run: bats .github/tests/nightly-failure.bats
# Or:  make test-workflow

# ---------------------------------------------------------------------------
# Helper: replicate the create-or-update decision logic
# ---------------------------------------------------------------------------

# notify_failure() — from nightly.yml "Create or update failure issue" step
# Args: $1 = EXISTING issue number (or empty)
# Echoes the action taken; does NOT call gh.
notify_failure() {
  local EXISTING="$1"
  local RUN_URL="${RUN_URL:-https://github.com/hop-top/fit/actions/runs/123}"

  local BODY="Nightly build failed on $(date -u +%Y-%m-%d).

Run: ${RUN_URL}

Please investigate and resolve."

  if [ -n "$EXISTING" ]; then
    echo "comment:${EXISTING}"
    echo "body:${BODY}"
  else
    echo "create"
    echo "body:${BODY}"
  fi
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@test "nightly failure: existing issue -> comment" {
  run notify_failure "42"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "comment:42" ]]
}

@test "nightly failure: no existing issue -> create" {
  run notify_failure ""
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "create" ]]
}

@test "nightly failure: body includes run URL" {
  RUN_URL="https://github.com/hop-top/fit/actions/runs/999" \
    run notify_failure ""
  [[ "$output" == *"runs/999"* ]]
}

@test "nightly failure: body includes date" {
  local today
  today=$(date -u +%Y-%m-%d)
  run notify_failure ""
  [[ "$output" == *"$today"* ]]
}
