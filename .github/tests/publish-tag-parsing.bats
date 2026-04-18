#!/usr/bin/env bats
# Tests for tag-parsing and version-verification shell logic
# across publish workflows.
#
# Covers:
#   publish-coach.yml  — coach tag prefix detection + version extraction
#   publish-crates.yml — v-prefix stripping + version comparison
#   publish-npm.yml    — v-prefix stripping + version comparison
#   publish-pypi.yml   — v-prefix stripping + version comparison
#
# Run: bats .github/tests/publish-tag-parsing.bats
# Or:  make test-workflow

# ---------------------------------------------------------------------------
# Helpers: inline the bash snippets from the workflow steps
# ---------------------------------------------------------------------------

# check_coach_tag() — from publish-coach.yml "Check tag prefix" step
check_coach_tag() {
  local TAG="$1"
  if [[ "$TAG" == coach-v* ]]; then
    echo "is-coach=true"
    echo "version=${TAG#coach-v}"
  else
    echo "is-coach=false"
  fi
}

# strip_v_prefix() — common pattern across publish-crates, publish-npm,
# publish-pypi: TAG_VERSION="${TAG#v}"
strip_v_prefix() {
  local TAG="$1"
  echo "${TAG#v}"
}

# verify_version() — common pattern: compare actual vs expected, exit 1
# on mismatch
verify_version() {
  local ACTUAL="$1"
  local EXPECTED="$2"
  if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "::error::version ($ACTUAL) != tag ($EXPECTED)"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Coach tag detection
# ---------------------------------------------------------------------------

@test "coach tag: coach-v1.2.3 detected as coach release" {
  run check_coach_tag "coach-v1.2.3"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "is-coach=true" ]]
  [[ "${lines[1]}" == "version=1.2.3" ]]
}

@test "coach tag: coach-v0.1.0-rc.1 extracts prerelease version" {
  run check_coach_tag "coach-v0.1.0-rc.1"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "is-coach=true" ]]
  [[ "${lines[1]}" == "version=0.1.0-rc.1" ]]
}

@test "coach tag: v1.2.3 is NOT a coach release" {
  run check_coach_tag "v1.2.3"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "is-coach=false" ]]
  [ "${#lines[@]}" -eq 1 ]
}

@test "coach tag: empty string is NOT a coach release" {
  run check_coach_tag ""
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "is-coach=false" ]]
}

@test "coach tag: 'coach-v' alone yields empty version" {
  run check_coach_tag "coach-v"
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" == "is-coach=true" ]]
  [[ "${lines[1]}" == "version=" ]]
}

# ---------------------------------------------------------------------------
# v-prefix stripping (publish-crates, publish-npm, publish-pypi)
# ---------------------------------------------------------------------------

@test "strip v-prefix: v1.0.0 -> 1.0.0" {
  run strip_v_prefix "v1.0.0"
  [ "$output" = "1.0.0" ]
}

@test "strip v-prefix: v0.3.1-beta.2 -> 0.3.1-beta.2" {
  run strip_v_prefix "v0.3.1-beta.2"
  [ "$output" = "0.3.1-beta.2" ]
}

@test "strip v-prefix: no prefix unchanged" {
  run strip_v_prefix "1.0.0"
  [ "$output" = "1.0.0" ]
}

@test "strip v-prefix: double v strips only first" {
  run strip_v_prefix "vv1.0.0"
  [ "$output" = "v1.0.0" ]
}

# ---------------------------------------------------------------------------
# Version verification
# ---------------------------------------------------------------------------

@test "verify version: match succeeds" {
  run verify_version "1.2.3" "1.2.3"
  [ "$status" -eq 0 ]
  [ "$output" = "" ]
}

@test "verify version: mismatch fails with error annotation" {
  run verify_version "1.2.3" "1.2.4"
  [ "$status" -eq 1 ]
  [[ "$output" == *"::error::"* ]]
  [[ "$output" == *"1.2.3"* ]]
  [[ "$output" == *"1.2.4"* ]]
}

@test "verify version: empty actual fails" {
  run verify_version "" "1.0.0"
  [ "$status" -eq 1 ]
}
