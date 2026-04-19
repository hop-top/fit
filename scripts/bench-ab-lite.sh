#!/usr/bin/env bash
# Quick-iteration A/B: SWE-bench Lite, 50 instance subset
# For local dev — not official runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 50-instance subset for fast iteration
LITE_IDS=(
  astropy__astropy-12907 astropy__astropy-14182
  django__django-11039 django__django-11179 django__django-11283
  django__django-11564 django__django-11620 django__django-11815
  django__django-11848 django__django-11964 django__django-12113
  django__django-12184 django__django-12286 django__django-12453
  django__django-12589 django__django-12747 django__django-12856
  django__django-13028 django__django-13158 django__django-13220
  django__django-13265 django__django-13315 django__django-13321
  django__django-13401 django__django-13447 django__django-13551
  django__django-13590 django__django-13658 django__django-13710
  django__django-13768 django__django-13925 django__django-13933
  django__django-14016 django__django-14155 django__django-14238
  django__django-14382 django__django-14411 django__django-14534
  django__django-14559 django__django-14580 django__django-14608
  django__django-14667 django__django-14730 django__django-14752
  django__django-14787 django__django-14855 django__django-14915
  django__django-14997 django__django-15061 django__django-15202
  django__django-15213
)

exec "$SCRIPT_DIR/bench-ab.sh" swe-bench \
  --instance-ids "${LITE_IDS[@]}" "$@"
