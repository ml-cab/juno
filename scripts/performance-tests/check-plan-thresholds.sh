#!/usr/bin/env bash
# check-plan-thresholds.sh — enforce the gap-closure plan's numeric-threshold rule.
#
# Execution rule 7 in docs/gap-closure-plan/README.md says every tier with a
# performance gate states a concrete pass/fail number, and that the rule is
# machine-checked. This is that check. It reads only the plan tree: no model, no
# GPU, no build, and it runs in well under a second.
#
# Checks, each a failure (exit 1), never a warning:
#   1. Every TIER-*.md that mentions a perf gate (any capitalization) carries at
#      least one **Threshold block holding a numeral and a comparison operator
#      (>=, <=, >, <, or their Unicode forms). A tier that has no gate says so with
#      an explicit **No perf gate** declaration instead, which exempts it.
#   2. No exit criterion (a "- [ ]" or "- [x]" line) says "no unexplained
#      regression", the phrasing rule 7 replaced with numbers.
#   3. Every row of the README's intermediate-milestone table is well formed, and
#      an active milestone asks for more than its reference reading — a milestone
#      already met before its tier starts measures nothing. A retired row must show
#      a reference that really does meet it.
#   4. Every ticked exit criterion ("- [x]", read with its indented notes) that
#      states a threshold (it contains Threshold, perf gate, >=, <=, or their
#      Unicode forms) cites a docs/perf-compare/<dir> or ../perf-compare/<dir>
#      that exists on disk, or carries an explicit **Evidence (not published):**
#      marker naming the evidence it was scored on. Checks 1 to 3 prove a number
#      was written down; this one proves something was scored against it.
#
# The documentation-hardening tier file (TIER-14-*) is excluded from check 1: it names
# the string to describe this check.
#
# Usage:
#   ./scripts/performance-tests/check-plan-thresholds.sh            # check the plan tree
#   ./scripts/performance-tests/check-plan-thresholds.sh --plan DIR # check another copy
#   ... --perf-compare DIR   # resolve check 4's citations here (default docs/perf-compare)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN_DIR="${ROOT}/docs/gap-closure-plan"
PERF_DIR="${ROOT}/docs/perf-compare"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN_DIR="$2"; shift 2 ;;
    --perf-compare) PERF_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) printf 'check-plan-thresholds: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -d "$PLAN_DIR" ]] || { printf 'check-plan-thresholds: no plan directory: %s\n' "$PLAN_DIR" >&2; exit 2; }

failures=0
fail() {
  printf 'FAIL  %s\n' "$*"
  failures=$((failures + 1))
}

# A **Threshold block runs from its line to the next top-level list item, heading
# or end of file, so a threshold stated as a lead-in line plus indented bullets is
# read as one block.
threshold_blocks_ok() {
  awk '
    /\*\*Threshold/ { inblk = 1; blk = ""; }
    inblk && (/^- / || /^#/) && !/\*\*Threshold/ { inblk = 0; if (ok(blk)) found = 1; }
    inblk { blk = blk " " $0 }
    END { if (inblk && ok(blk)) found = 1; exit(found ? 0 : 1) }
    function ok(b) { return (b ~ /[0-9]/) && (b ~ />=|<=|≥|≤|[ *][<>] ?[0-9]/) }
  ' "$1"
}

# Check 4. Prints one line per ticked criterion that states a threshold:
# "<line number>\t<first 70 chars>\t<space-separated cited perf-compare dirs>",
# or "<line number>\tEVIDENCE" when it carries the not-published marker. A criterion
# runs from its "- [x]" line through the indented or blank lines below it.
ticked_threshold_criteria() {
  awk '
    function flush() {
      if (!open) return
      open = 0
      if (crit !~ /Threshold|[Pp]erf gate|>=|<=|≥|≤/) return
      if (crit ~ /\*\*Evidence \(not published\):\*\*/) { printf "%d\tEVIDENCE\n", start; return }
      dirs = ""; rest = crit
      while (match(rest, /(docs\/|\.\.\/)perf-compare\/[A-Za-z0-9._-]+/)) {
        d = substr(rest, RSTART, RLENGTH); sub(/.*perf-compare\//, "", d)
        dirs = dirs " " d
        rest = substr(rest, RSTART + RLENGTH)
      }
      printf "%d\t%s\t%s\n", start, substr(head, 1, 70), dirs
    }
    /^- \[x\]/ { flush(); open = 1; start = NR; head = $0; crit = $0; next }
    open && (/^[ \t]/ || /^$/) { crit = crit " " $0; next }
    { flush() }
    END { flush() }
  ' "$1"
}

tier_count=0
for f in "$PLAN_DIR"/TIER-*.md; do
  [[ -f "$f" ]] || continue
  tier_count=$((tier_count + 1))
  name="$(basename "$f")"

  if grep -qE '^- \[[ x]\].*[Nn]o unexplained regression' "$f"; then
    fail "${name}: an exit criterion says 'no unexplained regression' instead of stating its number"
  fi

  while IFS=$'\t' read -r line head dirs; do
    [[ -n "$line" ]] || continue
    [[ "$head" == EVIDENCE ]] && continue
    for d in $dirs; do
      [[ -d "${PERF_DIR}/${d}" ]] \
        || fail "${name}:${line}: ticked threshold criterion cites docs/perf-compare/${d}, which does not exist"
    done
    if [[ -z "${dirs// /}" ]]; then
      fail "${name}:${line}: ticked threshold criterion cites no docs/perf-compare/<dir> and has no **Evidence (not published):** marker: ${head}"
    fi
  done < <(ticked_threshold_criteria "$f")

  [[ "$name" == TIER-14-* ]] && continue
  grep -qi 'perf gate' "$f" || continue
  if grep -q '\*\*No perf gate' "$f"; then
    continue
  fi
  if ! grep -q '\*\*Threshold' "$f"; then
    fail "${name}: mentions a perf gate but has no **Threshold block (or an explicit **No perf gate declaration)"
  elif ! threshold_blocks_ok "$f"; then
    fail "${name}: has a **Threshold block, but none carries both a numeral and a comparison operator"
  fi
done

(( tier_count > 0 )) || fail "no TIER-*.md files under ${PLAN_DIR}"

# Milestone table: the rows between the "Intermediate milestones" marker and the
# end of that table. Columns: After | Metric | Scope | Threshold | Reference | Status.
readme="${PLAN_DIR}/README.md"
if [[ ! -f "$readme" ]]; then
  fail "README.md missing under ${PLAN_DIR}"
else
  rows="$(awk '
    /\*\*Intermediate milestones\*\*/ { armed = 1; next }
    armed && /^\|/ { intable = 1; print; next }
    intable && !/^\|/ { exit }
  ' "$readme" | tail -n +3)"
  if [[ -z "$rows" ]]; then
    fail "README.md: no intermediate-milestone table found after the **Intermediate milestones** marker"
  else
    while IFS= read -r row; do
      IFS='|' read -r _ after metric scope threshold reference status _ <<<"$row"
      after="$(xargs <<<"$after")"; status="$(xargs <<<"$status")"
      thr="$(grep -oE '[0-9]+(\.[0-9]+)?' <<<"$threshold" | head -1 || true)"
      ref="$(grep -oE '[0-9]+(\.[0-9]+)?' <<<"$reference" | head -1 || true)"
      label="milestone row (after ${after}): $(xargs <<<"$metric") ($(xargs <<<"$scope"))"
      if [[ -z "$thr" ]] || ! grep -qE '>=|≥' <<<"$threshold"; then
        fail "${label}: threshold '$(xargs <<<"$threshold")' is not of the form '>= number'"
        continue
      fi
      case "$status" in
        active)
          if grep -qi 'unmeasured' <<<"$reference"; then
            continue
          fi
          if [[ -z "$ref" ]]; then
            fail "${label}: reference reading '$(xargs <<<"$reference")' is neither a number nor 'unmeasured'"
          elif awk -v t="$thr" -v r="$ref" 'BEGIN { exit !(r >= t) }'; then
            fail "${label}: already met before its tier starts (reference ${ref} >= threshold ${thr}); retire it or raise it"
          fi
          ;;
        retired*)
          if [[ -z "$ref" ]] || ! awk -v t="$thr" -v r="$ref" 'BEGIN { exit !(r >= t) }'; then
            fail "${label}: marked retired but the reference reading does not meet it"
          fi
          ;;
        *)
          fail "${label}: status '${status}' is neither 'active' nor 'retired'"
          ;;
      esac
    done <<<"$rows"
  fi
fi

if (( failures > 0 )); then
  printf 'check-plan-thresholds: %d failure(s)\n' "$failures"
  exit 1
fi
printf 'check-plan-thresholds: ok (%d tier files, milestone table checked)\n' "$tier_count"
