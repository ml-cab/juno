#!/usr/bin/awk -f
# Select matrix cells to run from scripts/performance-tests/matrix.tsv.
# Output: row_id<TAB>column (l1, l9, c1, c9)
#
# Variables:
#   mode   pending | all   (default: pending)
#   row    optional row id filter
#   col    optional column filter (l1|l9|c1|c9)
#   from   optional inclusive row id range start
#   to     optional inclusive row id range end

BEGIN {
    split("l1 l9 c1 c9", cols, " ")
    if (mode == "") mode = "pending"
}

function row_in_scope(id,    n) {
    n = id + 0
    if (row != "" && id != row) return 0
    if (from != "" && n < from + 0) return 0
    if (to != "" && n > to + 0) return 0
    return 1
}

function col_in_scope(c) {
    return col == "" || c == col
}

function cell_runnable(st) {
    if (st == "NA") return 0
    if (mode == "all") return 1
    return st == "P" || st == "A"
}

$0 ~ /^#/ || $0 ~ /^$/ { next }

{
    id = $1
    if (!row_in_scope(id)) next
    for (i = 1; i <= 4; i++) {
        c = cols[i]
        split($(8 + i), parts, ":")
        st = parts[1]
        if (!col_in_scope(c)) continue
        if (cell_runnable(st)) print id "\t" c
    }
}
