#!/usr/bin/awk -f
# Emit matrix cells to run: row_id column (status P or A only).

BEGIN {
    split("l1 l9 c1 c9", cols, " ")
}

$0 ~ /^#/ || $0 ~ /^$/ { next }

{
    for (i = 1; i <= 4; i++) {
        split($(8 + i), parts, ":")
        st = parts[1]
        if (st == "P" || st == "A") {
            print $1 "\t" cols[i]
        }
    }
}
