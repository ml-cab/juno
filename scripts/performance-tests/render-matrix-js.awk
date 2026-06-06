#!/usr/bin/awk -f
# Render matrix.tsv rows as JavaScript array entries for juno_test_matrix.html

BEGIN {
    OFS = ""
}

function js_cell(raw,    st, val, js_st) {
    split(raw, p, ":")
    st = p[1]
    val = p[2]
    if (st == "D") js_st = "D"
    else if (st == "P") js_st = "P"
    else if (st == "A") js_st = "A"
    else js_st = "NA"
    if (st == "D" && val != "") return "[" js_st ",\047" val "\047]"
    return "[" js_st ",\047\047]"
}

{
    if ($0 ~ /^#/ || $0 ~ /^$/) next
    id = $1
    lo = ($8 == "on") ? "on" : "off"
    line = "  {id:" id ", hw:\047" $2 "\047, pt:\047" $3 "\047, n:" $4 ", co:\047" $5 "\047, dt:\047" $6 "\047, bo:\047" $7 "\047, lo:\047" lo "\047, "
    line = line "l1:" js_cell($9) ",  l9:" js_cell($10) ",  c1:" js_cell($11) ",  c9:" js_cell($12) "}"
    rows[++row_count] = line
}

END {
    print "const rows = ["
    for (i = 1; i <= row_count; i++) {
        suf = (i < row_count) ? "," : ""
        print rows[i] suf
    }
    print "];"
}
