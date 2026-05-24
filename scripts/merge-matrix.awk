#!/usr/bin/awk -f
# Merge perf/matrix.tsv with parsed scenario TSV (hw pt n co dt bo lo col tps).

BEGIN {
    split("l1 l9 c1 c9", cols, " ")
    OFS = "\t"
    while ((getline line < matrix) > 0) {
        if (line ~ /^#/ || line ~ /^$/) {
            header[++hn] = line
            continue
        }
        nf = split(line, f, "\t")
        id = f[1]
        order[++rn] = id
        meta[id] = f[2] "\t" f[3] "\t" f[4] "\t" f[5] "\t" f[6] "\t" f[7] "\t" f[8]
        for (i = 1; i <= 4; i++) cells[id, cols[i]] = f[i + 8]
    }
    close(matrix)
}

function col_idx(c,    i) {
    for (i = 1; i <= 4; i++) if (cols[i] == c) return i
    return 0
}

function apply(key, col, tps,    id, i) {
    for (i = 1; i <= rn; i++) {
        id = order[i]
        split(meta[id], m, "\t")
        row_key = m[1] SUBSEP m[2] SUBSEP m[3] SUBSEP m[4] SUBSEP m[5] SUBSEP m[6] SUBSEP m[7]
        if (row_key != key) continue
        if (cells[id, col] ~ /^NA:/) return
        cells[id, col] = "D:" tps
        return
    }
}

{
    if (NF < 9) next
    key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP $5 SUBSEP $6 SUBSEP $7
    col = $8
    tps = $9
    if (col_idx(col) == 0) next
    apply(key, col, tps)
}

END {
    for (h = 1; h <= hn; h++) print header[h]
    for (i = 1; i <= rn; i++) {
        id = order[i]
        split(meta[id], m, "\t")
        printf "%s", id
        for (j = 1; j <= 7; j++) printf "\t%s", m[j]
        for (j = 1; j <= 4; j++) printf "\t%s", cells[id, cols[j]]
        print ""
    }
}
