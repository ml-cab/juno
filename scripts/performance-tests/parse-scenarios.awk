#!/usr/bin/awk -f
# Parse test-scenario.txt -> TSV: hw pt n co dt bo lo col tps
# Coordinator-side juno.TokenProduced.tps (max non-zero across merged JFR models).

BEGIN {
    sq = sprintf("%c", 39)
    default_n = 0
    default_co = ""
    section = ""
    hw = ""
    cmd = ""
    run_meta = ""
    in_fence = 0
    in_json = 0
    depth = 0
    json = ""
}

function reset_defaults() {
    default_n = 0
    default_co = ""
}

function trim(s) {
    sub(/^[ \t\r\n]+/, "", s)
    sub(/[ \t\r\n]+$/, "", s)
    return s
}

function col_from_context(enc,    s) {
    s = tolower(section)
    if (s ~ /conv x9/) return "c9"
    if (s ~ /multysession|multi.session/) return "l9"
    if (s ~ /conv/) return "c1"
    if (enc >= 9) return "l9"
    if (enc >= 3) return "c1"
    return "l1"
}

function parse_cmd(cmdline,    nodes, co, pt, dt, bo, lo, inst, out_hw, tmp) {
    nodes = 3
    co = "embedded"
    pt = "pipeline"
    dt = "FP16"
    bo = "BE"
    lo = "off"
    out_hw = hw

    if (match(cmdline, /--instance-type[ \t]+[^ \t]+/)) {
        inst = substr(cmdline, RSTART, RLENGTH)
        if (inst ~ /g4dn/) out_hw = "gpu"
        else if (inst ~ /m7i-flex|c7i-flex|t3\./) out_hw = "cpu"
    }

    if (match(cmdline, /--node-count[ \t]+[0-9]+/)) {
        tmp = substr(cmdline, RSTART, RLENGTH)
        sub(/^[^0-9]+/, "", tmp)
        nodes = tmp + 0
    } else if (default_n > 0) {
        nodes = default_n
    }

    if (cmdline ~ /--coordinator[ \t]+separate/) co = "separate"
    else if (default_co != "") co = default_co

    if (cmdline ~ /--ptype[ \t]+tensor/) pt = "tensor"

    if (cmdline ~ /--dtype[ \t]+FLOAT32/) dt = "FP32"
    else if (cmdline ~ /--dtype[ \t]+INT8/) dt = "INT8"

    if (cmdline ~ /--byteOrder[ \t]+LE|--byte-order[ \t]+LE|--byteorder[ \t]+LE/) bo = "LE"

    if (cmdline ~ /--lora-play/) lo = "on"

    return out_hw "\t" pt "\t" nodes "\t" co "\t" dt "\t" bo "\t" lo
}

function emit_json(    meta, enc, tps, col, jq_cmd) {
    if (json == "") return

    jq_cmd = "echo " sq json sq " | jq -r " sq "[.models[].metrics[\"juno.Tokenizer.encode.count\"] // 0] | max" sq " 2>/dev/null"
    jq_cmd | getline enc
    close(jq_cmd)
    if (enc == "") enc = 0

    jq_cmd = "echo " sq json sq " | jq -r " sq "[.models[].metrics[\"juno.TokenProduced.tps\"] // 0] | map(select(. > 0)) | if length > 0 then max else empty end" sq " 2>/dev/null"
    jq_cmd | getline tps
    close(jq_cmd)
    if (tps == "" || tps + 0 <= 0) {
        json = ""
        return
    }

    meta = run_meta
    if (meta == "") meta = parse_cmd(cmd)
    col = col_from_context(enc + 0)
    tps = sprintf("%.2f", tps + 0)
    print meta "\t" col "\t" tps
    json = ""
}

{
    cur = $0

    if (cur ~ /^=+$/) next

    if (cur ~ /^[0-9]+\./) {
        section = trim(cur)
        reset_defaults()
        next
    }

    if (cur ~ /^CPU conv x9/) {
        section = "CPU conv x9"
        hw = "cpu"
        reset_defaults()
        next
    }
    if (cur ~ /^GPU conv x9/) {
        section = "GPU conv x9"
        hw = "gpu"
        reset_defaults()
        next
    }

    if (cur ~ /node x5/) default_n = 5
    if (cur ~ /node x7/) default_n = 7
    if (cur ~ /nodes x 3/) default_n = 3
    if (cur ~ /separate coordinator/) default_co = "separate"

    if (cur ~ /CPU setup/) { hw = "cpu"; next }
    if (cur ~ /GPU setup/) { hw = "gpu"; next }

    if (cur ~ /juno-deploy\.sh setup/) {
        cmd = cur
        gsub(/^[ \t]+/, "", cmd)
        run_meta = parse_cmd(cmd)
        next
    }

    if (cur ~ /^```$/) {
        if (in_fence && in_json) {
            emit_json()
            in_json = 0
            depth = 0
        }
        in_fence = !in_fence
        next
    }

    if (!in_fence) next

    if (!in_json && cur ~ /^\{$/) {
        in_json = 1
        depth = 1
        json = cur "\n"
        next
    }

    if (in_json) {
        json = json cur "\n"
        for (i = 1; i <= length(cur); i++) {
            c = substr(cur, i, 1)
            if (c == "{") depth++
            else if (c == "}") depth--
        }
        if (depth <= 0) {
            emit_json()
            in_json = 0
            depth = 0
        }
    }
}

END {
    if (in_json) emit_json()
}
