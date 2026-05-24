#!/usr/bin/awk -f
# Read perf/scenarios.yaml -> tab-separated key/value lines.

BEGIN {
    ctx = ""
    in_long = 0
    in_conv = 0
    long_done = 0
}

/^model:/ && !/^model_url:/ {
    print "MODEL_ID\t" $2
}

/^model_url:/ {
    print "MODEL_URL\t" $2
}

/^  cpu:/ { ctx = "cpu" }
/^  gpu:/ { ctx = "gpu" }

ctx == "cpu" && /max_tokens:/ { print "CPU_MAX_TOKENS\t" $2 }
ctx == "gpu" && /max_tokens:/ { print "GPU_MAX_TOKENS\t" $2 }

/^  long:/ { in_long = 1; in_conv = 0; next }
/^  long_s9:/ { in_long = 0; in_conv = 0; next }
/^  conv:/ { in_conv = 1; in_long = 0; next }
/^  conv_s9:/ { in_conv = 0; in_long = 0; next }

/^[a-z_]/ && !/^  / { in_long = 0; in_conv = 0 }

in_long && /prompt:/ && !long_done {
    line = $0
    sub(/^[^"]*"/, "", line)
    sub(/"$/, "", line)
    print "LONG_PROMPT\t" line
    long_done = 1
}

in_conv && /user:/ {
    line = $0
    sub(/^[^"]*"/, "", line)
    sub(/"$/, "", line)
    print "CONV_MSG\t" line
}
