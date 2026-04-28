#!/usr/bin/env bash
# =============================================================================
# perf-runner.sh — Juno scenario runner and report formatter
#
# Subcommands:
#   run     Fire OpenAI-compatible requests against a coordinator, write JSON
#   report  Read per-run JSON files and print a formatted table
#
# Called by perf-test.sh; not intended for direct invocation.
# Dependencies: bash 4+, curl, awk, sed, date (GNU coreutils).
# =============================================================================

set -uo pipefail
export LC_NUMERIC=C

S1_CONTENT="could you please write me a long poem about love and war, please be as robust as you can"

S2_TURN_1="Hello, my name is Viktor, please remember this!"
S2_TURN_2="Could you please recall me what is my name?"
S2_TURN_3="Thank you so mutch, have a nice day!"
S2_EXPECT_1="Viktor"
S2_EXPECT_2="Viktor"

# =============================================================================
# JSON HELPERS — no jq dependency
# =============================================================================

# Escape a string for safe embedding inside a JSON string literal.
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

# Extract a scalar string value from JSON: "key":"value"
# Handles escaped quotes inside the value (one level only).
json_str() {
    local key="$1" json="$2"
    printf '%s' "$json" | awk -v k="\"${key}\"" '
    {
        p = index($0, k ":")
        if (!p) next
        rest = substr($0, p + length(k) + 1)
        gsub(/^[ \t]*/, "", rest)
        if (substr(rest,1,1) != "\"") next
        rest = substr(rest, 2)
        val = ""
        for (i = 1; i <= length(rest); i++) {
            c = substr(rest, i, 1)
            if (c == "\\") { i++; val = val substr(rest, i, 1); continue }
            if (c == "\"") break
            val = val c
        }
        print val; exit
    }'
}

# Extract a scalar non-string value (number, bool, null) from JSON: "key":value
json_val() {
    local key="$1" json="$2"
    printf '%s' "$json" | sed -n \
        "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\([^,}\] ][^,}\] ]*\).*/\1/p" | head -1
}

# Extract a JSON array as newline-delimited plain items (strings or scalars).
json_arr_items() {
    local key="$1" json="$2"
    printf '%s' "$json" | awk -v k="\"${key}\"" '
    {
        p = index($0, k ":")
        if (!p) next
        rest = substr($0, p + length(k) + 1)
        gsub(/^[ \t]*/, "", rest)
        if (substr(rest,1,1) != "[") next
        rest = substr(rest, 2)
        # Walk chars, emit items at depth 1
        item = ""; depth = 0; in_str = 0
        for (i = 1; i <= length(rest); i++) {
            c = substr(rest, i, 1)
            if (in_str) {
                if (c == "\\") { i++; continue }
                if (c == "\"") in_str = 0
                else item = item c
                continue
            }
            if (c == "\"") { in_str = 1; continue }
            if (c == "{" || c == "[") { depth++; item = item c; continue }
            if (c == "}" || c == "]") {
                if (depth == 0) { gsub(/^[ \t]+|[ \t]+$/, "", item); if (item != "") print item; break }
                depth--; item = item c; continue
            }
            if (c == "," && depth == 0) {
                gsub(/^[ \t]+|[ \t]+$/, "", item)
                if (item != "") print item
                item = ""; continue
            }
            item = item c
        }
    }'
}

# Count entries in a JSON array: "key": [...]
json_arr_len() {
    local key="$1" json="$2"
    json_arr_items "$key" "$json" | wc -l | tr -d ' '
}

# Extract one object from a top-level "configs" array by 0-based index.
json_config_at() {
    local idx="$1" json="$2"
    printf '%s' "$json" | awk -v want="$idx" '
    BEGIN { depth=0; obj=""; cnt=-1; in_obj=0 }
    {
        line = $0
        for (i = 1; i <= length(line); i++) {
            c = substr(line, i, 1)
            if (c == "{") {
                depth++
                if (depth == 2) { cnt++; if (cnt == want) { in_obj=1; obj="" } }
            }
            if (in_obj) obj = obj c
            if (c == "}") {
                if (depth == 2 && in_obj) { print obj; exit }
                depth--
            }
        }
    }'
}

# =============================================================================
# HTTP — blocking and SSE streaming
# =============================================================================

# Blocking POST.  Writes response body to $out_file; prints elapsed_ms on stdout.
http_post() {
    local url="$1" payload="$2" out_file="$3"
    local t0 t1
    t0=$(date +%s%3N)
    if curl -sf --connect-timeout 30 --max-time 300 \
            -X POST -H "Content-Type: application/json" \
            -d "$payload" -o "$out_file" "$url" 2>/dev/null; then
        t1=$(date +%s%3N)
        echo $(( t1 - t0 ))
    else
        t1=$(date +%s%3N)
        printf '{"error":{"message":"curl failed","code":"http_error"}}' > "$out_file"
        echo $(( t1 - t0 ))
    fi
}

# SSE POST.  Prints "ttft_ms total_ms token_count" on stdout.
http_post_stream() {
    local url="$1" payload="$2"
    local t0 t1 ttft_ms=0 tokens=0 first=1
    t0=$(date +%s%3N)
    while IFS= read -r line; do
        case "$line" in
            "data: [DONE]") break ;;
            data:*)
                if [[ $first -eq 1 ]]; then
                    ttft_ms=$(( $(date +%s%3N) - t0 ))
                    first=0
                fi
                (( tokens++ )) || true
                ;;
        esac
    done < <(curl -sf --connect-timeout 30 --max-time 300 \
                 -X POST -H "Content-Type: application/json" \
                 -N -d "$payload" "$url" 2>/dev/null)
    t1=$(date +%s%3N)
    echo "$ttft_ms $(( t1 - t0 )) $tokens"
}

# =============================================================================
# STATISTICS — p50, p95, mean, min, max  (input: one integer per line)
# =============================================================================

compute_stats_json() {
    awk '
    BEGIN { n = 0 }
    /^[0-9]+$/ { a[n++] = $1 }
    END {
        if (n == 0) {
            print "{\"p50\":0,\"p95\":0,\"mean\":0,\"min\":0,\"max\":0,\"count\":0}"
            exit
        }
        # insertion sort
        for (i = 1; i < n; i++) {
            v = a[i]; j = i - 1
            while (j >= 0 && a[j] > v) { a[j+1] = a[j]; j-- }
            a[j+1] = v
        }
        sum = 0
        for (i = 0; i < n; i++) sum += a[i]
        p50 = a[int(n * 0.50)]
        pi  = int(n * 0.95); if (pi >= n) pi = n - 1
        p95 = a[pi]
        mean = int(sum / n)
        printf "{\"p50\":%d,\"p95\":%d,\"mean\":%d,\"min\":%d,\"max\":%d,\"count\":%d}",
               p50, p95, mean, a[0], a[n-1], n
    }'
}

# =============================================================================
# SCENARIO S1 — single long request
# Each session runs in a background subprocess and writes JSON to $tmp_dir/s1_N.json
# =============================================================================

_run_s1_session() {
    local url="$1" model="$2" max_tokens="$3" session_idx="$4"
    local streaming="$5" session_id="$6" priority="$7"
    local temperature="$8" top_p="$9" top_k="${10}"
    local out_file="${11}"

    local content_esc
    content_esc=$(json_escape "$S1_CONTENT")

    local sid_fragment=""
    [[ -n "$session_id" ]] && sid_fragment=",\"x_juno_session_id\":\"$(json_escape "$session_id")\""

    local payload
    payload=$(printf \
        '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"stream":%s,"x_juno_priority":"%s","temperature":%s,"top_p":%s,"x_juno_top_k":%s%s}' \
        "$model" "$content_esc" "$max_tokens" "$streaming" "$priority" \
        "$temperature" "$top_p" "$top_k" "$sid_fragment")

    if [[ "$streaming" == "true" ]]; then
        read -r ttft_ms total_ms tokens < <(http_post_stream "$url" "$payload")
        printf '{"session":%d,"scenario":"s1","streaming":true,"ttft_ms":%d,"total_ms":%d,"tokens":%d,"error":null}\n' \
            "$session_idx" "${ttft_ms:-0}" "${total_ms:-0}" "${tokens:-0}" > "$out_file"
    else
        local body_file; body_file=$(mktemp)
        local elapsed_ms
        elapsed_ms=$(http_post "$url" "$payload" "$body_file")
        local body; body=$(cat "$body_file"); rm -f "$body_file"

        local err_msg tokens content
        err_msg=$(json_str  "message" "$body")
        tokens=$(printf '%s' "$body" | sed -n 's/.*"completion_tokens"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
        # Extract content: first "content":"..." after "message"
        content=$(printf '%s' "$body" | sed -n 's/.*"message"[^{]*{[^}]*"content"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
        content="${content:0:80}"
        content=$(json_escape "$content")

        local err_json="null"
        [[ -n "$err_msg" ]] && err_json="\"$(json_escape "$err_msg")\""

        printf '{"session":%d,"scenario":"s1","streaming":false,"total_ms":%d,"tokens":%d,"content":"%s","error":%s}\n' \
            "$session_idx" "${elapsed_ms:-0}" "${tokens:-0}" "$content" "$err_json" > "$out_file"
    fi
}

# =============================================================================
# SCENARIO S2 — multi-turn recall  (3 turns; checks Viktor recall on turn 2)
# Each session writes 3 JSON lines to $tmp_dir/s2_N_T.json
# =============================================================================

_run_s2_session() {
    local url="$1" model="$2" max_tokens="$3" session_idx="$4"
    local streaming="$5" use_session_id="$6"
    local temperature="$7" top_p="$8" top_k="$9"
    local out_dir="${10}"

    local sid=""
    [[ "$use_session_id" == "true" ]] && \
        sid="perf-kvcache-s${session_idx}-$(date +%s)"

    local turns=("$S2_TURN_1" "$S2_TURN_2" "$S2_TURN_3")
    local expects=("$S2_EXPECT_1" "$S2_EXPECT_2" "")
    local history_json="[]"
    local turn_idx

    for turn_idx in 0 1 2; do
        local user_text="${turns[$turn_idx]}"
        local expected="${expects[$turn_idx]}"
        local turn_num=$(( turn_idx + 1 ))

        # Append user message to history
        local msg; msg=$(printf '{"role":"user","content":"%s"}' "$(json_escape "$user_text")")
        # Build history as JSON array (simple append)
        if [[ "$history_json" == "[]" ]]; then
            history_json="[$msg"
        else
            history_json="${history_json},$msg"
        fi
        local messages_json="${history_json}]"

        local sid_fragment=""
        [[ -n "$sid" ]] && sid_fragment=",\"x_juno_session_id\":\"$(json_escape "$sid")\""

        local payload
        payload=$(printf \
            '{"model":"%s","messages":%s,"max_tokens":%d,"stream":%s,"temperature":%s,"top_p":%s,"x_juno_top_k":%s%s}' \
            "$model" "$messages_json" "$max_tokens" "$streaming" \
            "$temperature" "$top_p" "$top_k" "$sid_fragment")

        local out_file="${out_dir}/s2_${session_idx}_${turn_num}.json"

        if [[ "$streaming" == "true" ]]; then
            read -r ttft_ms total_ms tokens < <(http_post_stream "$url" "$payload")
            printf '{"session":%d,"scenario":"s2","turn":%d,"streaming":true,"ttft_ms":%d,"total_ms":%d,"tokens":%d,"error":null}\n' \
                "$session_idx" "$turn_num" \
                "${ttft_ms:-0}" "${total_ms:-0}" "${tokens:-0}" > "$out_file"
            # Append empty assistant turn to history
            history_json="${history_json},{\"role\":\"assistant\",\"content\":\"\"}"
        else
            local body_file; body_file=$(mktemp)
            local elapsed_ms
            elapsed_ms=$(http_post "$url" "$payload" "$body_file")
            local body; body=$(cat "$body_file"); rm -f "$body_file"

            local err_msg tokens content recall_ok
            err_msg=$(json_str "message" "$body")
            tokens=$(printf '%s' "$body" | sed -n 's/.*"completion_tokens"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
            content=$(printf '%s' "$body" | sed -n 's/.*"message"[^{]*{[^}]*"content"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)

            recall_ok="true"
            if [[ -n "$expected" ]]; then
                local lower_content lower_expected
                lower_content=$(printf '%s' "$content" | tr '[:upper:]' '[:lower:]')
                lower_expected=$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')
                [[ "$lower_content" != *"$lower_expected"* ]] && recall_ok="false"
            fi

            local snippet="${content:0:80}"
            local err_json="null"
            [[ -n "$err_msg" ]] && err_json="\"$(json_escape "$err_msg")\""

            printf '{"session":%d,"scenario":"s2","turn":%d,"streaming":false,"total_ms":%d,"tokens":%d,"content":"%s","recall_ok":%s,"error":%s}\n' \
                "$session_idx" "$turn_num" "${elapsed_ms:-0}" "${tokens:-0}" \
                "$(json_escape "${snippet}")" "$recall_ok" "$err_json" > "$out_file"

            # Append assistant reply to history
            history_json="${history_json},{\"role\":\"assistant\",\"content\":\"$(json_escape "$content")\"}"
        fi
    done
}

# =============================================================================
# VARIANT EXPANSION
# Variants from request_variants JSON: expand into list of
# (name, sampling_json, priority_mix_json)  written to $var_dir/variant_N.env
# =============================================================================

# Parse request_variants JSON object into per-variant env files.
# Each file contains: VAR_NAME, VAR_PARAMS_JSON, VAR_SESSIONS, VAR_PRIORITY_MIX_JSON
expand_variants() {
    local variants_json="$1" n_sessions="$2" out_dir="$3"
    local idx=0

    if [[ "$variants_json" == "null" || -z "$variants_json" ]]; then
        # Single unnamed variant
        printf 'VAR_NAME=\nVAR_PARAMS_JSON=null\nVAR_SESSIONS=%d\nVAR_PRIORITY_MIX_JSON=null\n' \
            "$n_sessions" > "${out_dir}/variant_0.env"
        echo 1
        return
    fi

    # Extract variant names by scanning top-level keys in the JSON object.
    # Expected format: {"name1":{...},"name2":{...}}
    local names
    names=$(printf '%s' "$variants_json" | awk '
    BEGIN { depth=0; in_key=0; key="" }
    {
        for (i=1; i<=length($0); i++) {
            c = substr($0, i, 1)
            if (c == "{") depth++
            else if (c == "}") depth--
            else if (c == "\"" && depth == 1) {
                if (!in_key) { in_key=1; key="" }
                else { print key; in_key=0 }
            } else if (in_key) {
                # Stop at colon that follows the closing quote
                key = key c
            }
        }
    }')

    for name in $names; do
        # Extract the value object for this variant name
        local var_obj
        var_obj=$(printf '%s' "$variants_json" | awk -v k="\"${name}\"" '
        BEGIN { depth=0; found=0; obj="" }
        {
            for (i=1; i<=length($0); i++) {
                c = substr($0, i, 1)
                if (!found) {
                    chunk = substr($0, i, length(k)+1)
                    if (chunk == k ":") { found=1; i += length(k); continue }
                    if (chunk == k " " || chunk == k "\t") {
                        # skip whitespace after key
                    }
                    continue
                }
                if (found && depth == 0) {
                    # skip whitespace before value
                    if (c == " " || c == "\t") continue
                }
                if (c == "{") { depth++; obj = obj c }
                else if (c == "}") {
                    depth--
                    obj = obj c
                    if (depth == 0) { print obj; exit }
                }
                else if (depth > 0) { obj = obj c }
            }
        }')

        # Check for nested sessions list (Suite 12: mixed_3_high_6_low)
        local sessions_check
        sessions_check=$(printf '%s' "$var_obj" | grep -o '"sessions"[[:space:]]*:' | head -1)
        local var_sessions="$n_sessions"
        local priority_mix_json="null"

        if [[ -n "$sessions_check" ]]; then
            # Has nested sessions list with count + priority
            var_sessions=$(printf '%s' "$var_obj" | awk '
            BEGIN { total=0; in_sessions=0; depth=0 }
            {
                for (i=1; i<=length($0); i++) {
                    c = substr($0, i, 1)
                    if (substr($0,i,10) == "\"sessions\"") in_sessions=1
                    if (in_sessions) {
                        if (c == "[") depth++
                        else if (c == "]") { depth--; if (depth==0) { print total; exit } }
                        else if (c=="\"") {
                            # look for "count":N
                            chunk = substr($0, i)
                            if (match(chunk, /"count"[ \t]*:[ \t]*[0-9]+/)) {
                                cnt = substr(chunk, RSTART, RLENGTH)
                                gsub(/[^0-9]/, "", cnt)
                                total += cnt
                            }
                        }
                    }
                }
            }')

            # Build priority mix JSON: {1:"HIGH",2:"HIGH",3:"LOW",...}
            priority_mix_json=$(printf '%s' "$var_obj" | awk '
            BEGIN { idx=1; out="{"; depth=0; in_sessions=0 }
            {
                for (i=1; i<=length($0); i++) {
                    c = substr($0, i, 1)
                    if (substr($0,i,10) == "\"sessions\"") in_sessions=1
                    if (!in_sessions) continue
                    if (c == "[") depth++
                    else if (c == "]") { depth--; if (depth==0) { print out "}"; exit } }
                    if (depth == 2) {
                        chunk = substr($0, i)
                        if (match(chunk, /"count"[ \t]*:[ \t]*[0-9]+/)) {
                            cnt = substr(chunk, RSTART, RLENGTH)
                            gsub(/[^0-9]/, "", cnt)
                        }
                        if (match(chunk, /"x_juno_priority"[ \t]*:[ \t]*"[A-Z]+"]/)) {
                            prio = substr(chunk, RSTART, RLENGTH)
                            gsub(/.*"/, "", prio); gsub(/".*/, "", prio)
                            for (j=0; j<cnt+0; j++) {
                                if (out != "{") out = out ","
                                out = out "\"" idx "\":\"" prio "\""
                                idx++
                            }
                        }
                    }
                }
            }')
        fi

        local params_json
        params_json=$(printf '%s' "$var_obj" | sed 's/"sessions"[[:space:]]*:[[:space:]]*\[.*\]//')

        printf 'VAR_NAME=%s\nVAR_PARAMS_JSON=%s\nVAR_SESSIONS=%d\nVAR_PRIORITY_MIX_JSON=%s\n' \
            "$name" "$params_json" "${var_sessions:-$n_sessions}" "$priority_mix_json" \
            > "${out_dir}/variant_${idx}.env"
        (( idx++ )) || true
    done

    echo "$idx"
}

# =============================================================================
# RUN SUBCOMMAND
# =============================================================================

cmd_run() {
    local host port model hw max_tokens n_sessions scenarios streaming
    local use_sid variants_json output_dir result_file hw_region config_json

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)        host="$2";         shift 2 ;;
            --port)        port="$2";         shift 2 ;;
            --model)       model="$2";        shift 2 ;;
            --hw)          hw="$2";           shift 2 ;;
            --max-tokens)  max_tokens="$2";   shift 2 ;;
            --sessions)    n_sessions="$2";   shift 2 ;;
            --scenarios)   scenarios="$2";    shift 2 ;;
            --streaming)   streaming="$2";    shift 2 ;;
            --session-id)  use_sid="$2";      shift 2 ;;
            --variants)    variants_json="$2";shift 2 ;;
            --output-dir)  output_dir="$2";   shift 2 ;;
            --result-file) result_file="$2";  shift 2 ;;
            --hw-region)   hw_region="$2";    shift 2 ;;
            --config)      config_json="$2";  shift 2 ;;
            *) echo "[runner] unknown arg: $1" >&2; shift ;;
        esac
    done

    host="${host:-localhost}"
    port="${port:-8080}"
    max_tokens="${max_tokens:-50}"
    n_sessions="${n_sessions:-1}"
    scenarios="${scenarios:-s1}"
    streaming="${streaming:-false}"
    use_sid="${use_sid:-false}"
    variants_json="${variants_json:-null}"
    hw_region="${hw_region:-}"
    config_json="${config_json:-{}}"

    # Normalize booleans
    [[ "$streaming" =~ ^(true|1|yes)$ ]] && streaming="true" || streaming="false"
    [[ "$use_sid"   =~ ^(true|1|yes)$ ]] && use_sid="true"   || use_sid="false"

    local url="http://${host}:${port}/v1/chat/completions"
    local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%S)
    mkdir -p "$output_dir"

    local tmp_root; tmp_root=$(mktemp -d)
    trap 'rm -rf "$tmp_root"' EXIT

    local scenario_results_json="["
    local first_sr=1

    # Iterate over comma-separated scenarios
    IFS=',' read -ra scenario_list <<< "$scenarios"
    for scenario in "${scenario_list[@]}"; do
        scenario="${scenario// /}"

        # Expand variants
        local var_dir="${tmp_root}/variants"
        mkdir -p "$var_dir"
        local n_variants
        n_variants=$(expand_variants "$variants_json" "$n_sessions" "$var_dir")

        for (( vi=0; vi<n_variants; vi++ )); do
            [[ -f "${var_dir}/variant_${vi}.env" ]] || continue
            # shellcheck disable=SC1090
            source "${var_dir}/variant_${vi}.env"

            # Extract sampling params from VAR_PARAMS_JSON (or defaults)
            local temperature top_p top_k
            temperature=$(json_str "temperature" "$VAR_PARAMS_JSON" 2>/dev/null); temperature="${temperature:-0.6}"
            top_p=$(json_str "top_p" "$VAR_PARAMS_JSON" 2>/dev/null); top_p="${top_p:-0.95}"
            top_k=$(json_str "top_k" "$VAR_PARAMS_JSON" 2>/dev/null)
            # top_k may be a number (not string) in JSON
            [[ -z "$top_k" ]] && top_k=$(json_val "top_k" "$VAR_PARAMS_JSON" 2>/dev/null)
            top_k="${top_k:-20}"

            local var_sessions="${VAR_SESSIONS:-$n_sessions}"
            local var_name="${VAR_NAME:-}"

            local label="[${hw}] scenario=${scenario} sessions=${var_sessions} stream=${streaming}"
            [[ -n "$var_name" ]] && label+=" variant=${var_name}"
            echo "$label"

            # Session result directory
            local sess_dir="${tmp_root}/${scenario}_${vi}"
            mkdir -p "$sess_dir"

            # Launch all sessions in parallel
            local pids=()

            case "$scenario" in
                s1)
                    for (( i=1; i<=var_sessions; i++ )); do
                        # Priority from mix or default NORMAL
                        local priority="NORMAL"
                        if [[ "$VAR_PRIORITY_MIX_JSON" != "null" && -n "$VAR_PRIORITY_MIX_JSON" ]]; then
                            local p; p=$(json_str "$i" "$VAR_PRIORITY_MIX_JSON" 2>/dev/null)
                            [[ -n "$p" ]] && priority="$p"
                        fi
                        _run_s1_session \
                            "$url" "$model" "$max_tokens" "$i" \
                            "$streaming" "" "$priority" \
                            "$temperature" "$top_p" "$top_k" \
                            "${sess_dir}/s1_${i}.json" &
                        pids+=($!)
                    done
                    ;;
                s2)
                    for (( i=1; i<=var_sessions; i++ )); do
                        _run_s2_session \
                            "$url" "$model" "$max_tokens" "$i" \
                            "$streaming" "$use_sid" \
                            "$temperature" "$top_p" "$top_k" \
                            "$sess_dir" &
                        pids+=($!)
                    done
                    ;;
                *)
                    echo "[runner] unknown scenario: $scenario — skipped" >&2
                    continue
                    ;;
            esac

            # Wait for all sessions
            for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done

            # Collect results
            local lat_file="${tmp_root}/latencies_${scenario}_${vi}.txt"
            local raw_results_json="["
            local first_r=1
            local total_tokens=0
            local errors=0
            local recall_failures=0
            local snippets_json="["
            local snippet_count=0

            for result_f in "${sess_dir}"/*.json; do
                [[ -f "$result_f" ]] || continue
                local rline; rline=$(cat "$result_f")
                [[ -z "$rline" ]] && continue

                # Append to raw array
                [[ $first_r -eq 1 ]] && first_r=0 || raw_results_json+=","
                raw_results_json+="$rline"

                # Collect latency
                local ms; ms=$(json_val "total_ms" "$rline")
                local err; err=$(json_val "error" "$rline")
                if [[ "$err" == "null" || -z "$err" ]]; then
                    [[ -n "$ms" ]] && echo "$ms" >> "$lat_file"
                    local toks; toks=$(json_val "tokens" "$rline")
                    (( total_tokens += toks )) || true
                else
                    (( errors++ )) || true
                fi

                # Recall check for S2
                if [[ "$scenario" == "s2" && "$streaming" != "true" ]]; then
                    local rok; rok=$(json_val "recall_ok" "$rline")
                    [[ "$rok" == "false" ]] && (( recall_failures++ )) || true
                fi

                # Snippets (first 3 results)
                if [[ $snippet_count -lt 3 ]]; then
                    local toks; toks=$(json_val "tokens" "$rline")
                    local snip; snip=$(json_str "content" "$rline")
                    [[ -z "$snip" ]] && snip=$(json_str "error" "$rline")
                    snip="${snip:0:60}"
                    [[ $snippet_count -gt 0 ]] && snippets_json+=","
                    snippets_json+="\"[${toks}t] $(json_escape "$snip")\""
                    (( snippet_count++ )) || true
                fi
            done

            raw_results_json+="]"
            snippets_json+="]"

            # Compute stats
            local stats_json
            stats_json=$(compute_stats_json < "$lat_file" 2>/dev/null || echo '{"p50":0,"p95":0,"mean":0,"min":0,"max":0,"count":0}')

            # TPS
            local mean_ms; mean_ms=$(json_val "mean" "$stats_json"); mean_ms="${mean_ms:-0}"
            local tps="0.00"
            if [[ "$mean_ms" -gt 0 ]] 2>/dev/null; then
                tps=$(awk "BEGIN{printf \"%.2f\", $total_tokens / ($mean_ms / 1000.0)}")
            fi

            # Inline summary
            local p50 p95 mean_v
            p50=$(json_val "p50" "$stats_json")
            p95=$(json_val "p95" "$stats_json")
            mean_v=$(json_val "mean" "$stats_json")
            printf '  p50=%sms  p95=%sms  mean=%sms  tokens=%d  TPS~%s  errors=%d\n' \
                "${p50:-0}" "${p95:-0}" "${mean_v:-0}" "$total_tokens" "$tps" "$errors"
            [[ $recall_failures -gt 0 ]] && \
                printf '  RECALL FAILURES: %d turn(s)\n' "$recall_failures"

            # Assemble scenario result JSON
            local streaming_bool="false"
            [[ "$streaming" == "true" ]] && streaming_bool="true"
            local var_json="null"
            [[ -n "$var_name" ]] && var_json="\"$(json_escape "$var_name")\""

            local sr
            sr=$(printf \
                '{"scenario":"%s","variant":%s,"sessions":%d,"streaming":%s,"latency_stats":%s,"total_tokens":%d,"tps":%s,"errors":%d,"recall_failures":%d,"snippets":%s,"raw":%s}' \
                "$scenario" "$var_json" "$var_sessions" "$streaming_bool" \
                "$stats_json" "$total_tokens" "$tps" "$errors" "$recall_failures" \
                "$snippets_json" "$raw_results_json")

            [[ $first_sr -eq 1 ]] && first_sr=0 || scenario_results_json+=","
            scenario_results_json+="$sr"
        done
    done

    scenario_results_json+="]"

    # Write result file
    local result
    result=$(printf \
        '{"hw":"%s","region":"%s","config":%s,"coordinator":"%s:%s","timestamp":"%s","scenario_results":%s}' \
        "$(json_escape "$hw")" "$(json_escape "$hw_region")" \
        "$config_json" \
        "$(json_escape "$host")" "$port" \
        "$ts" "$scenario_results_json")

    printf '%s\n' "$result" > "$result_file"
    echo "Result written: $result_file"
}

# =============================================================================
# REPORT SUBCOMMAND
# =============================================================================

_print_report() {
    local suite_id="$1" timestamp="$2"
    shift 2
    local result_files=("$@")

    local W=82
    printf '\n'
    printf '%*s\n' $W '' | tr ' ' '='
    printf '  SUITE : %s\n' "$suite_id"
    printf '  TIME  : %s\n' "$timestamp"
    printf '  RUNS  : %d\n' "${#result_files[@]}"
    printf '%*s\n' $W '' | tr ' ' '='

    for rf in "${result_files[@]}"; do
        [[ -f "$rf" ]] || continue
        local run; run=$(cat "$rf")

        local hw; hw=$(json_str "hw" "$run");   hw="${hw:-?}"
        local region; region=$(json_str "region" "$run"); region="${region:-?}"
        local coord; coord=$(json_str "coordinator" "$run"); coord="${coord:-?}"
        local err; err=$(json_str "error" "$run")

        # config fields
        local cfg_itype cfg_nodes cfg_ptype cfg_dtype cfg_border cfg_lora cfg_coord_mode
        local cfg_block; cfg_block=$(printf '%s' "$run" | sed -n 's/.*"config"[[:space:]]*:{//{/;s/}.*//p' | head -1)
        cfg_itype=$(json_str "instance_type" "$cfg_block");   cfg_itype="${cfg_itype:-?}"
        cfg_nodes=$(json_val "node_count"    "$cfg_block");   cfg_nodes="${cfg_nodes:-?}"
        cfg_ptype=$(json_str "ptype"         "$cfg_block");   cfg_ptype="${cfg_ptype:-?}"
        cfg_dtype=$(json_str "dtype"         "$cfg_block");   cfg_dtype="${cfg_dtype:-?}"
        cfg_border=$(json_str "byte_order"   "$cfg_block");   cfg_border="${cfg_border:-?}"
        cfg_lora=$(json_val  "lora"          "$cfg_block");   cfg_lora="${cfg_lora:-?}"
        cfg_coord_mode=$(json_str "coordinator" "$cfg_block"); cfg_coord_mode="${cfg_coord_mode:-?}"

        printf '\n'
        printf '  Hardware  : %s  (%s)  region=%s  coord=%s\n' \
            "${hw^^}" "$cfg_itype" "$region" "$coord"
        printf '  Config    : nodes=%s  ptype=%s  dtype=%s  byteOrder=%s  lora=%s  coord-mode=%s\n' \
            "$cfg_nodes" "$cfg_ptype" "$cfg_dtype" "$cfg_border" "$cfg_lora" "$cfg_coord_mode"

        if [[ -n "$err" ]]; then
            printf '  [DEPLOYMENT ERROR: %s]\n' "$err"
            printf '%*s\n' $W '' | tr ' ' '-'
            continue
        fi

        printf '\n'
        printf '  %-8s %-24s %3s %3s  %7s %7s %8s  %7s  %6s  %4s\n' \
            "Scenario" "Variant" "N" "Str" "p50ms" "p95ms" "mean ms" "Tokens" "TPS" "Err"
        printf '  %s\n' "$(printf '%*s' $(( W-2 )) '' | tr ' ' '-')"

        # Parse scenario_results array — iterate over objects
        local sr_block sr_idx=0
        # Extract each scenario result object
        while true; do
            sr_block=$(json_config_at "$sr_idx" \
                "$(printf '%s' "$run" | sed -n 's/.*"scenario_results"[[:space:]]*:[[:space:]]*\[/[/p' | head -1 | sed 's/\][^]]*$/]/')")
            [[ -z "$sr_block" ]] && break

            local scen var_name sessions streaming_flag p50 p95 mean_v total_tok tps errs recall_f
            scen=$(json_str "scenario" "$sr_block");         scen="${scen:-?}"
            var_name=$(json_str "variant" "$sr_block");      var_name="${var_name:-}"
            sessions=$(json_val "sessions" "$sr_block");     sessions="${sessions:-0}"
            streaming_flag=$(json_val "streaming" "$sr_block"); streaming_flag="${streaming_flag:-false}"
            local str_flag="N"; [[ "$streaming_flag" == "true" ]] && str_flag="Y"

            local ls_block
            ls_block=$(printf '%s' "$sr_block" | sed -n 's/.*"latency_stats"[[:space:]]*:{/{/;s/}.*//p' | head -1)
            ls_block="${ls_block}}"
            p50=$(json_val "p50" "$ls_block");   p50="${p50:-0}"
            p95=$(json_val "p95" "$ls_block");   p95="${p95:-0}"
            mean_v=$(json_val "mean" "$ls_block"); mean_v="${mean_v:-0}"

            total_tok=$(json_val "total_tokens" "$sr_block"); total_tok="${total_tok:-0}"
            tps=$(json_val "tps" "$sr_block");               tps="${tps:-0.0}"
            errs=$(json_val "errors" "$sr_block");           errs="${errs:-0}"
            recall_f=$(json_val "recall_failures" "$sr_block"); recall_f="${recall_f:-0}"

            local recall_tag=""
            [[ "${recall_f:-0}" -gt 0 ]] 2>/dev/null && recall_tag="  [recall failures: ${recall_f}]"

            printf '  %-8s %-24s %3s %3s  %7s %7s %8s  %7s  %6s  %4s%s\n' \
                "$scen" "${var_name:0:24}" "$sessions" "$str_flag" \
                "$p50" "$p95" "$mean_v" "$total_tok" "$tps" "$errs" "$recall_tag"

            # Print snippets
            local snip_idx=0
            while true; do
                local snip
                snip=$(json_arr_items "snippets" "$sr_block" | sed -n "$((snip_idx+1))p")
                [[ -z "$snip" ]] && break
                printf '           %s\n' "${snip:0:72}"
                (( snip_idx++ )) || true
                [[ $snip_idx -ge 3 ]] && break
            done

            (( sr_idx++ )) || true
        done

        printf '%*s\n' $W '' | tr ' ' '-'
    done

    printf '\n'
}

cmd_report() {
    local suite_id="" timestamp="" result_files_arg="" report_out=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --suite-id)     suite_id="$2";        shift 2 ;;
            --timestamp)    timestamp="$2";       shift 2 ;;
            --result-files) result_files_arg="$2";shift 2 ;;
            --report-out)   report_out="$2";      shift 2 ;;
            *) shift ;;
        esac
    done

    IFS=',' read -ra result_files <<< "$result_files_arg"

    # Optionally write combined JSON report
    if [[ -n "$report_out" ]]; then
        mkdir -p "$(dirname "$report_out")"
        local runs_json="["
        local first=1
        for rf in "${result_files[@]}"; do
            [[ -f "$rf" ]] || continue
            [[ $first -eq 1 ]] && first=0 || runs_json+=","
            runs_json+=$(cat "$rf")
        done
        runs_json+="]"
        printf '{"suite_id":"%s","timestamp":"%s","runs":%s}\n' \
            "$(json_escape "$suite_id")" "$(json_escape "$timestamp")" "$runs_json" \
            > "$report_out"
    fi

    _print_report "$suite_id" "$timestamp" "${result_files[@]}"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    [[ $# -lt 1 ]] && { echo "Usage: $0 <run|report> [options]" >&2; exit 1; }
    local cmd="$1"; shift
    case "$cmd" in
        run)    cmd_run    "$@" ;;
        report) cmd_report "$@" ;;
        *) echo "Unknown subcommand: $cmd" >&2; exit 1 ;;
    esac
}

main "$@"