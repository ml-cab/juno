#!/usr/bin/env bash
# =============================================================================
# yaml-parse.sh — YAML parsing helpers for perf-test.yaml
#
# Provides:
#   yaml_list_suites  FILE
#   yaml_parse_suite  SELECTOR FILE   (outputs JSON consumed by run_suite)
#
# Targets the exact perf-test.yaml structure (2-space indents, no anchors).
# No external dependencies beyond bash 4+ and awk.
# =============================================================================

# =============================================================================
# PUBLIC: yaml_list_suites FILE
# =============================================================================
yaml_list_suites() {
    local file="$1"
    awk '
    BEGIN {
        in_suites=0; n=0
        sid=""; hw=""; desc=""; desc_next=0
        printf "%-4s %-28s %-12s %s\n", "#", "ID", "HW", "Description"
        printf "%s\n", "--------------------------------------------------------------------------------"
    }
    function flush(    hw_s) {
        if (sid == "") return
        n++
        hw_s = hw
        gsub(/\[|\]| /, "", hw_s)
        gsub(/,/, "+", hw_s)
        printf "%-4d %-28s %-12s %s\n", n, sid, hw_s, substr(desc,1,48)
        sid=""; hw=""; desc=""; desc_next=0
    }
    /^suites:/ { in_suites=1; next }
    !in_suites { next }
    /^[a-z]/ && !/^suites:/ { flush(); in_suites=0; next }  # top-level key ends suites
    /^  - id:/ {
        flush()
        sid = $0; sub(/.*- id:[[:space:]]*/, "", sid); gsub(/#.*$/, "", sid)
        gsub(/[[:space:]]+$/, "", sid)
        desc_next=0
        next
    }
    /^    hardware:/ {
        hw = $0; sub(/.*hardware:[[:space:]]*/, "", hw); gsub(/#.*$/, "", hw)
        gsub(/[[:space:]]+$/, "", hw)
        next
    }
    /^    description:/ {
        desc = $0; sub(/.*description:[[:space:]]*/, "", desc)
        gsub(/[>|]$/, "", desc); gsub(/[[:space:]]+$/, "", desc)
        if (desc == "") desc_next=1
        next
    }
    desc_next && /^      / {
        line = $0; gsub(/^[[:space:]]+/, "", line); gsub(/[[:space:]]+$/, "", line)
        desc = line; desc_next=0
        next
    }
    END { flush() }
    ' "$file"
}

# =============================================================================
# PUBLIC: yaml_parse_suite SELECTOR FILE
# Outputs JSON: {"suite_id":...,"model_url":...,"model_name":...,"jfr":...,"hw_defs":{...},"configs":[...]}
# =============================================================================
yaml_parse_suite() {
    local selector="$1" file="$2"

    # Resolve numeric selector to suite id
    local suite_id
    if [[ "$selector" =~ ^[0-9]+$ ]]; then
        suite_id=$(awk -v want="$selector" '
        BEGIN { in_suites=0; cnt=0 }
        /^suites:/ { in_suites=1; next }
        !in_suites { next }
        /^[a-z]/ && !/^suites:/ { in_suites=0; next }
        /^  - id:/ {
            cnt++
            if (cnt == want+0) {
                id=$0; sub(/.*- id:[[:space:]]*/, "", id)
                gsub(/#.*$/, "", id); gsub(/[[:space:]]+$/, "", id)
                print id; exit
            }
        }
        ' "$file")
        if [[ -z "$suite_id" ]]; then
            printf '{"error":"suite %s not found"}\n' "$selector"; return 1
        fi
    else
        suite_id="$selector"
        # Verify it exists
        local found
        found=$(awk -v want="$selector" '
        /^  - id:/ {
            id=$0; sub(/.*- id:[[:space:]]*/, "", id)
            gsub(/#.*$/, "", id); gsub(/[[:space:]]+$/, "", id)
            if (id == want) { print "yes"; exit }
        }' "$file")
        if [[ "$found" != "yes" ]]; then
            printf '{"error":"suite %s not found"}\n' "$selector"; return 1
        fi
    fi

    # --- Parse constants section ---
    local model_url model_name jfr
    local _const_block
    _const_block=$(awk 'BEGIN{p=0} /^constants:/{p=1;next} p && /^[a-z]/{p=0} p{print}' "$file")
    model_url=$(printf '%s\n' "$_const_block" | grep 'model_url:' | head -1 | \
        sed 's/.*model_url:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//' | tr -d '"')
    model_name=$(printf '%s\n' "$_const_block" | grep 'model_name:' | head -1 | \
        sed 's/.*model_name:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//' | tr -d '"')
    jfr=$(printf '%s\n' "$_const_block" | grep 'jfr:' | head -1 | \
        sed 's/.*jfr:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//' | tr -d '"')

    # --- Parse hardware section ---
    local hw_cpu_itype hw_cpu_max_tokens hw_gpu_itype hw_gpu_max_tokens
    local _hw_block
    _hw_block=$(awk 'BEGIN{p=0} /^hardware:/{p=1;next} p && /^[a-z]/{p=0} p{print}' "$file")
    hw_cpu_itype=$(printf '%s\n' "$_hw_block" | awk 'BEGIN{p=0} /^  cpu:/{p=1;next} p && /^  [a-z]/{p=0} p && /instance_type:/{print;exit}' | \
        sed 's/.*instance_type:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    hw_cpu_max_tokens=$(printf '%s\n' "$_hw_block" | awk 'BEGIN{p=0} /^  cpu:/{p=1;next} p && /^  [a-z]/{p=0} p && /max_tokens:/{print;exit}' | \
        sed 's/.*max_tokens:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    hw_gpu_itype=$(printf '%s\n' "$_hw_block" | awk 'BEGIN{p=0} /^  gpu:/{p=1;next} p && /^  [a-z]/{p=0} p && /instance_type:/{print;exit}' | \
        sed 's/.*instance_type:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    hw_gpu_max_tokens=$(printf '%s\n' "$_hw_block" | awk 'BEGIN{p=0} /^  gpu:/{p=1;next} p && /^  [a-z]/{p=0} p && /max_tokens:/{print;exit}' | \
        sed 's/.*max_tokens:[[:space:]]*//' | sed 's/#.*//' | sed 's/[[:space:]]*$//')

    hw_cpu_itype="${hw_cpu_itype:-m7i-flex.large}"
    hw_cpu_max_tokens="${hw_cpu_max_tokens:-50}"
    hw_gpu_itype="${hw_gpu_itype:-g4dn.2xlarge}"
    hw_gpu_max_tokens="${hw_gpu_max_tokens:-500}"

    local hw_defs_json
    hw_defs_json=$(printf \
        '{"cpu":{"instance_type":"%s","max_tokens":%s},"gpu":{"instance_type":"%s","max_tokens":%s}}' \
        "$hw_cpu_itype" "$hw_cpu_max_tokens" "$hw_gpu_itype" "$hw_gpu_max_tokens")

    # --- Extract the raw suite block from the file ---
    # Returns lines from "  - id: X" up to (not including) the next "  - id:" or end of suites
    local suite_block
    suite_block=$(awk -v want="$suite_id" '
    BEGIN { in_suites=0; capturing=0 }
    /^suites:/ { in_suites=1; next }
    !in_suites { next }
    /^[a-z]/ && !/^suites:/ { capturing=0; in_suites=0; next }
    /^  - id:/ {
        id=$0; sub(/.*- id:[[:space:]]*/, "", id)
        gsub(/#.*$/, "", id); gsub(/[[:space:]]+$/, "", id)
        if (id == want) { capturing=1; print; next }
        if (capturing) { exit }
    }
    capturing { print }
    ' "$file")

    # --- Parse fields from the suite block ---
    # Sweep-able keys (if value is [a, b, ...] it becomes a sweep dimension)
    local -a SWEEP_KEYS=(dtype byte_order ptype lora sessions streaming coordinator node_count)

    declare -A fields  # field_name -> value (raw, may be @LIST:a,b,c)

    # Parse 4-space fields from the suite block (strip inline comments)
    while IFS= read -r line; do
        # 4-space key: value
        if [[ "$line" =~ ^"    "([a-z_]+):[[:space:]]*(.*) ]]; then
            local k="${BASH_REMATCH[1]}"
            local v="${BASH_REMATCH[2]}"
            v="${v%%#*}"          # strip comment
            v="${v%"${v##*[^ ]}"}" # rtrim
            fields["$k"]="$v"
        fi
    done <<< "$suite_block"

    # --- Parse request_params sub-fields (6-space indent) ---
    declare -A rp_fields
    local in_rp=0
    while IFS= read -r line; do
        if [[ "$line" =~ ^"    request_params:" ]]; then in_rp=1; continue; fi
        if [[ $in_rp -eq 1 ]]; then
            if [[ "$line" =~ ^"      "([a-z_]+):[[:space:]]*(.*) ]]; then
                local rk="${BASH_REMATCH[1]}"
                local rv="${BASH_REMATCH[2]}"
                rv="${rv%%#*}"; rv="${rv%"${rv##*[^ ]}"}"
                rp_fields["$rk"]="$rv"
            elif [[ "$line" =~ ^"    "[a-z] ]]; then
                in_rp=0
            fi
        fi
    done <<< "$suite_block"

    # --- Extract request_variants block as raw YAML, then convert to JSON ---
    local rv_json="null"
    local in_rv=0
    local rv_yaml=""
    while IFS= read -r line; do
        if [[ "$line" =~ ^"    request_variants:" ]]; then in_rv=1; continue; fi
        if [[ $in_rv -eq 1 ]]; then
            if [[ "$line" =~ ^"    "[a-z] ]] && [[ ! "$line" =~ ^"      " ]]; then
                in_rv=0; continue
            fi
            rv_yaml+="$line"$'\n'
        fi
    done <<< "$suite_block"

    if [[ -n "$rv_yaml" ]]; then
        rv_json=$(_parse_request_variants "$rv_yaml")
    fi

    # --- Identify swept dimensions ---
    local -a swept_keys=()
    declare -A swept_vals  # key -> space-separated list of values

    for k in "${SWEEP_KEYS[@]}"; do
        local v="${fields[$k]:-}"
        if [[ "$v" =~ ^\[.*,.*\]$ ]]; then
            # Multi-value inline list — this is a sweep dimension
            local inner="${v:1:${#v}-2}"  # strip [ ]
            inner=$(echo "$inner" | sed 's/[[:space:]]//g')  # remove all spaces
            swept_keys+=("$k")
            swept_vals["$k"]="${inner//,/ }"
        fi
    done

    # --- Build cartesian product of swept dimensions ---
    local -a combos=()
    if [[ ${#swept_keys[@]} -eq 0 ]]; then
        combos=("")
    else
        # Build combos iteratively
        combos=("")
        for k in "${swept_keys[@]}"; do
            local new_combos=()
            for val in ${swept_vals[$k]}; do
                for existing in "${combos[@]}"; do
                    local entry="$k=$val"
                    if [[ -n "$existing" ]]; then
                        new_combos+=("${existing} ${entry}")
                    else
                        new_combos+=("$entry")
                    fi
                done
            done
            combos=("${new_combos[@]}")
        done
    fi

    # --- Build hardware JSON array ---
    local hw_val="${fields[hardware]:-cpu}"
    local hw_arr_json
    if [[ "$hw_val" =~ ^\[ ]]; then
        local hw_inner="${hw_val:1:${#hw_val}-2}"
        hw_arr_json="["
        local hfirst=1
        IFS=',' read -ra hitems <<< "$hw_inner"
        for hi in "${hitems[@]}"; do
            hi="${hi// /}"
            [[ $hfirst -eq 1 ]] && hfirst=0 || hw_arr_json+=","
            hw_arr_json+="\"$hi\""
        done
        hw_arr_json+="]"
    else
        hw_arr_json="[\"${hw_val// /}\"]"
    fi

    # --- Build scenarios JSON array ---
    local sc_val="${fields[scenarios]:-s1}"
    local sc_arr_json
    if [[ "$sc_val" =~ ^\[ ]]; then
        local sc_inner="${sc_val:1:${#sc_val}-2}"
        sc_arr_json="["
        local sfirst=1
        IFS=',' read -ra sitems <<< "$sc_inner"
        for si in "${sitems[@]}"; do
            si="${si// /}"
            [[ $sfirst -eq 1 ]] && sfirst=0 || sc_arr_json+=","
            sc_arr_json+="\"$si\""
        done
        sc_arr_json+="]"
    else
        sc_arr_json="[\"${sc_val// /}\"]"
    fi

    # --- Build request_params JSON (if any sub-fields found) ---
    local rp_json=""
    if [[ ${#rp_fields[@]} -gt 0 ]]; then
        local rp_frag=""
        for k in "${!rp_fields[@]}"; do
            [[ -n "$rp_frag" ]] && rp_frag+=","
            rp_frag+="$(_json_kv "$k" "${rp_fields[$k]}")"
        done
        rp_json="{$rp_frag}"
    fi

    # --- Build fixed JSON fragment (fields that are NOT swept) ---
    _build_fixed() {
        local combo="$1"
        # Parse combo into a map of overrides
        declare -A overrides
        for kv in $combo; do
            local ck="${kv%%=*}" cv="${kv#*=}"
            overrides["$ck"]="$cv"
        done

        local frag=""

        # id
        frag+="\"id\":\"$suite_id\""

        # hardware
        frag+=",\"hardware\":${hw_arr_json}"

        # scenarios
        frag+=",\"scenarios\":${sc_arr_json}"

        # Simple scalar fields
        local -a SCALAR_KEYS=(node_count coordinator ptype dtype byte_order lora lora_play_path streaming x_juno_session_id)
        for k in "${SCALAR_KEYS[@]}"; do
            local v="${overrides[$k]:-${fields[$k]:-}}"
            [[ -z "$v" ]] && continue
            # Skip inline lists for swept keys (use the override value instead)
            [[ "$v" =~ ^\[.*\] ]] && continue
            frag+=","
            frag+="$(_json_kv "$k" "$v")"
        done

        # sessions: extract single value from list if single-element
        local sessions_v="${overrides[sessions]:-${fields[sessions]:-1}}"
        if [[ "$sessions_v" =~ ^\[([^,]+)\]$ ]]; then
            sessions_v="${BASH_REMATCH[1]}"
        fi
        [[ "$sessions_v" =~ ^\[.*\] ]] || { frag+=","; frag+="$(_json_kv "sessions" "$sessions_v")"; }

        # request_params
        [[ -n "$rp_json" ]] && frag+=",\"request_params\":$rp_json"

        # request_variants
        [[ "$rv_json" != "null" ]] && frag+=",\"request_variants\":$rv_json"

        printf '%s' "$frag"
    }

    # --- Assemble configs array ---
    local configs_json="["
    local cfirst=1
    for combo in "${combos[@]}"; do
        local cfg_frag
        cfg_frag=$(_build_fixed "$combo")
        [[ $cfirst -eq 1 ]] && cfirst=0 || configs_json+=","
        configs_json+="{${cfg_frag}}"
    done
    configs_json+="]"

    # --- Output ---
    printf '{"suite_id":"%s","model_url":"%s","model_name":"%s","jfr":"%s","hw_defs":%s,"configs":%s}\n' \
        "$suite_id" "$model_url" "$model_name" "$jfr" \
        "$hw_defs_json" "$configs_json"
}

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

# Format a key:value JSON fragment (auto-detects type).
_json_kv() {
    local key="$1" val="$2"
    # Strip surrounding quotes if present
    val="${val#\"}" ; val="${val%\"}"
    val="${val#\'}" ; val="${val%\'}"
    # Number
    if [[ "$val" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        printf '"%s":%s' "$key" "$val"; return
    fi
    # Bool / null
    if [[ "$val" == "true" || "$val" == "false" || "$val" == "null" ]]; then
        printf '"%s":%s' "$key" "$val"; return
    fi
    printf '"%s":"%s"' "$key" "$val"
}

# Parse request_variants YAML block (6-space indented) → JSON object.
# Input: raw YAML lines from the block (passed as a string).
_parse_request_variants() {
    local yaml_block="$1"
    # Each variant is at 6 spaces, its fields at 8 spaces.
    # Sessions list items at 10 spaces, their fields at 12 spaces.
    local out="{"
    local vfirst=1
    local cur_var="" var_frag="" in_sessions=0 sess_arr="" sl_first=1 sl_frag=""

    _flush_var() {
        [[ -z "$cur_var" ]] && return
        if [[ $in_sessions -eq 1 ]]; then
            [[ -n "$sl_frag" ]] && sess_arr+=",{$sl_frag}"; sl_frag=""
            var_frag+=",\"sessions\":[${sess_arr#,}]"
            in_sessions=0; sess_arr=""
        fi
        [[ $vfirst -eq 1 ]] && vfirst=0 || out+=","
        out+="\"$cur_var\":{${var_frag#,}}"
        cur_var=""; var_frag=""
    }

    while IFS= read -r line; do
        # 6-space: variant name
        if [[ "$line" =~ ^"      "([a-z_]+):[[:space:]]*$ ]]; then
            _flush_var
            cur_var="${BASH_REMATCH[1]}"
            in_sessions=0; sess_arr=""; sl_first=1; sl_frag=""
            continue
        fi
        # 8-space: variant field
        if [[ "$line" =~ ^"        "([a-z_]+):[[:space:]]*(.*) ]]; then
            local k="${BASH_REMATCH[1]}" v="${BASH_REMATCH[2]}"
            v="${v%%#*}"; v="${v%"${v##*[^ ]}"}"
            if [[ "$k" == "sessions" ]]; then
                in_sessions=1; sess_arr=""; sl_first=1; sl_frag=""
            else
                var_frag+=","
                var_frag+="$(_json_kv "$k" "$v")"
            fi
            continue
        fi
        # 10-space: sessions list item  "- ..."
        if [[ $in_sessions -eq 1 && "$line" =~ ^"          - "([a-z_]+):[[:space:]]*(.*) ]]; then
            # Flush previous session object
            if [[ -n "$sl_frag" ]]; then
                sess_arr+=",{${sl_frag#,}}"
                sl_frag=""
            fi
            local k="${BASH_REMATCH[1]}" v="${BASH_REMATCH[2]}"
            v="${v%%#*}"; v="${v%"${v##*[^ ]}"}"
            sl_frag="$(_json_kv "$k" "$v")"
            continue
        fi
        # 12-space: additional fields of a session list object
        if [[ $in_sessions -eq 1 && "$line" =~ ^"            "([a-z_]+):[[:space:]]*(.*) ]]; then
            local k="${BASH_REMATCH[1]}" v="${BASH_REMATCH[2]}"
            v="${v%%#*}"; v="${v%"${v##*[^ ]}"}"
            sl_frag+=",$(_json_kv "$k" "$v")"
            continue
        fi
    done <<< "$yaml_block"

    _flush_var
    out+="}"
    printf '%s' "$out"
}