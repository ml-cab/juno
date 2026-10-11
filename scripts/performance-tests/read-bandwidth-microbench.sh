#!/usr/bin/env bash
# Host attainable memory read bandwidth: the denominator of the CPU decode
# roofline (weight bytes read per token x tg over this figure).
#
# Runs ReadBandwidth.java (the JVM reading) and, when a C compiler with OpenMP
# is present, read-bandwidth.c (a native cross-check) over the same buffer
# size and thread counts, and records the host state alongside: CPU model and
# flags, governor, turbo, load average, DIMM population from EDAC, and the DIMM
# speed when dmidecode can be read without a password.
#
# Usage: read-bandwidth-microbench.sh [--out DIR] [--gib N] [--threads 1,2,...]
#                                     [--reps N] [--passes N]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

OUT_DIR="target/read-bandwidth"
GIB=3
THREADS="1,2,4,6,8,12"
REPS=3
PASSES=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --gib) GIB="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --passes) PASSES="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/results.jsonl"
HOST="$OUT_DIR/host.txt"
: > "$RESULTS"

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "logical_cpus: $(nproc)"
  echo "simd_flags: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse4_2|avx|avx2|fma|avx512f|avx512_vnni|avx_vnni)$' | paste -sd' ')"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown)"
  echo "loadavg_before: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "jvm: $(java -version 2>&1 | head -1)"
  echo "jvm_use_avx: $(java -XX:+PrintFlagsFinal -version 2>/dev/null | awk '$2=="UseAVX"{print $4}')"
  if [[ -d /sys/devices/system/edac/mc/mc0 ]]; then
    echo "memory_controller: $(cat /sys/devices/system/edac/mc/mc0/mc_name)"
    for d in /sys/devices/system/edac/mc/mc0/dimm*; do
      echo "dimm: $(cat "$d/dimm_label") size_mb=$(cat "$d/size") type=$(cat "$d/dimm_mem_type")"
    done
  fi
  if sudo -n true 2>/dev/null; then
    echo "dimm_speed: $(sudo -n dmidecode -t memory | grep -E '^\s*(Configured Memory Speed|Speed):' | sort | uniq -c | paste -sd';')"
  else
    echo "dimm_speed: not read (dmidecode needs root; run: sudo dmidecode -t memory | grep -i speed)"
  fi
} > "$HOST"
cat "$HOST"

HEAP_GIB=$(awk -v g="$GIB" 'BEGIN { printf "%d", g + 2 }')
echo "[read-bandwidth] java: ${GIB} GiB buffer, threads ${THREADS}, reps ${REPS}, passes ${PASSES}"
java "-Xms${HEAP_GIB}g" "-Xmx${HEAP_GIB}g" -XX:+UseParallelGC \
  "$HERE/ReadBandwidth.java" --gib "$GIB" --threads "$THREADS" --reps "$REPS" --passes "$PASSES" \
  | tee -a "$RESULTS"

if command -v cc >/dev/null 2>&1; then
  BIN="$OUT_DIR/read-bandwidth"
  if cc -O3 -march=native -fopenmp "$HERE/read-bandwidth.c" -o "$BIN" 2>/dev/null; then
    echo "[read-bandwidth] native cross-check"
    OMP_PROC_BIND=false "$BIN" "$GIB" "$THREADS" "$REPS" "$PASSES" | tee -a "$RESULTS"
  else
    echo "[read-bandwidth] native cross-check skipped: cc -fopenmp failed"
  fi
fi
echo "loadavg_after: $(cut -d' ' -f1-3 /proc/loadavg)" >> "$HOST"
echo "[read-bandwidth] results: $RESULTS"
