#!/usr/bin/env bash
# One line per second: UTC time, device free MiB, then pid:MiB for every process on the GPU. Stops when the given file appears.
stop=$1
while [ ! -e "$stop" ]; do
  t=$(date -u +%H:%M:%S)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  procs=$(nvidia-smi | awk '/^\|  +0 +N\/A/ {pid=$5; mem=$(NF-1); sub(/MiB/,"",mem); printf "%s:%s ", pid, mem}')
  echo "$t free=$free $procs"
  sleep 1
done
