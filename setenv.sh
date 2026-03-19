#!/usr/bin/env bash
# Created by Yevhen Soldatov
# Initial implementation: 2026
#
# Optional: set up PATH and LD_LIBRARY_PATH for GPU runs so JavaCPP/cudart can load libcudart.so.
# Source before ./run.sh cluster ... --gpu if CUDA is not already on PATH.
#
# Usage: source setenv.sh   (then ./run.sh cluster ... --gpu)
#    or: . setenv.sh

if [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH/bin" ]; then
  export PATH="$CUDA_PATH/bin:$PATH"
  [ -d "$CUDA_PATH/lib64" ] && export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[juno] Added CUDA_PATH to PATH and LD_LIBRARY_PATH"
fi
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/bin" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  [ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[juno] Added CUDA_HOME to PATH and LD_LIBRARY_PATH"
fi
if [ -z "$CUDA_PATH" ] && [ -z "$CUDA_HOME" ]; then
  echo "[juno] Set CUDA_PATH or CUDA_HOME to your NVIDIA CUDA install, e.g. /usr/local/cuda"
fi
