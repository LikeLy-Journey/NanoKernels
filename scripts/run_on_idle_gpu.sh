#!/usr/bin/env bash
# Pick the most idle GPU (lowest memory used, then lowest util) and echo its index.
# Usage:
#   GPU=$(scripts/run_on_idle_gpu.sh)          # just print the index
#   scripts/run_on_idle_gpu.sh <cmd> [args...] # run cmd with CUDA_VISIBLE_DEVICES set
set -euo pipefail

pick_idle_gpu() {
  # columns: index, mem.used(MiB), util(%)
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader,nounits \
  | sort -t, -k2,2n -k3,3n \
  | head -1 \
  | cut -d, -f1 \
  | tr -d ' '
}

GPU="$(pick_idle_gpu)"

if [ "$#" -eq 0 ]; then
  echo "$GPU"
  exit 0
fi

echo "[run_on_idle_gpu] selected GPU $GPU" >&2
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$@"
