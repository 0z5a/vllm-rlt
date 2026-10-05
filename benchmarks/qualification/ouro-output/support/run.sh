#!/usr/bin/env bash
# Start only after explicit whole-window handoff. Locks fail without waiting.
set -u
bundle=$(cd -- "$(dirname -- "$0")/.." && pwd)
output="$bundle/results"
mkdir -p "$bundle/logs"
exec 8<>/root/autodl-tmp/0z5a/gpu0-perf.lock
exec 9<>/root/autodl-tmp/0z5a/heavy-io.lock
flock -n 8 || exit 73
flock -n 9 || exit 74
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$bundle/runtime:$bundle/support"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="$bundle/triton-cache"
export CUDA_CACHE_PATH="$bundle/cuda-cache"
/root/miniconda3/bin/python -u "$bundle/support/qualify.py" \
    --bundle "$bundle" \
    --model /root/autodl-tmp/0z5a/flashrlt-20261005/models/ouro-1.4b \
    --output "$output" > "$bundle/logs/qualification.log" 2>&1
result=$?
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$bundle/logs/ended-utc.txt"
printf '{"natural_exit_code": %s}\n' "$result" > "$bundle/logs/exit.json"
exit "$result"
