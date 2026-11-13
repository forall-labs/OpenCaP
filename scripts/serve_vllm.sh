#!/usr/bin/env bash
# Launch vLLM for one model from the ladder, with the roadmap's frozen settings.
# Usage: scripts/serve_vllm.sh qwen3-8b [PORT] [GPU_ID]
set -euo pipefail

MODEL_KEY="${1:?usage: serve_vllm.sh MODEL_KEY [PORT] [GPU_ID]}"
PORT="${2:-8000}"
GPU_ID="${3:-0}"

ARGS=$(python3 -c "
from opencap.models import BY_KEY
import shlex, sys
m = BY_KEY['${MODEL_KEY}']
print(' '.join(shlex.quote(a) for a in m.vllm_args(${PORT})))
")

# Wave 1 needs no perception GPU, so on T1 both cards serve models: pin each
# server to its own card and run two model keys concurrently.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_LOGGING_LEVEL=WARNING

echo "[serve] ${MODEL_KEY} on GPU ${GPU_ID}, port ${PORT}"
echo "[serve] args: ${ARGS}"
# shellcheck disable=SC2086
exec vllm serve ${ARGS} --disable-log-requests
