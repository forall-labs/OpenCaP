#!/usr/bin/env bash
# Serve a model and run cells against it inside ONE process group.
#
# colabctl's `run` tears down the process group when the call returns, so a
# vLLM daemon started by one `run` is dead before the next one arrives. Server
# and client therefore have to live in the same detached job.
#
# Usage: serve_and_run.sh MODEL_KEY TIER TASK [TRIALS]
set -uo pipefail

MODEL_KEY="${1:?usage: serve_and_run.sh MODEL_KEY TIER TASK [TRIALS]}"
TIER="${2:?}"; TASK="${3:?}"; TRIALS="${4:-5}"
PORT="${PORT:-8110}"
CAPXS="${OPENCAP_DIR:-/content/capx-scaling}"
CAPX="${CAPX_DIR:-/content/cap-x}"
# Default to the Hub: /content dies with the runtime, and Colab reaps
# runtimes every 30-60 min. A local-only corpus loses work on every reap.
STORE="${OPENCAP_STORE:-hf:savoji/capx-scaling-traces}"

REPO=$(cd "$CAPXS" && python3 -c "from opencap.models import BY_KEY;print(BY_KEY['${MODEL_KEY}'].repo)")
QUANT=$(cd "$CAPXS" && python3 -c "
from opencap.models import BY_KEY
p = BY_KEY['${MODEL_KEY}'].precision
print('' if p == 'bf16' else p)")

echo "[serve] ${MODEL_KEY} -> ${REPO}"
# `vllm serve` initialises the engine then never binds its port on Colab
# (reproduced twice, stalling after CUDA graph capture with the process alive).
# The shim keeps vLLM's engine and replaces only the HTTP layer.
STATS="/content/stats_${MODEL_KEY}.jsonl"
SHIM_ARGS=(--model "$REPO" --port "$PORT" --max-model-len 32768
           --gpu-memory-utilization 0.80 --seed 0 --stats-file "$STATS")
[ -n "$QUANT" ] && SHIM_ARGS+=(--quantization "$QUANT")
python3 "$CAPXS/scripts/vllm_shim.py" "${SHIM_ARGS[@]}"     > /content/vllm_${MODEL_KEY}.log 2>&1 &
VLLM_PID=$!
cleanup() { echo "[serve] stopping $VLLM_PID"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[serve] waiting for :${PORT}"
for _ in $(seq 1 120); do
  # The shim answers 503 while loading, 200 once the engine is ready.
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[serve] up"; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[serve] DIED before binding; last lines:"
    tail -20 "/content/vllm_${MODEL_KEY}.log"
    exit 1
  fi
  sleep 5
done
curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { echo "[serve] TIMEOUT"; exit 1; }

cd "$CAPXS" || exit 1
OPENCAP_AGENT_LLM="http://127.0.0.1:${PORT}/v1/chat/completions" \
OPENCAP_STATS_FILE="$STATS" \
python3 scripts/run_cell.py --model "$MODEL_KEY" --tier "$TIER" --task "$TASK" \
    --capx "$CAPX" --store "$STORE" --trials "$TRIALS" --num-workers 5
echo "CELL_EXIT=$?"
