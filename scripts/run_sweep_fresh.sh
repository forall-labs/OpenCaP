#!/usr/bin/env bash
# Sweep with a FRESH ENGINE PER CELL.
#
# run_sweep.sh starts one engine per model and serves every cell from it. That
# is ~2 min cheaper per cell, and it corrupts the data: the same cell scored
# 27/30 on a fresh engine and 0/30 when served by an engine that had already
# run a different cell. Only the first cell of each model was ever measured on
# a clean engine.
#
# This variant reloads the engine for every cell. It costs ~2 min/cell -- ~70
# extra minutes on a 36-cell sweep -- and it is the price of a measurement
# that does not depend on sweep order.
#
#   scripts/run_sweep_fresh.sh MODEL_KEY "S1 S4" "cube_lifting cube_stack" [TRIALS]
set -uo pipefail
MODEL_KEY="${1:?usage: run_sweep_fresh.sh MODEL_KEY \"TIERS\" \"TASKS\" [TRIALS]}"
TIERS="${2:?}"; TASKS="${3:?}"; TRIALS="${4:-30}"
PORT="${PORT:-8110}"
CAPXS="${OPENCAP_DIR:-/content/capx-scaling}"
CAPX="${CAPX_DIR:-/content/cap-x}"
STORE="${OPENCAP_STORE:-hf:savoji/capx-scaling-traces}"
export OPENCAP_BATCH_MAX="${OPENCAP_BATCH_MAX:-1}"

cd "$CAPXS" || exit 1
WORKERS="${OPENCAP_WORKERS:-$(( $(nproc) < 5 ? $(nproc) : 5 ))}"
REPO=$(python3 -c "from opencap.models import BY_KEY;print(BY_KEY['${MODEL_KEY}'].repo)")
QUANT=$(python3 -c "
from opencap.models import BY_KEY
p = BY_KEY['${MODEL_KEY}'].precision
print('' if p == 'bf16' else p)")

ok=0; fail=0; skip=0
for TIER in $TIERS; do
  for TASK in $TASKS; do
    exists=$(python3 -c "
from opencap.cells import cell_exists
print('1' if cell_exists('$TASK', '$TIER') else '0')")
    [ "$exists" = 1 ] || { skip=$((skip+1)); continue; }
    done_already=$(python3 -c "
import sys; sys.path.insert(0,'.')
from opencap.cells import Cell
from opencap.store import open_store
print('1' if Cell('$MODEL_KEY','$TIER','$TASK') in open_store('$STORE').completed() else '0')")
    [ "$done_already" = 1 ] && { echo "[fresh] skip $TIER $TASK (in corpus)"; skip=$((skip+1)); continue; }

    echo "[fresh] ===== $MODEL_KEY $TIER $TASK (clean engine) ====="
    STATS="/content/stats_${MODEL_KEY}_${TIER}_${TASK}.jsonl"
    SHIM=(--model "$REPO" --port "$PORT" --max-model-len 32768
          --gpu-memory-utilization "${OPENCAP_GPU_UTIL:-0.90}" --seed 0 --stats-file "$STATS")
    [ -n "$QUANT" ] && SHIM+=(--quantization "$QUANT")
    [ "${OPENCAP_DETERMINISTIC:-0}" = 1 ] && SHIM+=(--deterministic)

    setsid python3 -u "$CAPXS/scripts/vllm_shim.py" "${SHIM[@]}" \
      > "/content/vllm_${MODEL_KEY}_${TIER}_${TASK}.log" 2>&1 &
    SP=$!
    up=0
    for _ in $(seq 1 180); do
      curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { up=1; break; }
      kill -0 "$SP" 2>/dev/null || break
      sleep 5
    done
    if [ "$up" != 1 ]; then
      echo "[fresh] engine failed for $TIER $TASK"; tail -5 "/content/vllm_${MODEL_KEY}_${TIER}_${TASK}.log"
      fail=$((fail+1))
    else
      OPENCAP_AGENT_LLM="http://127.0.0.1:${PORT}/v1/chat/completions" OPENCAP_STATS_FILE="$STATS" \
      python3 -u scripts/run_cell.py --model "$MODEL_KEY" --tier "$TIER" --task "$TASK" \
        --capx "$CAPX" --store "$STORE" --trials "$TRIALS" --num-workers "$WORKERS" \
        && ok=$((ok+1)) || fail=$((fail+1))
    fi

    kill -TERM -"$SP" 2>/dev/null; sleep 3; kill -KILL -"$SP" 2>/dev/null
    for _ in $(seq 1 30); do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -z "$u" ] && break
      [ "$u" -lt 2000 ] && break
      sleep 2
    done
  done
done
echo "[fresh] $MODEL_KEY done: $ok ok, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
