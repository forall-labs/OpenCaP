#!/usr/bin/env bash
# Does a cell's result depend on what ran BEFORE it on the same engine?
#
# run_sweep.sh starts one shim per model and serves every cell of that model
# from it. The determinism runs always used a fresh engine. Five fresh-engine
# measurements of qwen3-4b S1 cube_stack gave 22-28; the sweep, where
# cube_lifting ran first on the same engine, gave 8. That is far outside the
# ~1-trial noise floor, so engine state is the prime suspect -- most likely
# automatic prefix caching, where a warm cache takes a different code path.
#
# If confirmed, cell ORDER is a confound in every multi-cell sweep, and the
# harness must either reset the engine per cell or disable prefix caching.
#
#   run H: fresh engine -> cube_stack            (expect ~27)
#   run I: same engine  -> cube_lifting, then cube_stack   (sweep conditions)
set -uo pipefail
MODEL="${OE_MODEL:-Qwen/Qwen3-4B}"
LIFT=env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml
STACK=env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml

# OE_DET=1 disables prefix caching. If the order effect is caused by a warm
# KV cache taking a different code path, this removes it -- and lets us keep
# model-major sweeps, saving ~2 min of engine load per cell. If the effect
# survives, the only remedy is a fresh engine per cell.
DET_FLAG=""
[ "${OE_DET:-0}" = 1 ] && DET_FLAG="--deterministic"

start_shim () {
  cd /content/capx-scaling || exit 1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENCAP_BATCH_MAX=1
  # shellcheck disable=SC2086
  setsid python3 -u scripts/vllm_shim.py --model "$MODEL" --port 8110 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 --seed 0 $DET_FLAG \
    > "/content/oe_shim_$1.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 180); do
    curl -sf -m 3 http://127.0.0.1:8110/v1/models >/dev/null 2>&1 && return 0
    kill -0 $SP 2>/dev/null || { echo "$1: SHIM_DIED"; return 1; }
    sleep 5
  done
  return 1
}
stop_shim () {
  kill -TERM -$SP 2>/dev/null; sleep 3; kill -KILL -$SP 2>/dev/null
  for _ in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -lt 2000 ] && break; sleep 2
  done
}
run_cfg () {  # $1 = config, $2 = tag
  cd /content/cap-x || exit 1
  uv run capx/envs/launch.py --config-path "$1" \
    --server-url http://127.0.0.1:8110/v1/chat/completions --model "$MODEL" \
    --temperature 0.0 --max-tokens 2048 --reasoning-effort medium \
    --total-trials 30 --num-workers 5 --record-video False \
    --use-parallel-ensemble False --use-multimodel False \
    --output-dir "/content/outputs/oe_$2" 2>&1 | grep -E '^[0-9]\.[0-9]+/' | tail -1
  cd /content/capx-scaling || exit 1
}

echo "=== run H: FRESH engine, cube_stack only ==="
start_shim H && run_cfg "$STACK" H_stack
stop_shim

echo "=== run I: SAME engine, cube_lifting FIRST, then cube_stack ==="
start_shim I || exit 1
echo "-- warming with cube_lifting --"
run_cfg "$LIFT" I_lift
echo "-- now cube_stack on the warmed engine --"
run_cfg "$STACK" I_stack
stop_shim
echo OE_DONE
