#!/usr/bin/env bash
# Is generation reproducible on a FIXED host, and does micro-batching break it?
#
# qwen3-4b S1 cube_stack gave 24/30 on one runtime and 8/30 on another, with
# identical seeds, greedy decoding and code. Two candidate causes:
#
#   (a) HOST effect      -- different physical A100, different numerics
#   (b) BATCHING effect  -- the shim batches up to 8 requests on a 50ms window,
#                           so batch composition depends on arrival timing, and
#                           vLLM numerics are batch-size dependent. Greedy
#                           decoding is then NOT bit-reproducible.
#
# This runs the SAME cell three times on ONE host: twice with batching as
# shipped, once with BATCH_MAX=1 (batching disabled).
#   runs 1,2 differ            -> (b), batching is the culprit; fix it
#   runs 1,2 agree but != 24/30 -> (a), a host effect; needs a host control
#   all three agree at 24/30    -> neither; look elsewhere
set -uo pipefail
MODEL="${DT_MODEL:-Qwen/Qwen3-4B}"
KEY="${DT_KEY:-qwen3-4b}"
CFG=env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml

run_once () {  # $1 = tag, $2 = batch_max, $3 = "det" to enable deterministic mode
  cd /content/capx-scaling || exit 1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENCAP_BATCH_MAX="$2"
  local DET=""
  [ "${3:-}" = det ] && DET="--deterministic"
  # shellcheck disable=SC2086
  setsid python3 -u scripts/vllm_shim.py --model "$MODEL" --port 8110 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 --seed 0 $DET \
    > "/content/dt_shim_$1.log" 2>&1 &
  local SP=$!
  for _ in $(seq 1 180); do
    curl -sf -m 3 http://127.0.0.1:8110/v1/models >/dev/null 2>&1 && break
    kill -0 $SP 2>/dev/null || { echo "$1: SHIM_DIED"; return 1; }
    sleep 5
  done
  cd /content/cap-x || exit 1
  uv run capx/envs/launch.py --config-path "$CFG" \
    --server-url http://127.0.0.1:8110/v1/chat/completions --model "$MODEL" \
    --temperature 0.0 --max-tokens 2048 --reasoning-effort medium \
    --total-trials 30 --num-workers 5 --record-video False \
    --use-parallel-ensemble False --use-multimodel False \
    --output-dir "/content/outputs/dt_$1" 2>&1 | grep -E '^1\.000|Task completed|Code generation' | tail -2
  kill -TERM -$SP 2>/dev/null; sleep 3; kill -KILL -$SP 2>/dev/null
  for _ in $(seq 1 30); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -lt 2000 ] && break; sleep 2
  done
}

# A single BATCH_MAX=1 run cannot demonstrate determinism -- it is one draw,
# indistinguishable from another sample of the same noisy distribution. Only a
# MATCHING PAIR shows reproducibility. The first version of this test ran
# BATCH_MAX=1 once, which was a design flaw.
MODE="${DT_MODE:-full}"
if [ "$MODE" = "det" ]; then
  # The decisive test: a MATCHED PAIR with every known nondeterminism source
  # disabled. Anything less than an exact match means reproducibility is not
  # achievable on this stack and must be reported as a noise floor instead.
  echo "=== run F: deterministic (batch=1, no prefix cache, eager) ==="; run_once F 1 det
  echo "=== run G: deterministic (batch=1, no prefix cache, eager) ==="; run_once G 1 det
elif [ "$MODE" = "nobatch" ]; then
  echo "=== run D: batching disabled (BATCH_MAX=1) ==="; run_once D 1
  echo "=== run E: batching disabled (BATCH_MAX=1) ==="; run_once E 1
else
  echo "=== run A: batching as shipped (BATCH_MAX=8) ==="; run_once A 8
  echo "=== run B: batching as shipped (BATCH_MAX=8) ==="; run_once B 8
  echo "=== run C: batching disabled (BATCH_MAX=1)  ==="; run_once C 1
fi
echo DT_DONE
