#!/usr/bin/env bash
# Independent check of the one non-zero cell (qwen3-14b S1 cube_lifting = 30/30).
#
# Re-running the same 30 seeds proves only determinism -- same scenes, greedy
# decoding, same code, same result. The question that matters is whether 30/30
# was a favourable draw, so this runs 60 trials: seeds 1-30 reproduce the
# original, seeds 31-60 are a DISJOINT sample of scenes the model has never
# been evaluated on. Comparing the two halves separates "the model solves this
# task" from "the frozen states happened to be easy".
set -uo pipefail
MODEL="${MODEL:-Qwen/Qwen3-14B}"
TRIALS="${TRIALS:-60}"
OUT="${OUT:-/content/outputs/verify60}"
cd /content/capx-scaling || exit 1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

setsid python3 -u scripts/vllm_shim.py --model "$MODEL" --port 8110 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --seed 0 \
  > /content/shim60.log 2>&1 &
SP=$!
cleanup() { kill -TERM -"$SP" 2>/dev/null; sleep 2; kill -KILL -"$SP" 2>/dev/null; }
trap cleanup EXIT

echo "waiting for shim"
for _ in $(seq 1 180); do
  curl -sf -m 3 http://127.0.0.1:8110/v1/models >/dev/null 2>&1 && break
  kill -0 "$SP" 2>/dev/null || { echo SHIM_DIED; tail -6 /content/shim60.log; exit 1; }
  sleep 5
done
curl -sf -m 3 http://127.0.0.1:8110/v1/models >/dev/null || { echo SHIM_TIMEOUT; exit 1; }
echo "shim up; running $TRIALS trials"

cd /content/cap-x || exit 1
uv run capx/envs/launch.py \
  --config-path env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml \
  --server-url http://127.0.0.1:8110/v1/chat/completions --model "$MODEL" \
  --temperature 0.0 --max-tokens 2048 --reasoning-effort medium \
  --total-trials "$TRIALS" --num-workers 5 --record-video False \
  --use-parallel-ensemble False --use-multimodel False --output-dir "$OUT"
echo "V60_RC=$?"
