#!/usr/bin/env bash
# Model-major sweep: serve one model once, run every cell against it, unload.
#
# This is the shape the whole project assumes. Engine startup is ~2 min
# (weights + torch.compile + CUDA graphs) against ~90 s for a 5-trial cell, so
# paying it per cell instead of per model would dominate the campaign. It also
# happens to be the only shape Colab allows: `colabctl run` tears down the
# process group when it returns, so the server and every cell it serves must
# live inside one detached job.
#
# Resumes from the corpus: completed cells are skipped, so a reaped runtime
# costs at most the cell in flight.
#
#   scripts/run_sweep.sh MODEL_KEY "S1 S4 M1" "cube_stack spill_wipe" [TRIALS]
set -uo pipefail

MODEL_KEY="${1:?usage: run_sweep.sh MODEL_KEY \"TIERS\" \"TASKS\" [TRIALS]}"
TIERS="${2:?}"; TASKS="${3:?}"; TRIALS="${4:-30}"
PORT="${PORT:-8110}"
CAPXS="${OPENCAP_DIR:-/content/capx-scaling}"
CAPX="${CAPX_DIR:-/content/cap-x}"
# Scratch for engine logs, stats and GPU samples. Was hardcoded to /content,
# which only exists on Colab; set OPENCAP_LOGDIR when running anywhere else.
LOGDIR="${OPENCAP_LOGDIR:-/content}"
mkdir -p "$LOGDIR"
STORE="${OPENCAP_STORE:-hf:savoji/capx-scaling-traces}"
STATS="$LOGDIR/stats_${MODEL_KEY}.jsonl"

# Which physical GPU this sweep owns. Colab had exactly one card, so nothing
# here needed an index; a shared multi-GPU box does.
#
# nvidia-smi does NOT honour CUDA_VISIBLE_DEVICES -- NVML always enumerates
# from physical index 0 -- so every nvidia-smi call below must pass `-i $GPU`
# explicitly. Without it, on a box where another job holds GPU 0, the VRAM
# teardown check reads that job's memory and never clears, and the utilisation
# sampler records that job's load as ours. Both fail silently in the direction
# of looking fine.
GPU="${OPENCAP_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
# MuJoCo's EGL backend enumerates devices through EGL, not CUDA, so it ignores
# CUDA_VISIBLE_DEVICES and would render on physical GPU 0.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPU}"

# The engine needs vLLM installed; CaP-X and this harness deliberately do not
# share that environment (cap-x pins conflicting vLLM versions behind its
# molmo/verl extras, so installing vLLM into its venv corrupts the sim env).
# Point this at the interpreter that has vLLM.
VLLM_PYTHON="${OPENCAP_VLLM_PYTHON:-python3}"
if ! "$VLLM_PYTHON" -c "import vllm" 2>/dev/null; then
  echo "[sweep] $VLLM_PYTHON cannot import vllm -- set OPENCAP_VLLM_PYTHON"; exit 1
fi
# Recorded onto every episode: the serving stack is confounded with the host
# unless it is written down. It was a dead field in the Colab corpus.
export OPENCAP_VLLM_VERSION="$("$VLLM_PYTHON" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo '')"
echo "[sweep] gpu=$GPU vllm=$OPENCAP_VLLM_VERSION python=$VLLM_PYTHON"

cd "$CAPXS" || exit 1

# Sim workers are CPU-bound; Colab ties vCPU count to GPU tier (T4=2, A100=12).
# Oversubscribing thrashes badly enough to turn a 40s run into a >1h timeout.
WORKERS="${OPENCAP_WORKERS:-$(( $(nproc) < 5 ? $(nproc) : 5 ))}"
echo "[sweep] workers=$WORKERS (vCPUs=$(nproc))"

# Skip the whole model if the corpus already has every cell -- avoids paying
# 2 min of engine startup to discover there is nothing to do.
REMAINING=$(python3 - "$MODEL_KEY" "$TIERS" "$TASKS" "$STORE" <<'PY'
import sys
sys.path.insert(0, ".")
from opencap.cells import Cell, cell_exists
from opencap.store import open_store
model, tiers, tasks, store_spec = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4]
done = open_store(store_spec).completed()
todo = [f"{t}:{k}" for t in tiers for k in tasks
        if cell_exists(k, t) and Cell(model, t, k) not in done]
print(" ".join(todo))
PY
)
if [ -z "$REMAINING" ]; then
  echo "[sweep] $MODEL_KEY: nothing left to run"; exit 0
fi
echo "[sweep] $MODEL_KEY: $(echo "$REMAINING" | wc -w) cells to run"

REPO=$(python3 -c "from opencap.models import BY_KEY;print(BY_KEY['${MODEL_KEY}'].repo)")
QUANT=$(python3 -c "
from opencap.models import BY_KEY
p = BY_KEY['${MODEL_KEY}'].precision
print('' if p == 'bf16' else p)")

# 0.80 left only 32GB of a 40GB A100 and put qwen3-14b bf16 ~1GB short of the
# 33GB it needs at a 32K context -- the model was reachable all along, the
# serving config was not. Raised to 0.90 (36GB). Tiers that render (S2/M2/M3)
# need VRAM outside vLLM's pool, so drop this back for Wave 2 perception runs.
GPU_UTIL="${OPENCAP_GPU_UTIL:-0.90}"
# `auto`: size the pool from what is actually free on the card at launch. On a
# shared box the card can carry another account's job (2026-09-14: 76GB of a
# 143GB H200 4 held by someone else's engine), and vLLM refuses to start when
# free memory is below total x util -- 0.90 would then fail every model, not
# just the ones that do not fit. This is NOT a frozen control: at
# OPENCAP_BATCH_MAX=1 exactly one sequence is in flight, and the pool only bounds
# KV concurrency (STATUS 2026-09-09 ran part of the rebuild at 0.40 for the
# same reason). OPENCAP_GPU_UTIL_MAX caps the claim so a briefly empty card does
# not get taken whole from under the other job's next launch.
if [ "$GPU_UTIL" = "auto" ]; then
  # QUANTISED to a 0.10 ladder, deliberately. The unrounded version sized the
  # KV pool from whatever happened to be free at launch, so two runs of the
  # same cell got different pools -- and a different pool changes scheduling,
  # which can flip a token. Measured on the seeded corpus: 1 of 84 baseline
  # cells emitted two distinct programs across its 30 episodes, and 4 of 15 in
  # the replicate control collected later on a busier card, with the splits
  # INTERLEAVED rather than contiguous (so not an engine restart).
  # Rounding down to a coarse ladder makes the pool reproducible across runs
  # unless the card's occupancy has genuinely moved by a tenth, while keeping
  # the shared-card behaviour that `auto` exists for.
  GPU_UTIL=$(nvidia-smi -i "$GPU" --query-gpu=memory.free,memory.total \
               --format=csv,noheader,nounits 2>/dev/null | head -1 | \
    awk -v cap="${OPENCAP_GPU_UTIL_MAX:-0.90}" -F, '{
      u = ($1 - 5000) / $2; if (u > cap) u = cap; if (u < 0.05) u = 0.05;
      q = int(u * 10) / 10; if (q < 0.05) q = 0.05;
      printf "%.2f", q }')
  echo "[sweep] gpu util auto -> $GPU_UTIL of GPU $GPU (quantised to 0.10 steps,"
  echo "[sweep] cap ${OPENCAP_GPU_UTIL_MAX:-0.90}). A pool that varies between runs"
  echo "[sweep] costs within-cell determinism; see STATUS 2026-09-16."
fi
# BATCH_MAX=1: our micro-batching was the single largest source of run-to-run
# variation (spread 5 trials -> 1). Reproducibility beats throughput in an
# evaluation harness.
export OPENCAP_BATCH_MAX="${OPENCAP_BATCH_MAX:-1}"
# OPENCAP_MAX_MODEL_LEN: an UPPER BOUND on prompt + output, not a sampling
# parameter -- at batch 1 without prefix caching it changes no token. Lowered
# only to fit a raised-cap test (8192 out + ~2k prompt) into a shared card.
SHIM_ARGS=(--model "$REPO" --port "$PORT" --max-model-len "${OPENCAP_MAX_MODEL_LEN:-32768}"
           --gpu-memory-utilization "$GPU_UTIL" --seed 0 --stats-file "$STATS")
# --deterministic is REQUIRED, not optional: it disables vLLM's automatic
# prefix caching, without which a warm engine contaminates every cell after
# the first (same cell: 27/30 fresh, 0/30 reused).
#
# That measurement was made on Colab. It is a CONTROL, and controls validated
# on hardware whose decode behaviour we now know differs (artifact #12) have to
# be re-validated, not assumed. OPENCAP_ALLOW_PREFIX_CACHE=1 exists ONLY so
# scripts/run_controls.sh can reproduce the contaminated condition deliberately
# and measure it here. It must never be set for a real arm -- a cell collected
# with prefix caching on is not comparable to anything in the corpus.
if [ "${OPENCAP_ALLOW_PREFIX_CACHE:-0}" = "1" ]; then
  echo "[sweep] WARNING: prefix caching ENABLED. This is a control condition."
  echo "[sweep] Cells collected this way must not enter a baseline store."
else
  SHIM_ARGS+=(--deterministic)
fi
[ -n "$QUANT" ] && SHIM_ARGS+=(--quantization "$QUANT")
# setsid puts the shim in its own process group so we can signal the WHOLE
# tree. vLLM forks a separate EngineCore process that holds the VRAM; killing
# only the parent orphans it, and the next model then dies on OOM. Measured:
# 18.9GB still held after the first model, killing models 2 and 3 instantly.
# Refuse to start on top of another sweep's engine. On 2026-09-09 an orphaned
# pilot.sh (ppid 1, surviving a relaunch that never killed it) still held this
# port; our shim died with "Address already in use" and the sweep ran anyway.
if python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(1)
try: s.connect(('127.0.0.1', $PORT)); sys.exit(0)
except Exception: sys.exit(1)
finally: s.close()" 2>/dev/null; then
  echo "[sweep] :$PORT is already in use -- another sweep or a leftover shim."
  echo "[sweep] refusing to start; clean it up first."
  exit 1
fi

# Memory on the card BEFORE our engine starts. The teardown check below is
# relative to this: on a shared card an absolute "< 2000MiB" can never be met
# while another job holds its own pool, and the check would spend its whole
# window and then warn on every model -- a warning that fires every time is
# not a check.
USED_BEFORE=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
USED_BEFORE="${USED_BEFORE:-0}"

setsid "$VLLM_PYTHON" "$CAPXS/scripts/vllm_shim.py" "${SHIM_ARGS[@]}" \
    > "$LOGDIR/vllm_${MODEL_KEY}.log" 2>&1 &
VLLM_PID=$!

wait_for_vram_release() {
  # CUDA teardown is not instant; starting the next engine too early OOMs.
  for _ in $(seq 1 30); do
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && return 0
    [ "$used" -lt $(( USED_BEFORE + 2000 )) ] && return 0
    sleep 2
  done
  echo "[sweep] WARNING: ${used}MiB resident after teardown (${USED_BEFORE}MiB before launch)"
}

cleanup() {
  kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
  sleep 3
  kill -KILL -"$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null
  wait_for_vram_release
}
trap cleanup EXIT

# The readiness gate used to be a bare curl on /v1/models. That endpoint
# answers from the shim's --model flag without ever touching the engine, so it
# returns 200 both for a shim whose EngineCore has died and for a FOREIGN
# shim left behind by another sweep. Both were true at once on 2026-09-09:
# an orphaned pilot held :8110 with coder-1.5b, our own qwen3-0.6b shim died
# on "Address already in use", this check passed, and qwen3-0.6b's episodes
# were generated by coder-1.5b's weights. A readiness probe that cannot fail
# is worse than no probe -- it turns a loud crash into silent mislabelling.
# The gate now requires three things: our shim alive, the served model id
# equal to the repo we asked for, and a real completion coming back.
serving_expected_model() {
  python3 - "$PORT" "$REPO" <<'PY'
import json, sys, urllib.request
port, repo = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
        served = [m["id"] for m in json.load(r).get("data", [])]
except Exception:
    sys.exit(2)                          # nothing answering yet
sys.exit(0 if repo in served else 3)     # 3 == someone else's engine
PY
}

echo "[sweep] waiting for engine"
for _ in $(seq 1 180); do
  serving_expected_model; rc=$?
  [ "$rc" -eq 0 ] && break
  if [ "$rc" -eq 3 ]; then
    echo "[sweep] PORT HIJACK: :$PORT serves another model, not $REPO."
    echo "[sweep] refusing -- episodes would carry the wrong model label."
    exit 1
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[sweep] engine DIED:"; tail -20 "$LOGDIR/vllm_${MODEL_KEY}.log"; exit 1
  fi
  sleep 5
done
serving_expected_model || { echo "[sweep] engine TIMEOUT"; exit 1; }

# /v1/models is metadata. Only a real completion proves the EngineCore is
# alive behind it -- the zombie that caused this fix answered metadata fine
# for 40 minutes after its EngineCore took a SIGTERM.
echo "[sweep] probing generation"
python3 - "$PORT" "$REPO" <<'PY' || { echo "[sweep] engine NOT GENERATING"; exit 1; }
import json, sys, urllib.request
port, repo = sys.argv[1], sys.argv[2]
body = json.dumps({"model": repo, "max_tokens": 4,
                   "messages": [{"role": "user", "content": "hi"}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
except Exception as e:
    print(f"  generation probe failed: {type(e).__name__}: {e}"); sys.exit(1)
if not out.get("choices"):
    print(f"  generation probe returned no choices: {out}"); sys.exit(1)
PY
echo "[sweep] engine up (generation confirmed)"

# Sample GPU utilisation DURING cells, not between them -- sampling after a
# cell finishes always reads 0% and answers nothing. Gate 1b needs the busy
# figure to decide whether to add sim workers.
UTIL="$LOGDIR/util_${MODEL_KEY}.csv"
( while true; do
    nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
      >> "$UTIL" 2>/dev/null
    sleep 5
  done ) &
UTIL_PID=$!
trap 'kill "$UTIL_PID" 2>/dev/null; cleanup' EXIT

ok=0; fail=0
for spec in $REMAINING; do
  TIER="${spec%%:*}"; TASK="${spec#*:}"
  echo "[sweep] --- $MODEL_KEY $TIER $TASK ---"
  OPENCAP_AGENT_LLM="http://127.0.0.1:${PORT}/v1/chat/completions" \
  OPENCAP_STATS_FILE="$STATS" \
  python3 -u scripts/run_cell.py --model "$MODEL_KEY" --tier "$TIER" --task "$TASK" \
      --capx "$CAPX" --store "$STORE" --trials "$TRIALS" --num-workers "$WORKERS" \
    && ok=$((ok+1)) || fail=$((fail+1))
done
kill "$UTIL_PID" 2>/dev/null
if [ -s "$UTIL" ]; then
  awk -F, '{u+=$1; if($1>mx)mx=$1; n++} END{if(n)printf "[sweep] gpu util: mean %.1f%% peak %d%% over %d samples\n", u/n, mx, n}' "$UTIL"
fi
echo "[sweep] $MODEL_KEY done: $ok ok, $fail failed"
[ "$fail" -eq 0 ]
