#!/usr/bin/env bash
# The fourth service the reduced API binds, and the only one with no launcher
# in the checkout: molmo, the pointing model behind `point_prompt_molmo`.
#
#     bash scripts/start_molmo_server.sh           # start if not already up
#     bash scripts/start_molmo_server.sh --check   # report only, exit 1 if down
#
# WHY THIS EXISTS
# ---------------
# `capx/integrations/vision/molmo.py` line 20 hardcodes
# `SERVICE_URL = "http://127.0.0.1:8122/v1"`, and `control_reduced.py` calls
# `init_molmo()` unconditionally in its constructor, so EVERY S3/S4 cell is
# handed a `point_prompt_molmo` primitive. But no config declares port 8122,
# so CaP-X's `_start_api_servers` never starts it, and `capx/serving/` ships
# launchers for graspnet, curobo, owlvit, pyroki, sam2 and sam3 -- and nothing
# for molmo. The service simply cannot come up from this checkout.
#
# It cost 25 cells across S3 and S4, and worse than that: the S4 manipulation
# (no in-context example) makes a model MORE likely to reach for molmo, so the
# refusals are correlated with the treatment. Two of the three models that
# ever scored at S3 cube_lifting lost their S4 cell this way, including
# qwen35-9b 5/30, the best cell in the tier. That is a collider, and it made
# the S3-vs-S4 comparison uninterpretable. See STATUS 2026-10-01.
#
# WHAT IT SERVES AND WHY THAT MODEL
# ---------------------------------
# Not a guess. `init_molmo()` defaults to `model_name="allenai/Molmo2-8B"` and
# its own docstring says "the served model name (as registered in vLLM)", with
# the OpenRouter alternative commented out directly above. The client builds
# `{base_url}/chat/completions` and posts an OpenAI vision payload. So a plain
# vLLM OpenAI server on 8122 serving that repo is exactly what the code was
# written against -- this restores the intended configuration rather than
# inventing one.
#
# The OpenRouter path is deliberately NOT taken here: it would send every
# observation image to a third party, which is not a decision a launcher
# script should make quietly.
#
# --trust-remote-code IS REQUIRED AND IS A REAL DECISION. vLLM refuses the
# repo without it ("contains custom code which must be executed to correctly
# load the model"), because Molmo ships its own modeling classes rather than
# using a stock architecture. The flag executes Python downloaded from the
# HF repo on this machine. It is enabled here because the repo is public,
# ungated, published by AllenAI, and is the exact model `init_molmo()` names
# as its default -- i.e. restoring the intended configuration, not choosing a
# new one. It is still third-party code execution on a shared box, so it is
# called out here rather than left as an unexplained flag.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

PORT=8122
MODEL="${OPENCAP_MOLMO_MODEL:-allenai/Molmo2-8B}"
UTIL="${OPENCAP_MOLMO_UTIL:-0.20}"
LOGDIR="${OPENCAP_LOGDIR:-/data/capx-logs}"
LOG="$LOGDIR/molmo_server.log"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

up() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

if up; then
  echo "[molmo] already listening on :$PORT"
  exit 0
fi
if [ "$CHECK_ONLY" = 1 ]; then
  echo "[molmo] NOT listening on :$PORT"
  exit 1
fi

# Molmo2-8B is PUBLIC and ungated, unlike facebook/sam3 -- so our own writable
# cache is correct here and HF_HUB_OFFLINE must NOT be set: the weights are
# not present yet and the first start downloads them.
echo "[molmo] starting $MODEL on :$PORT, GPU $OPENCAP_GPU, pool $UTIL"
echo "[molmo] HF_HOME=$HF_HOME (weights download on first run, ~16-20 GB)"
echo "[molmo] log $LOG"

# APPEND, never truncate -- same reason as start_vision_servers.sh: a restart
# is exactly when someone needs to know how the last instance died.
{
  echo
  echo "======== molmo $MODEL started $(date -u +'%Y-%m-%d %H:%M:%S UTC') ========"
} >> "$LOG"

CUDA_VISIBLE_DEVICES="$OPENCAP_GPU" setsid nohup \
  "${OPENCAP_VLLM_PYTHON:-/data/venvs/vllm028/bin/python}" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len 4096 \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-stats \
    --seed 0 \
  >> "$LOG" 2>&1 < /dev/null &

# First start downloads weights, so the window is generous. A readiness probe
# that gives up during a download would leave a half-fetched cache and a
# "failed" verdict on a service that was working.
#
# 2026-10-01: the first version set this to 240 (20 min) and gave up while the
# download was still running -- the vLLM process kept going, reached 33 GB and
# came up fine, but the launcher had already printed a failure. Exactly the
# mistake the comment above warns about, made in the same file. Molmo2-8B is
# ~33 GB over eight shards and HF read timeouts force retries, so the window
# has to cover a slow fetch, not a fast one.
WAIT="${OPENCAP_MOLMO_WAIT:-720}"   # x5s = 60 min default
for i in $(seq 1 "$WAIT"); do
  sleep 5
  if up; then echo "[molmo] listening after ~$((i * 5))s"; break; fi
  if [ "$i" = "$WAIT" ]; then
    echo "[molmo] DID NOT COME UP in $((WAIT * 5))s. Last lines of $LOG:"
    tail -20 "$LOG"
    exit 1
  fi
done

# A port is not a working model. The pointing calls go to /chat/completions
# with an image, so prove the OpenAI route answers before declaring success --
# the same lesson as the sweep's "probing generation" step, which exists
# because an engine answered metadata for 40 minutes after its core died.
echo "[molmo] probing /v1/chat/completions"
python3 - "$PORT" "$MODEL" <<'PY' || { echo "[molmo] UP BUT NOT ANSWERING"; exit 1; }
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
body = json.dumps({"model": model, "max_tokens": 8, "temperature": 0.0,
                   "messages": [{"role": "user", "content": "hi"}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        json.load(r)["choices"][0]["message"]
    print("[molmo] generation confirmed")
except Exception as e:
    print(f"[molmo] probe failed: {type(e).__name__}: {e}")
    raise SystemExit(1)
PY

echo "[molmo] ready on :$PORT"
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$OPENCAP_GPU"
exit 0
