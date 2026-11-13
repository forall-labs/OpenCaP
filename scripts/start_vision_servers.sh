#!/usr/bin/env bash
# Bring up ALL the services the S3/S4 configs declare, ONCE, before a tier
# grid: SAM3 (segmentation), ContactGraspNet (grasp planning), PyRoKi (IK).
#
#     bash scripts/start_vision_servers.sh          # start if not already up
#     bash scripts/start_vision_servers.sh --check  # report only, exit 1 if down
#
# WHY THIS EXISTS
# ---------------
# S3 and S4 are NOT text-only tiers. Their configs carry an `api_servers:`
# block, and the reduced API they bind
# (capx/integrations/franka/control_reduced.py) hands the model
# segment_sam3_*, detect_object_owlvit, point_prompt_molmo, plan_grasp and
# solve_ik. A generated program that reaches for any of those dies without
# the service behind it.
#
# THREE services, not one -- found the hard way, one failed cell at a time:
#   8114  sam3              segmentation
#   8115  contact_graspnet  plan_grasp
#   8116  pyroki            solve_ik
#
# And the declarations are not complete. env_configs/spill_wipe/
# ...reduced_api.yaml declares only 8114 and 8116, yet the API it binds
# advertises plan_grasp, which calls 8115 -- qwen3-8b duly wrote a program
# that used it and lost all 30 trials. Starting the union of all three, for
# every task, is the only version of this that does not fail on the next
# primitive nobody predicted.
#
# CaP-X starts those servers itself -- but `_start_api_servers` waits 120s for
# the port and then prints a WARNING and RUNS THE CELL ANYWAY. On 2026-09-28
# that produced qwen3-0.6b S3 spill_wipe at 0/30 with all thirty trials dead on
# "Failed to communicate with SAM3 service at 127.0.0.1:8114", because the card
# was too full for SAM3 to load beside the LLM engine. A warning instead of an
# error, and a full cell of plausible zeros -- the house pattern.
#
# Starting it ONCE, here, fixes three things at the same time:
#   * `_start_api_servers` explicitly skips a port already in use ("already
#     running, skipping"), so every cell reuses this instance;
#   * the 4 GB reservation is made once and stays visible, instead of being
#     raced 168 times against a neighbour's swings;
#   * a failure to start is loud and happens BEFORE 19 hours of collection,
#     not silently inside cell 37.
#
# The server is identical to the one the config would spawn -- same entry
# point, same device, same port. Only its lifetime differs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

LOGDIR="${OPENCAP_LOGDIR:-/data/capx-logs}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# name:port:python-call
SERVERS=(
  "sam3:8114:from capx.serving.launch_sam3_server import main; main(device='cuda', port=8114)"
  "graspnet:8115:from capx.serving.launch_contact_graspnet_server import main; main(device='cuda', port=8115)"
  "pyroki:8116:from capx.serving.launch_pyroki_server import main; main(robot='panda_description', target_link='panda_hand', port=8116)"
)

up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# HF_HOME, DELIBERATELY NOT OURS.
#
# env_server.sh pins HF_HOME=/data/hf_cache unconditionally, and with
# good reason: /shared/huggingface is world-writable at the top level but each
# model directory is owned by whoever pulled it first, so a DOWNLOAD into it
# dies with EACCES partway through.
#
# SAM3 is the case that rule breaks. `facebook/sam3` is a GATED repo and this
# account is not on the authorized list, so resolving it against our private
# cache 403s -- the weights cannot be fetched at all. They are already in the
# shared cache, pulled by someone who does have access, and we only need to
# READ them. HF_HUB_OFFLINE stops any attempt to re-fetch.
#
# This is why a hand-run start worked and the first scripted one did not: the
# hand-run command never sourced env_server.sh.
export HF_HOME=/shared/huggingface
export HF_HUB_OFFLINE=1

MISSING=0
for spec in "${SERVERS[@]}"; do
  NAME="${spec%%:*}"; rest="${spec#*:}"; PORT="${rest%%:*}"; CALL="${rest#*:}"
  LOG="$LOGDIR/${NAME}_server.log"

  if up "$PORT"; then
    echo "[vision] $NAME already listening on :$PORT"
    continue
  fi
  if [ "$CHECK_ONLY" = 1 ]; then
    echo "[vision] $NAME NOT listening on :$PORT"
    MISSING=$((MISSING + 1))
    continue
  fi

  echo "[vision] starting $NAME on :$PORT, GPU $OPENCAP_GPU, log $LOG"
  # APPEND, NEVER TRUNCATE. On 2026-09-29 all three services were killed
  # abruptly -- GraspNet's log ends mid-request on a 200 OK, no traceback, no
  # shutdown line -- and restarting with `>` truncated sam3_server.log before
  # anyone had read why SAM3 died. That cause is now unrecoverable. The log of
  # a service that died is the only evidence of how it died, and a restart is
  # exactly the moment someone is about to need it.
  {
    echo
    echo "======== $NAME started $(date -u +'%Y-%m-%d %H:%M:%S UTC') "\
"gpu=$OPENCAP_GPU pid=$$ ========"
  } >> "$LOG"
  ( cd "$CAPX_DIR" && CUDA_VISIBLE_DEVICES="$OPENCAP_GPU" \
      setsid nohup uv run --no-sync python -c "$CALL" >> "$LOG" 2>&1 < /dev/null & )

  READY=0
  for i in $(seq 1 72); do
    sleep 5
    if up "$PORT"; then echo "[vision] $NAME ready after ~$((i * 5))s"; READY=1; break; fi
  done
  if [ "$READY" = 0 ]; then
    echo "[vision] $NAME DID NOT COME UP in 360s. Last lines of $LOG:"
    tail -15 "$LOG"
    MISSING=$((MISSING + 1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo "[vision] $MISSING service(s) down. Do NOT collect S3/S4: launch.py"
  echo "[vision] warns and runs every cell anyway, scoring a dead service as"
  echo "[vision] a model failure -- 30 plausible zeros with no error."
  exit 1
fi
echo "[vision] all services up"
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$OPENCAP_GPU"
exit 0
