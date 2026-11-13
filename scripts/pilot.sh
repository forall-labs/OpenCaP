#!/usr/bin/env bash
# Gate 1b pilot. Answers the three questions that size Wave 1:
#   floor        -- does the bottom of the ladder score above zero at n=30?
#   cost         -- real wall-clock per cell, per tier
#   parallelism  -- GPU utilisation, which sets the sim worker count
#
# Model-major: each model's engine is loaded once (~2 min) and every cell for
# that model runs against it. Two tasks rather than three, chosen to bracket
# difficulty, because the pilot needs full n=30 per cell more than it needs
# task breadth -- halving n would defeat the purpose of measuring the floor.
set -uo pipefail

# A relaunch must never coexist with a previous attempt. On 2026-09-09 a
# pilot.sh from an aborted attempt was orphaned to init (ppid 1) and survived
# the relaunch; both ran against one GPU and one shim port, and the surviving
# engine served the *other* arm's model under this arm's label. flock makes
# the second instance refuse rather than race.
exec 9>"${OPENCAP_PILOT_LOCK:-/tmp/opencap_pilot.lock}"
if ! flock -n 9; then
  echo "[pilot] REFUSING: another pilot.sh already holds the lock."
  echo "[pilot] holder: $(cat "${OPENCAP_PILOT_LOCK:-/tmp/opencap_pilot.lock}.pid" 2>/dev/null || echo unknown)"
  echo "[pilot] kill it first, or the two runs will contaminate each other."
  exit 1
fi
echo "pid=$$ started=$(date -u +%FT%TZ)" > "${OPENCAP_PILOT_LOCK:-/tmp/opencap_pilot.lock}.pid"

MODELS="${PILOT_MODELS:-qwen3-0.6b qwen3-1.7b qwen3-8b}"
TIERS="${PILOT_TIERS:-S1 S4 M1}"
TASKS="${PILOT_TASKS:-cube_lifting nut_assembly}"
TRIALS="${PILOT_TRIALS:-30}"
CAPXS="${OPENCAP_DIR:-/content/capx-scaling}"

echo "=== pilot: $(echo $MODELS | wc -w) models x $(echo $TIERS | wc -w) tiers x $(echo $TASKS | wc -w) tasks x ${TRIALS} states ==="
date

for m in $MODELS; do
  echo
  echo "############ $m ############"
  t0=$(date +%s)
  # Do NOT hard-code the host here -- it silently overrode the caller and
  # stamped A100 cells as colab-l4, which would corrupt the very control
  # (analysis.host_effect) that exists to detect cross-machine differences.
  bash "$CAPXS/scripts/run_sweep.sh" "$m" "$TIERS" "$TASKS" "$TRIALS"
  echo "[pilot] $m took $(( $(date +%s) - t0 ))s"
  # -i: nvidia-smi ignores CUDA_VISIBLE_DEVICES, so an unqualified query
  # reports physical GPU 0 -- another job's card on a shared box.
  nvidia-smi -i "${OPENCAP_GPU:-0}" --query-gpu=utilization.gpu,memory.used --format=csv,noheader
done

echo
echo "=== pilot complete ==="
date
