#!/usr/bin/env bash
# Sequential verification queue for hand-written minimal repairs.
#
# Append lines "task tier pid code_file [--save] [--no-defect]" to $Q; each runs
# once, in order, through verify_reduced_repair.py. Repairs get written faster
# than they verify, so the GPU works through the backlog while the next ones
# are being read and written. No free-text args: lines are word-split.
#
#   bash scripts/vrr_queue.sh            # foreground; usually run in background
#   touch /data/capx-out/reduced_repairs/queue.off    # stop after current
#
# A SECOND worker, VRR_FROM_END=1, takes lines from the END of the queue and
# writes each to the done list when it CLAIMS it, so the head worker never sees
# it. It only takes a line when at least 3 are pending, so it can never take
# the head worker's current line (the first pending one). It logs to
# queue_rev.log. A claimed line whose run died has no entry in
# results/reduced_verify.json -- re-append it to re-run.
#
#   VRR_FROM_END=1 bash scripts/vrr_queue.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
W=/data/capx-out/reduced_repairs
Q=$W/queue.txt; DONE=$W/queue.done; LOGF=$W/queue.log
REV="${VRR_FROM_END:-0}"
[ "$REV" = 1 ] && LOGF=$W/queue_rev.log
touch "$Q" "$DONE"
# OPENCAP_GPU BEFORE sourcing: env_server.sh derives CUDA_VISIBLE_DEVICES at
# source time and defaults to GPU 2 (RESUME.md trap 1). GPU 6 only.
export OPENCAP_GPU="${OPENCAP_GPU:-6}"
source scripts/env_server.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$OPENCAP_GPU"
export MUJOCO_EGL_DEVICE_ID="$OPENCAP_GPU"
echo "[queue] start $(date -u) on GPU $OPENCAP_GPU" >> "$LOGF"
while [ ! -e "$W/queue.off" ]; do
  if [ "$REV" = 1 ]; then
    pending=$(grep -v '^\s*$' "$Q" | grep -vxF -f "$DONE")
    if [ "$(printf '%s\n' "$pending" | grep -c .)" -lt 3 ]; then sleep 60; continue; fi
    line=$(printf '%s\n' "$pending" | tail -1)
    echo "$line" >> "$DONE"            # claim: the head worker skips it from now on
  else
    line=$(grep -v '^\s*$' "$Q" | grep -vxF -f "$DONE" | head -1)
  fi
  if [ -z "$line" ]; then sleep 30; continue; fi
  read -r task tier pid code rest <<<"$line"
  echo "=== $(date -u +%m-%d\ %H:%M) $line" >> "$LOGF"
  # shellcheck disable=SC2086
  python3 scripts/verify_reduced_repair.py "$task" --tier "$tier" --pid "$pid" \
      --code "$code" $rest >> "$LOGF" 2>&1
  [ "$REV" = 1 ] || echo "$line" >> "$DONE"
done
echo "[queue] stopped $(date -u)" >> "$LOGF"
