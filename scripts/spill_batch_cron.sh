#!/usr/bin/env bash
# Restart the spill_wipe batch if something killed it. Every 20 min from cron.
#
# The batch is RESUMABLE -- spill_wipe_batch.py skips any pid already at
# ceiling -- so re-invoking it costs nothing and an external GPU sweep costs
# minutes instead of hours. Same reasoning as supervisor_watchdog.sh: a
# user-space daemon cannot survive a sweep of user-space daemons, cron can.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="${PATH:+$PATH:}/usr/bin:/bin:/usr/local/bin"
LOG=/data/capx-logs/spill_batch.log
OFF=/data/capx-logs/spill_batch.off
[ -e "$OFF" ] && exit 0
grep -qa 'SPILL_BATCH_DONE' "$LOG" 2>/dev/null && { echo "done $(date -u)" > "$OFF"; exit 0; }
pgrep -f '[r]un_spill_batch' >/dev/null 2>&1 && exit 0
echo "[cron $(date -u +'%m-%d %H:%M')] batch not running and not finished -- restarting" >> "$LOG"
OPENCAP_GPU=6 setsid nohup bash scripts/run_spill_batch.sh >> "$LOG" 2>&1 < /dev/null &
