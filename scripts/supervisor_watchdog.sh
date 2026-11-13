#!/usr/bin/env bash
# Put the supervisor back when something kills IT.
#
# Installed in cron, every 5 minutes:
#   */5 * * * * /usr/bin/bash /data/code/playground/capx-scaling/scripts/supervisor_watchdog.sh
#
# WHY A THIRD LAYER
# -----------------
# There are three things that can die, and each layer only covers the one
# below it:
#
#   cron       -> survives everything short of a reboot
#    watchdog  -> puts the supervisor back
#     supervisor -> puts the collection back
#      collection -> the actual work
#
# `setsid nohup` already makes the collection survive a closed session: PPID
# 1, no controlling terminal, stdin on /dev/null. That was never the problem.
# The problem is that the external GPU-process sweep kills by user, so it
# takes the supervisor along with everything it was supervising -- four times
# in ~77 hours (2026-09-29 19:05, 09-30 07:51, 10-01 13:16, 10-01 23:40). The
# 10-01 23:40 kill cost four hours precisely because the supervisor died with
# the job and nobody was left to notice.
#
# A user-space daemon cannot protect against a sweep of user-space daemons.
# cron is launched by crond, in its own session, and is not a GPU process.
#
# HOW IT REFUSES TO BE A NUISANCE
# -------------------------------
#   * it STOPS for good once the job prints its own completion marker --
#     otherwise it would respawn a supervisor every 5 minutes forever, each
#     one exiting immediately, writing log noise until someone noticed;
#   * it OBEYS an explicit human stop. `supervise.sh --stop` writes
#     watchdog.off, which this script never deletes. Stopping the work has to
#     mean it stays stopped, or the operator is fighting their own tooling;
#   * it starts NOTHING if a supervisor is already up;
#   * it touches no GPU memory and no process it did not start.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# APPENDED, not prepended. cron's PATH is minimal, so the system dirs have to
# be here or nvidia-smi and pgrep do not resolve -- but prepending them makes
# the script impossible to drive against stubs, and untested control flow is
# how the supervisor came to treat a queued job as a broken one. The ${PATH:+}
# guard keeps an empty PATH from leaving a bare "" element, which means CWD.
export PATH="${PATH:+$PATH:}/usr/bin:/bin:/usr/local/bin"

LOGDIR="${OPENCAP_LOGDIR:-/data/capx-logs}"
JOBLOG="${OPENCAP_JOBLOG:-/data/capx-out/phase9.log}"
WDLOG="$LOGDIR/watchdog.log"
OFF="$LOGDIR/watchdog.off"
WHAT="${OPENCAP_WATCHDOG_TARGET:-bigrung}"
ARGS="${OPENCAP_WATCHDOG_ARGS:-S3 S4}"
GPU="${OPENCAP_GPU:-5}"

say() { echo "[wd $(date -u +'%m-%d %H:%M:%S')] $*" >> "$WDLOG"; }

[ -e "$OFF" ] && exit 0

case "$WHAT" in
  bigrung) DONE_MARK='BIGRUNG_S4_RC=' ;;
  grid)    DONE_MARK='TIER_S4_RC='    ;;
  *)       say "unknown target '$WHAT' -- doing nothing"; exit 1 ;;
esac

# Finished for real? Then stay out of the way permanently, and say so once.
if tail -400 "$JOBLOG" 2>/dev/null | grep -q "$DONE_MARK"; then
  say "job log shows '$DONE_MARK' -- work is COMPLETE. Disabling myself."
  echo "completed $(date -u)" > "$OFF"
  exit 0
fi

# Already supervised? Nothing to do, and say nothing -- this is the normal
# case 99% of the time and it must not fill the log.
pgrep -f '[s]upervise\.sh '"$WHAT" >/dev/null 2>&1 && exit 0

say "no supervisor running and the work is not finished -- starting one"
OPENCAP_GPU="$GPU" setsid nohup bash scripts/supervise.sh "$WHAT" "$ARGS" \
  >> "$LOGDIR/supervise.log" 2>&1 < /dev/null &
sleep 3
if pgrep -f '[s]upervise\.sh '"$WHAT" >/dev/null 2>&1; then
  say "supervisor is up"
else
  say "supervisor did NOT come up -- a person should look at supervise.log"
fi
