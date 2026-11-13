#!/usr/bin/env bash
# Arm the watchdog in cron, idempotently.
#
#     bash scripts/install_watchdog_cron.sh          # install or refresh
#     bash scripts/install_watchdog_cron.sh --remove # take it out again
#
# Idempotent because this will be run more than once, by someone who does not
# remember whether they already ran it, and a crontab with the same job three
# times starts three supervisors.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd -P)"
LINE="*/5 * * * * /usr/bin/bash $REPO/scripts/supervisor_watchdog.sh"
TAG="# capx supervisor watchdog -- see scripts/supervisor_watchdog.sh"

current="$(crontab -l 2>/dev/null)"
cleaned="$(printf '%s\n' "$current" | grep -v 'supervisor_watchdog' | grep -v "^${TAG}$")"

if [ "${1:-}" = "--remove" ]; then
  printf '%s\n' "$cleaned" | grep -v '^$' | crontab - 2>/dev/null || crontab -r 2>/dev/null
  echo "[cron] watchdog removed. The supervisor keeps running; nothing will"
  echo "[cron] put it back if it is killed."
  exit 0
fi

{ printf '%s\n' "$cleaned" | grep -v '^$'; echo "$TAG"; echo "$LINE"; } | crontab -
echo "[cron] installed:"
crontab -l | grep -A1 'capx supervisor watchdog'
echo
echo "[cron] It checks every 5 minutes and starts a supervisor only if none"
echo "[cron] is running and the job has not printed its completion marker."
echo "[cron] Stop everything with: bash scripts/supervise.sh --stop"
