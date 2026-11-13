#!/usr/bin/env bash
# Progress of the detached prompt-fix arm. Read-only -- safe to run any time,
# from any shell, while the sweep is running.
LOG="${1:-/data/capx-logs/arm.log}"
PFIX="${OPENCAP_SCRATCH:-/data/capx-out/pfix}"

done_n=$(grep -cE '^\[ok\]' "$LOG" 2>/dev/null)
fail_n=$(grep -cE '^\[fail\]' "$LOG" 2>/dev/null)
echo "prompt-fix arm: $done_n of 34 cells committed, $fail_n failed"

if pgrep -f run_promptfix_arm.sh > /dev/null; then
  echo "status : RUNNING (pid $(pgrep -f run_promptfix_arm.sh | head -1), detached; ppid 1)"
  echo "model  : $(grep -E '^#{4}' "$LOG" | tail -1 | tr -d '# ')"
  for d in "$PFIX"/*/*__S1__*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name 'trial_*' -type d 2>/dev/null | wc -l)
    [ "$n" -lt 30 ] && echo "cell   : $(basename "$d") $n/30 trials"
  done
elif grep -q ARM_RC "$LOG" 2>/dev/null; then
  echo "status : FINISHED ($(grep ARM_RC "$LOG" | tail -1))"
else
  echo "status : NOT RUNNING and no ARM_RC -- it died; check $LOG"
fi

echo
echo "committed cells (model / task / score / wall):"
grep -E '^\[ok\]' "$LOG" | sed -E "s/.*model='([^']+)'.*task='([^']+)'\): ([0-9]+)\/30.* in ([0-9.]+)s.*/\1 \2 \3 \4/" \
  | awk '{printf "  %-20s %-14s %2s/30  %6.0fs\n",$1,$2,$3,$4}'
