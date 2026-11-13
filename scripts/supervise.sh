#!/usr/bin/env bash
# Keep a collection running unattended: restart the services and the job when
# something external kills them, and STOP when the work is actually done.
#
#     bash scripts/supervise.sh bigrung          # supervise run_bigrung_pass.sh
#     bash scripts/supervise.sh grid "S3 S4"     # supervise run_tier_grid.sh
#     bash scripts/supervise.sh --status         # what it thinks, exit
#     bash scripts/supervise.sh --stop           # stop it AND the watchdog
#     bash scripts/supervise.sh --resume         # undo --stop
#
# Launch it detached, like the job itself:
#   OPENCAP_GPU=5 setsid nohup bash scripts/supervise.sh bigrung \
#     > /data/capx-logs/supervise.log 2>&1 < /dev/null &
#
# WHY THIS EXISTS
# ---------------
# The collection already survives a closed session -- it is started with
# `setsid nohup`, so it has no controlling terminal and is reparented to init.
# What it does NOT survive is the thing that actually keeps happening: an
# external sweep of GPU processes. Three times in ~54 hours (2026-09-29 19:05,
# 2026-09-30 07:51, 2026-10-01 13:16) the grid AND all its services were
# killed outright. Each time the run sat idle until a human noticed -- the
# second one cost 6h12m.
#
# A kill is cheap to detect and cheap to recover from. Nobody watching is the
# expensive part.
#
# WHAT IT DELIBERATELY WILL NOT DO
# --------------------------------
# An auto-restarter is a loaded gun pointed at a shared GPU, so this one is
# built to stop rather than to persist:
#
#   * it STOPS when the job completes -- it looks for the job's own
#     completion marker in the log, not a timeout, so a finished run is never
#     restarted into an infinite loop;
#   * it STOPS after MAX_RESTARTS (default 5). A job that dies repeatedly is
#     broken, not unlucky, and hammering it wastes a card somebody else wants;
#   * it STOPS if the restart happens suspiciously fast (< MIN_ALIVE), which
#     is the signature of a job that fails on startup rather than one that was
#     killed mid-flight;
#   * it NEVER starts a second collection beside a live one -- it checks for
#     the running job first, and pilot.sh's flock is the backstop;
#   * it WAITS rather than restarts when the card is full. A neighbour's job
#     holding the memory is a queue, not a failure, and the pass's own
#     memory guard refuses in four seconds -- fast enough that the
#     startup-failure test below would otherwise give up on a run that was
#     only ever waiting its turn;
#   * it touches nothing but its own job and services. It does not kill
#     anyone else's processes, and it does not free memory by force.
#
# It also writes every decision to its log with a timestamp, so the next
# person can see what it did rather than inferring it from cell counts.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

WHAT="${1:-bigrung}"
ARGS="${2:-S3 S4}"
GPU="${OPENCAP_GPU:-5}"
LOGDIR="${OPENCAP_LOGDIR:-/data/capx-logs}"
JOBLOG="${OPENCAP_JOBLOG:-/data/capx-out/phase9.log}"
STATE="$LOGDIR/supervise.state"
STOPFILE="$LOGDIR/supervise.stop"
INTERVAL="${OPENCAP_SUPERVISE_INTERVAL:-120}"      # seconds between checks
MAX_RESTARTS="${OPENCAP_SUPERVISE_MAX_RESTARTS:-5}"
MIN_ALIVE="${OPENCAP_SUPERVISE_MIN_ALIVE:-600}"    # a restart that dies faster
                                                 # than this is a broken job
MAX_WAIT="${OPENCAP_SUPERVISE_MAX_WAIT:-86400}"      # give up waiting for a busy
                                                 # card after this long (24h)
NEED_STREAK="${OPENCAP_SUPERVISE_HEADROOM_CHECKS:-3}" # consecutive clear samples
                                                 # before trusting the headroom

case "$WHAT" in
  bigrung) JOB_PAT='[r]un_bigrung'; DONE_PAT='BIGRUNG_S4_RC=|BIGRUNG_S3_RC=.*\n.*BIGRUNG_S4_RC='
           START_CMD="bash scripts/run_bigrung_pass.sh $ARGS"
           DONE_MARK='BIGRUNG_S4_RC=' ;;
  grid)    JOB_PAT='[r]un_tier_grid'
           START_CMD="bash scripts/run_tier_grid.sh $ARGS"
           DONE_MARK='TIER_S4_RC=' ;;
  --status)
           echo "[sup] state: $(cat "$STATE" 2>/dev/null || echo none)"
           pgrep -f '[r]un_bigrung|[r]un_tier_grid' >/dev/null 2>&1 \
             && echo "[sup] a collection IS running" || echo "[sup] no collection running"
           for p in 8114 8115 8116 8122; do
             (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null \
               && echo "[sup]   :$p up" || echo "[sup]   :$p DOWN"
           done
           if [ -e "$LOGDIR/watchdog.off" ]; then
             echo "[sup] watchdog is OFF: $(cat "$LOGDIR/watchdog.off")"
           elif crontab -l 2>/dev/null | grep -q supervisor_watchdog; then
             echo "[sup] watchdog is armed in cron"
           else
             echo "[sup] watchdog is NOT in cron -- a kill of the supervisor"
             echo "[sup] would go unnoticed. Install it with:"
             echo "[sup]   bash scripts/install_watchdog_cron.sh"
           fi
           exit 0 ;;
  --stop)  # Stop BOTH layers. The cron watchdog would otherwise put a new
           # supervisor back within 5 minutes, and an operator fighting their
           # own tooling is worse than no tooling. watchdog.off is never
           # deleted by the watchdog -- only by --resume, or by hand.
           : > "$STOPFILE"
           echo "stopped by hand $(date -u)" > "$LOGDIR/watchdog.off"
           echo "[sup] wrote $STOPFILE; supervisor exits within ${INTERVAL}s"
           echo "[sup] wrote $LOGDIR/watchdog.off; cron will NOT restart it"
           echo "[sup] undo with: bash scripts/supervise.sh --resume"
           exit 0 ;;
  --resume)
           rm -f "$LOGDIR/watchdog.off" "$STOPFILE"
           echo "[sup] cleared the stop files; cron starts a supervisor within 5 min"
           echo "[sup] (or start one now: OPENCAP_GPU=${GPU} setsid nohup bash scripts/supervise.sh bigrung >> $LOGDIR/supervise.log 2>&1 < /dev/null &)"
           exit 0 ;;
  *)       echo "[sup] unknown target '$WHAT' (use bigrung|grid|--status|--stop|--resume)"; exit 1 ;;
esac

say() { echo "[sup $(date -u +'%m-%d %H:%M:%S')] $*"; }

rm -f "$STOPFILE"
restarts=0
waited=0
ok_streak=0
say "supervising '$WHAT' on GPU $GPU; interval ${INTERVAL}s, max $MAX_RESTARTS restarts"
say "stop me with: bash scripts/supervise.sh --stop"

while :; do
  if [ -e "$STOPFILE" ]; then say "stop file present -- exiting"; rm -f "$STOPFILE"; exit 0; fi

  # 1. Has the job finished for real? Its own marker, not a guess.
  if tail -400 "$JOBLOG" 2>/dev/null | grep -q "$DONE_MARK"; then
    say "found '$DONE_MARK' in the job log -- work is COMPLETE, exiting"
    echo "complete $(date -u)" > "$STATE"
    exit 0
  fi

  # 2. Is it running? Then there is nothing to do.
  if pgrep -f "$JOB_PAT" >/dev/null 2>&1; then
    echo "alive $(date -u +%s)" > "$STATE"
    sleep "$INTERVAL"; continue
  fi

  # 3. Is the card simply busy? Then there is nothing to restart INTO, and
  #    this is not a death worth counting.
  #
  #    run_bigrung_pass.sh refuses in about four seconds when the pool will
  #    not fit, which the MIN_ALIVE test below reads as a startup failure --
  #    so the supervisor gives up and exits, on a run that is merely queued.
  #    That is what happened on 2026-10-02: a neighbour's two inference jobs
  #    took 81 GB of GPU 5, the 03:45 relaunch was refused, and the work sat
  #    idle. A neighbour holding the card is a QUEUE, not a broken job.
  #
  #    So: wait, do not spend a restart, and do not touch anyone's memory.
  if [ "$WHAT" = bigrung ]; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    need=$(awk -v t="${total:-0}" -v u="${OPENCAP_BIGRUNG_UTIL:-0.60}" 'BEGIN{printf "%d", t*u + 2048}')
    if [ -n "${free:-}" ] && [ "$free" -lt "$need" ] 2>/dev/null; then
      ok_streak=0
      if [ "$waited" -ge "$MAX_WAIT" ]; then
        say "the card has been too full for $((waited / 3600))h (${free} MiB free,"
        say "need ${need}). Not waiting any longer -- a person should decide"
        say "whether to queue behind the neighbour or drop OPENCAP_BIGRUNG_UTIL."
        echo "gave-up-waiting $(date -u)" > "$STATE"
        exit 1
      fi
      # Non-numeric second field on purpose: the MIN_ALIVE parse below reads
      # field 2 as an epoch, and a wait must not look like a live job.
      echo "waiting-for-card $(date -u +%Y-%m-%dT%H:%M:%SZ) free=${free}MiB need=${need}MiB" > "$STATE"
      # One line every ~30 min, not every interval -- a day of waiting should
      # not bury the decisions that matter in the log.
      if [ "$waited" -eq 0 ] || [ $((waited % 1800)) -lt "$INTERVAL" ]; then
        say "card busy: ${free} MiB free, need ${need}. Waited $((waited / 60))m. Holding."
      fi
      # Counted AFTER the report, so the first line says 0m rather than
      # claiming a wait that has not happened yet.
      waited=$((waited + INTERVAL))
      sleep "$INTERVAL"; continue
    fi
    # There is room -- but is it STABLE room? The neighbour's jobs allocate
    # and release in bursts: GPU 5 read 34, 57 and 36 GB free inside five
    # minutes on 2026-10-02. Launching into a transient peak is how you earn
    # vLLM's "Error in memory profiling. Initial free memory X, current free
    # Y" -- the engine dies, the rung collects nothing, and the log looks
    # like a model failure rather than a neighbour.
    ok_streak=$((ok_streak + 1))
    if [ "$ok_streak" -lt "$NEED_STREAK" ]; then
      say "room for ${free} MiB, holding ${ok_streak}/${NEED_STREAK} checks -- confirming it is not a dip"
      echo "confirming-headroom $(date -u +%Y-%m-%dT%H:%M:%SZ) ${ok_streak}/${NEED_STREAK} free=${free}MiB" > "$STATE"
      waited=$((waited + INTERVAL))
      sleep "$INTERVAL"; continue
    fi
    if [ "$waited" -gt 0 ]; then
      say "card free for ${NEED_STREAK} consecutive checks (${free} MiB) after $((waited / 60))m -- proceeding"
      waited=0
    fi
  fi

  # 3. It is not running and not finished -- so it died.
  last=$(awk '{print $2}' "$STATE" 2>/dev/null)
  now=$(date -u +%s)
  if [ -n "${last:-}" ] && [ "$last" -gt 0 ] 2>/dev/null; then
    alive=$((now - last))
    if [ "$alive" -lt "$MIN_ALIVE" ]; then
      say "job died ${alive}s after the last check (< ${MIN_ALIVE}s): that is a"
      say "startup failure, not an external kill. NOT restarting -- a human"
      say "should look. Exiting."
      echo "startup-failure $(date -u)" > "$STATE"
      exit 1
    fi
  fi

  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    say "already restarted $MAX_RESTARTS times -- refusing to keep going."
    say "Something is killing this repeatedly and it needs a person."
    echo "gave-up $(date -u)" > "$STATE"
    exit 1
  fi

  say "job is gone and the work is not finished -- restart $restarts/$MAX_RESTARTS"

  # Services first, and refuse to start the job if they do not come up:
  # launch.py warns and runs the cell anyway against a dead service, which
  # scores an environment failure as a model defect.
  if ! OPENCAP_GPU="$GPU" bash scripts/start_vision_servers.sh; then
    say "vision servers would not start -- NOT starting the job"
    sleep "$INTERVAL"; continue
  fi
  if ! OPENCAP_GPU="$GPU" bash scripts/start_molmo_server.sh; then
    say "molmo would not start -- NOT starting the job (25 cells were lost"
    say "that way before :8122 had a launcher)"
    sleep "$INTERVAL"; continue
  fi

  say "services up; relaunching: $START_CMD"
  OPENCAP_GPU="$GPU" OPENCAP_GPU_UTIL="${OPENCAP_GPU_UTIL:-0.35}" \
    setsid nohup bash -c "$START_CMD >> '$JOBLOG' 2>&1" < /dev/null \
    > "$LOGDIR/supervise_restart_$(date -u +%m%d_%H%M).log" 2>&1 &
  echo "alive $(date -u +%s)" > "$STATE"
  sleep "$INTERVAL"
done
