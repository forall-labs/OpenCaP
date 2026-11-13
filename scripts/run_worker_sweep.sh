#!/usr/bin/env bash
# Is the cube_stack scene-13 noise floor ordering, or contention?
#
#     bash scripts/run_worker_sweep.sh              # the missing counts (2)
#     bash scripts/run_worker_sweep.sh 2 8          # specific counts
#
# THE QUESTION
# ------------
# Phase 10 showed a varying trial fails in 11 of 16 draws at workers=4 and 0 of
# 16 at workers=1 (Fisher p=7e-5), pooled over a failing program and its
# verified repair. The repair is AT CEILING at workers=1 -- 30/30 in all eight
# draws -- and loses scene 13 in half its workers=4 draws. So this project's
# entire measured noise floor is worker-count dependent, and "the noise floor
# is one scene" describes which scene moves, not a property of the scene.
#
# What that does NOT say is WHY. workers=1 removes three things at once: trial
# interleaving, CPU contention, and timing overlap. (It does not remove the IK
# server -- the servers were already down when Phase 10 ran and every record
# returned ok, so cube_stack S1 never contacts :8116.)
#
# A middle worker count separates them:
#
#     workers=2 as high as workers=4   BINARY   -> interleaving / ordering
#     workers=2 between 1 and 4        MONOTONE -> contention or load
#     workers=2 shows nothing          THRESHOLD-> load, above 2 workers
#
# 8 draws x 2 programs per count. ~15 minutes.
#
# WHAT IT NEEDS: a GPU, because MuJoCo renders through EGL. NO vLLM, NO model
# weights, NO vision servers, NO IK server. This is the cheapest kind of job
# this project has.
#
# WHAT IT NEEDS THAT IS EASY TO GET WRONG: the SAME card under the SAME load as
# the existing draws. Contention is the quantity under test, so a differently
# occupied card is not a comparison, it is a different experiment. This script
# records the card's occupancy before and after every count and prints it with
# the result, so a run on a busy card is still usable as long as it is read
# with that in mind.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

PID="${OPENCAP_SWEEP_PID:-81ee85d0f933}"
TASK="${OPENCAP_SWEEP_TASK:-cube_stack}"
DRAWS="${OPENCAP_SWEEP_DRAWS:-8}"
OUT="${OPENCAP_SWEEP_OUT:-results/scene_probe_alt.json}"
COUNTS="${*:-2}"
LOG="${OPENCAP_LOGDIR:-/tmp}/worker_sweep_$(date -u +%Y%m%dT%H%M%SZ).log"

card() {
  # Who else is on our card. nvidia-smi ignores CUDA_VISIBLE_DEVICES, so query
  # the physical index explicitly -- the mistake that put work on a
  # neighbour's GPU while the logs said otherwise.
  nvidia-smi -i "${OPENCAP_GPU}" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}

{
  echo "[sweep] pid=$PID task=$TASK draws=$DRAWS counts='$COUNTS'"
  echo "[sweep] out=$OUT gpu=$OPENCAP_GPU"
  echo "[sweep] LLM-free: no engine, no weights, no vision servers, no IK server."
  echo
  echo "[sweep] ---- card occupancy BEFORE ----"
  card
  echo

  RC=0
  for W in $COUNTS; do
    echo "############ workers=$W ############"
    date -u
    echo "[sweep] card before this count:"; card
    # Resume is keyed on (pid, version, draw, workers), so counts already in
    # the file are skipped and this appends.
    python3 scripts/run_scene_probe.py \
      --pids "$PID" --task "$TASK" --draws "$DRAWS" --workers "$W" --out "$OUT"
    r=$?; [ $r -ne 0 ] && RC=$r
    echo "WORKERS_${W}_RC=$r"
    echo "[sweep] card after this count:"; card
    echo
  done

  echo "[sweep] ---- card occupancy AFTER ----"
  card
  echo
  echo "############ per-draw failures ############"
  python3 scripts/alternation_report.py "$OUT"
  echo
  echo "############ the verdict, re-derived ############"
  # The register reads the worker counts out of the file rather than a list in
  # code -- the first version hardcoded (4, 1) and would have ignored this run.
  python3 scripts/claims_register.py 2>/dev/null \
    | grep -A 2 "noise floor is a property" || true
  echo
  echo "[sweep] commit $OUT. Record in the STATUS entry: which card, who else"
  echo "[sweep] was on it, and whether the project's servers were up."
  echo "SWEEP_RC=$RC"
} 2>&1 | tee "$LOG"
echo "[sweep] log: $LOG"
