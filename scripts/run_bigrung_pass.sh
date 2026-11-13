#!/usr/bin/env bash
# The rungs the tier grid could not start: 27B and 32B at S3/S4, on a pool
# big enough to hold them.
#
#     bash scripts/run_bigrung_pass.sh              # S3 S4, pool 0.60
#     bash scripts/run_bigrung_pass.sh S3           # one tier
#     OPENCAP_BIGRUNG_UTIL=0.70 bash scripts/run_bigrung_pass.sh
#
# WHY THIS IS NOT OPTIONAL TIDYING
# -------------------------------
# It is the difference between an honest headline row and a misleading one.
# With the tier grid's fixed 0.35 pool, every 27B/32B engine failed to start,
# so the S3 corpus has NO model above 14B in it. The rate table still prints a
# ">=12B" row for both tiers:
#
#     band      S1                       S3
#     >=12B     58.9% [55.3,62.4] n=720  0.0% [0.0,2.5] n=150
#
# but S1's row is 24 cells across gemma3-12b, qwen3-14b, gemma3-27b,
# qwen35-27b, qwen3-32b and qwen3vl-32b, while S3's is 5 cells of gemma3-12b
# and qwen3-14b alone. The band label means something different in each
# column, and it is absent from S3 exactly where S1 scores highest. Reported
# as-is, the abstraction gap is overstated by construction.
#
# So this pass is not "24 more cells for completeness". It is the cells that
# decide whether the top row can be published at all.
#
# WHY A SEPARATE PASS AND NOT A BIGGER POOL EVERYWHERE
# ---------------------------------------------------
# A 0.60 pool starves the perception services (SAM3 ~7 GB, ContactGraspNet
# ~11 GB) and leaves nothing for a neighbour. The grid's 0.35 is right for the
# 17 rungs that fit it; the remaining 4 need a card mostly to themselves. Two
# passes at two pools collect 21 models; one pass at any single pool does not.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# CAPTURE BEFORE SOURCING. env_server.sh sets OPENCAP_GPU=2 unconditionally, so
# reading it afterwards cannot tell "the caller asked for 2" from "the caller
# said nothing and env_server defaulted". Reading it after is how the first
# version of this script announced gpu=2 while the grid, the perception
# services and the free card were all on 5.
GPU="${OPENCAP_GPU:-5}"

source scripts/env_server.sh
export OPENCAP_GPU="$GPU"
# RE-DERIVE, do not just re-export. env_server.sh computes CUDA_VISIBLE_DEVICES
# and MUJOCO_EGL_DEVICE_ID from OPENCAP_GPU at SOURCE time, so setting OPENCAP_GPU
# afterwards moves the NVML reads (nvidia-smi -i) without moving the work. The
# card you meant is reported; the card you did not gets the job, and nothing in
# the log looks wrong. This bit a verification run on 2026-10-02.
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_EGL_DEVICE_ID="$GPU"

TIERS="${*:-S3 S4}"
UTIL="${OPENCAP_BIGRUNG_UTIL:-0.60}"
STORE="${OPENCAP_STORE:-hf:savoji/capx-ab-on-seeded}"

#: The rungs that could not start at 0.35. Not a guess -- each one failed in
#: the 2026-09-29 walk and the failure is in phase9.log.
MODELS="${OPENCAP_BIGRUNG_MODELS:-gemma3-27b qwen35-27b qwen3-32b qwen3vl-32b}"

echo "############ big-rung pass: $MODELS ############"
echo "[bigrung] tiers=$TIERS pool=$UTIL gpu=$GPU -> $STORE"

# --- GUARD 1: not beside the grid. --------------------------------------
# run_sweep binds a fixed :8110 and pilot.sh holds an flock, so a second
# instance does not queue -- it fails, slowly, one engine at a time, and the
# failures look exactly like the memory failures this pass exists to fix.
if pgrep -f '[r]un_tier_grid' >/dev/null 2>&1; then
  echo "[bigrung] REFUSING: run_tier_grid.sh is running. Same card, same"
  echo "[bigrung] :8110, same flock. Wait for it to finish -- completed cells"
  echo "[bigrung] skip, so nothing is lost by waiting."
  exit 1
fi
if (exec 3<>/dev/tcp/127.0.0.1/8110) 2>/dev/null; then
  echo "[bigrung] REFUSING: something already serves :8110."
  exit 1
fi

# --- GUARD 2: the card must actually hold the pool. ---------------------
# Failing here costs four seconds. Failing in vLLM costs ~90s per engine per
# model per tier, and the run 'completes' having collected nothing -- the
# exact shape of the 2026-09-28 auto-pool loss.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
NEED_MIB=$(awk -v t="$TOTAL_MIB" -v u="$UTIL" 'BEGIN{printf "%d", t*u + 2048}')
echo "[bigrung] GPU $GPU: ${FREE_MIB} MiB free of ${TOTAL_MIB}; pool $UTIL needs ~${NEED_MIB} MiB"
if [ "$FREE_MIB" -lt "$NEED_MIB" ]; then
  echo "[bigrung] REFUSING: not enough free memory for a $UTIL pool."
  echo "[bigrung] Either wait for the card, or lower OPENCAP_BIGRUNG_UTIL --"
  echo "[bigrung] but below ~0.55 the 32B rungs will not start and this pass"
  echo "[bigrung] collects nothing while appearing to run."
  exit 1
fi

# --- GUARD 3: the tiers are perception-backed. --------------------------
# S3/S4 bind segment_sam3_*, plan_grasp and solve_ik. launch.py warns and runs
# the cell anyway when a service is missing, scoring a dead server as a model
# failure -- 30 plausible zeros with no error.
bash scripts/start_vision_servers.sh --check || {
  echo "[bigrung] REFUSING: start them first with"
  echo "[bigrung]   OPENCAP_GPU=$GPU bash scripts/start_vision_servers.sh"
  exit 1
}
# And molmo, the FOURTH service -- added 2026-10-01, because it was missing
# from this check while :8122 had no launcher at all. It now does, and a
# reduced-API cell whose program calls point_prompt_molmo dies without it:
# that cost 25 cells and confounded the whole S3-vs-S4 comparison.
bash scripts/start_molmo_server.sh --check || {
  echo "[bigrung] REFUSING: molmo is down. Start it with"
  echo "[bigrung]   OPENCAP_GPU=$GPU bash scripts/start_molmo_server.sh"
  echo "[bigrung] Without it, every program that points at an object fails"
  echo "[bigrung] and the cell is refused -- 25 cells lost that way already."
  exit 1
}

# --- GUARD 4: the same simulator precondition the grid runs. ------------
# Artifact #13 was the seeding patch reaching one simulator in four. Do not
# skip this to save four minutes; a big rung drawing random scenes is just as
# wrong as a small one, and more expensive to recollect.
python3 scripts/verify_tier_simulators.py "${CAPX_DIR:-/data/cap-x}" \
  --tiers "$(echo $TIERS | tr ' ' ',')" --strict || {
  echo "[bigrung] REFUSING: a config binds a simulator the seeding patch does"
  echo "[bigrung] not cover, or a binding could not be resolved."
  exit 1
}

for T in $TIERS; do
  echo
  echo "############ big rungs, tier $T ############"
  date -u
  # A FIXED pool, never auto: `auto` reads free memory, quantises to a 0.10
  # ladder and the engine starts seconds later, so a neighbour moving in
  # either direction breaks it. Growing -> vLLM refuses the claim. Shrinking
  # -> "Error in memory profiling. Initial free memory X, current free Y".
  # Both cost the whole rung.
  OPENCAP_GPU="$GPU" \
  OPENCAP_GPU_UTIL="$UTIL" \
  OPENCAP_STORE="$STORE" \
  OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/tier_$T" \
  PILOT_MODELS="$MODELS" \
  PILOT_TIERS="$T" \
    bash scripts/run_arm.sh baseline
  echo "BIGRUNG_${T}_RC=$?"
done

echo
echo "############ rate table with the top of the ladder in it ############"
python3 scripts/tier_report.py $TIERS

echo
echo "[bigrung] Check the >=12B row now covers the same models in both"
echo "[bigrung] columns before quoting the abstraction gap. If a rung still"
echo "[bigrung] did not start, say which -- a band whose membership differs"
echo "[bigrung] between tiers is not a comparison."
