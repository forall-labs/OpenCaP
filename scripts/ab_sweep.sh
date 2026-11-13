#!/usr/bin/env bash
# Corpus-wide A/B: how much of measured failure is instruction-following
# rather than robot control, and how does that split move with scale?
#
# Each cell runs twice -- identical models, seeds, tiers, tasks and greedy
# decoding -- differing only in whether numpy is pre-imported. Per cell we get:
#   success(off), success(on)          -> the artifact share
#   failure-mode histogram in each arm -> WHICH failures the import explains
#
# Scope is chosen, not exhaustive:
#   models  set by AB_MODELS. The first pass ran 4B/8B/14B; the ladder was
#           then extended down to 0.6B/1.7B to test whether the scaffold's
#           share of measured failure grows as models shrink.
#   tiers   set by AB_TIERS, default S1+S4 -- the abstraction endpoints. S3
#           is skipped because S3 and S4 were identical (in-context examples
#           bought nothing), so it adds cost without contrast.
#   tasks   cube_lifting, cube_stack, spill_wipe -- excludes nut_assembly
#           (oracle 0.278, 31% noise floor) and two_arm_* (33% noise floor),
#           where n=30 cannot resolve anything.
#
# Arms write to SEPARATE HF repos so a reaped runtime loses nothing and the
# two conditions can never be silently mixed.
set -uo pipefail
MODELS="${AB_MODELS:-qwen3-4b qwen3-8b qwen3-14b}"
TIERS="${AB_TIERS:-S1 S4}"
TASKS="${AB_TASKS:-cube_lifting cube_stack spill_wipe}"

for arm in off on; do
  if [ "$arm" = on ]; then export OPENCAP_PREIMPORT_NUMPY=1; else unset OPENCAP_PREIMPORT_NUMPY; fi
  echo "############################ ARM: numpy=$arm ############################"
  # Separate scratch per arm: sharing one output dir makes the second arm's
  # parse read a mixture of both conditions (caught by cross-check, but only
  # after one cell had already committed).
  OPENCAP_SCRATCH="/content/outputs_$arm" \
  OPENCAP_STORE="hf:savoji/capx-ab-$arm" \
  PILOT_MODELS="$MODELS" PILOT_TIERS="$TIERS" PILOT_TASKS="$TASKS" \
    bash /content/capx-scaling/scripts/pilot.sh
  echo "[arm $arm done rc=$?]"
done
echo AB_SWEEP_RC=0
