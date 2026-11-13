#!/usr/bin/env bash
# A/B: how much of the measured floor is a missing import, not robot control?
#
# Same models, same task, same seeds, same greedy decoding. The ONLY difference
# is whether numpy is pre-imported into the execution namespace. The gap is the
# share of "cannot control a robot" that is really "did not follow the prompt's
# import instruction".
#
# Chosen cell: cube_lifting S1, where qwen3-4b wrote a correct program, grasped
# the cube (reward 0.525) and scored 0/30 on `NameError: name 'np'`.
set -uo pipefail
MODELS="${AB_MODELS:-qwen3-4b qwen3-8b qwen3-14b}"
TASK="${AB_TASK:-cube_lifting}"
TIER="${AB_TIER:-S1}"

for arm in off on; do
  if [ "$arm" = on ]; then export OPENCAP_PREIMPORT_NUMPY=1; else unset OPENCAP_PREIMPORT_NUMPY; fi
  echo "======== numpy preimport: $arm ========"
  OPENCAP_STORE="local:/content/ab_$arm" \
  PILOT_MODELS="$MODELS" PILOT_TIERS="$TIER" PILOT_TASKS="$TASK" \
    bash /content/capx-scaling/scripts/pilot.sh
done
echo AB_RC=$?
