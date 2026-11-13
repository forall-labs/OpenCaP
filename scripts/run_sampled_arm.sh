#!/usr/bin/env bash
# The sampled nut_assembly arm: error bars on a corpus that has none.
#
#     bash scripts/run_sampled_arm.sh              # k=1..5, all 21 models
#     bash scripts/run_sampled_arm.sh 3            # just draw 3 (resume/extend)
#
# WHY
# ---
# Greedy decoding is deliberate: a robot policy has to be trustworthy, and in
# most deployments that means deterministic, so greedy IS the deployment
# condition. The cost is n=30 for the ENVIRONMENT and n=1 for the MODEL. Every
# cross-model claim in this project -- the scale table, defects per program,
# distance by band -- rests on one draw per cell. The >=12B row is 13 programs
# with one draw each. Eight headline findings have inverted, and thin data is
# the common thread.
#
# This does not replace the baseline. It is a separate store, a separate
# condition, and a separate claim: "what is the spread of this model's
# behaviour", against the baseline's "what does this deployed policy do".
#
# SCOPED TO nut_assembly, deliberately:
#   * it is the one task where the panel sits at the floor (3 successes in 630)
#     and the one where success rate and code quality come apart -- lowest edit
#     distance of any task;
#   * it is the only task where the FRAME class appears at all, so 5x the
#     programs takes that hypothesis from n=4 toward n~20;
#   * 21 cells per draw instead of 84 makes it ~11h instead of ~45h.
#
# 21 models x 5 draws x 30 scenes = 105 cells.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

DRAWS="${*:-1 2 3 4 5}"
TEMP="${OPENCAP_TEMPERATURE:-0.7}"
# A4 (2026-10-14) needs this arm at another task and tier: 8 of 21 models
# flip whether they attempt a coordinate transform between draws, so every
# PER-MODEL claim in the paper rests on one draw. Task and tier are now
# arguments rather than facts of the script, and the store name carries them
# so two scopes cannot land in one place -- a cell path is
# (wave, model, tier, task) and the task is already in it, but the TIER is
# too, and a nut_assembly S1 draw and a cube_stack S3 draw of the same model
# are different experiments that must not share a store name by accident.
TASK="${OPENCAP_SAMPLED_TASK:-nut_assembly}"
TIER="${OPENCAP_SAMPLED_TIER:-S1}"
# `tr`, not ${TIER,,}: that is bash 4+, and on bash 3.2 it does not fail --
# it expands to nothing, so a cube_stack/S3 run would have written into the
# nut_assembly store and the two scopes would have overwritten each other in
# place. Portable, and checked below.
SCOPE=""
if [ "$TASK" != "nut_assembly" ] || [ "$TIER" != "S1" ]; then
  SCOPE="-${TASK}-$(printf '%s' "$TIER" | tr '[:upper:]' '[:lower:]')"
fi
case "$SCOPE" in
  *'${'*|*',,'*) echo "[sampled] REFUSING: store scope did not expand ($SCOPE)."
                 echo "[sampled] Your shell dropped a substitution; the run would"
                 echo "[sampled] have written into the default store."; exit 1 ;;
esac

# --- PRECONDITION: whatever this tier declares, must be listening ----------
# This arm had NO server check, and A4's original run (nut_assembly/S1) did not
# need one. A8 runs cube_lifting at S3, which declares sam3 :8114, graspnet
# :8115 and pyroki :8116 -- so without this, 105 draws could be collected
# against a dead stack and every perception failure would score as a model
# defect. That is exactly how A3 lost 7h20m on 2026-10-15 and committed a
# false zero.
#
# verify_servers.py reads the tier's OWN config and prints "no api_servers
# declared -- nothing to check" when there is nothing, so this is a no-op for
# S1 and the only protection for S3. It takes <task> <tier>: called bare it
# exits on an argparse error, which reads as "servers down" -- that bug kept A2
# from ever starting.
echo "############ precondition: declared api_servers are listening ############"
python3 scripts/verify_servers.py "$TASK" "$TIER" \
    --root "${CAPX_DIR:-/data/cap-x}"
if [ $? -ne 0 ]; then
  echo "[sampled] REFUSING: $TASK/$TIER declares an api_server that is not up."
  echo "[sampled]   bash scripts/start_vision_servers.sh"
  echo "[sampled]   bash scripts/start_molmo_server.sh"
  echo "[sampled] This arm measures DRAW-TO-DRAW variation, so a stack that is"
  echo "[sampled] up for some draws and down for others is the one confound it"
  echo "[sampled] cannot survive -- the spread would be the servers, not the"
  echo "[sampled] decode, and that is the number A2 is about to be judged on."
  exit 1
fi

# The store name MUST contain '-sampled': opencap/store.py refuses to write a
# sampled cell into a greedy store and a greedy cell into a sampled one,
# because the two share a shard path and one would replace the other in place.
# An unset OPENCAP_TEMPERATURE in one shell of a multi-shell run is exactly how
# a mixed corpus would happen, and that refusal is what catches it.
for K in $DRAWS; do
  STORE="hf:savoji/capx-nut-sampled${SCOPE}-k${K}"
  echo
  echo "############ sampled draw k=$K  $TASK/$TIER  temp=$TEMP  -> $STORE ############"
  date -u
  OPENCAP_TEMPERATURE="$TEMP" \
  OPENCAP_SAMPLE_INDEX="$K" \
  OPENCAP_STORE="$STORE" \
  OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/sampled_k${K}" \
  PILOT_TASKS="$TASK" \
  PILOT_TIERS="$TIER" \
    bash scripts/run_arm.sh baseline
  echo "DRAW_${K}_RC=$?"
done

echo
echo "############ sampled report ############"
python3 scripts/sampled_report.py $DRAWS
echo
echo "[sampled] cells are on HF, one store per draw. Nothing here writes to"
echo "[sampled] the greedy baseline -- the store guard refuses it."
