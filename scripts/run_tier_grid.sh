#!/usr/bin/env bash
# The scaffold axis: does the ABSTRACTION LEVEL change what kind of failure you
# get, or only how many?
#
#     bash scripts/run_tier_grid.sh            # S3 then S4, 21 models x 4 tasks
#     bash scripts/run_tier_grid.sh S3         # one tier
#
# WHY THIS RUN
# ------------
# This project is called scaffold-scaling and has collected exactly one tier.
# 84 of 84 cells are S1. H1 (substitution), H2 (multiplication), H3 and H4 from
# the roadmap are all untested, because the scaffold axis was never varied.
#
# The failure-mode work is now the primary result, and this run is what gives
# it a second dimension. We have a taxonomy of what open-weight models get
# wrong at S1: overwhelmingly software integration and instruction-following --
# an uncalled function, a shadowed primitive, a missing final action, the wrong
# object's coordinates -- and almost never manipulation reasoning.
#
# The question no one else can answer with our corpus: AT A LOWER ABSTRACTION
# LEVEL, DOES THE MIX CHANGE? If S1 failures are integration defects and S4
# failures become genuinely robotic -- bad IK, bad trajectories, bad geometry --
# then a benchmark score conflates two different things with different fixes.
# If the mix is flat and only the rate moves, that is equally publishable and
# considerably more surprising.
#
#   S1  privileged state + high-level primitives      (collected)
#   S3  reduced API                                   _reduced_api
#   S4  reduced API, no in-context example            _reduced_api_exampleless
#
# S3 and S4 DO declare perception servers (SAM3, graspnet, pyroki) despite
# being 'wave 1' -- the wave split is about whether the MODEL needs vision,
# not about whether the tier needs services. The precondition below reads
# each tier's own config and refuses if what it declares is not up.
# For S2 use scripts/run_s2_arm.sh, which also checks the seeding.
# 21 models x 4 tasks
# x 2 tiers = 168 cells. Budget ~18-20h at the Phase 5 rate; the qwen35-0.8b
# and -2b rungs are the long tail and they get longer at a reduced API, not
# shorter.
#
# EXPECT LOW SCORES AND DO NOT READ THAT AS A FAILED RUN. S1 is the most
# privileged tier and the panel already scores 26.5% there. S4 may floor most
# of the ladder. A floor is a result here: the object of study is the failure
# MIX among programs that fail, not the success rate.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

TIERS="${*:-S3 S4}"
STORE="${OPENCAP_STORE:-hf:savoji/capx-ab-on-seeded}"

# --- PRECONDITION: whatever these tiers declare, must be listening ---------
# Not keyed on anyone's prose about which tiers "need perception". That prose
# is what cost A3 7h20m: run_m1_arm.sh calls M1 the privileged API, this
# script's own header says "text-only, no perception servers", and
# opencap.cells says wave 1 "needs no perception GPU" -- all three are false for
# at least one tier here. S3 and S4 declare SAM3, graspnet and pyroki; every
# `_multiturn` config declares them too; S2 is bare but non-privileged and
# declares them as well.
#
# verify_servers.py reads the tier's OWN config and checks only what it
# declares, printing "no api_servers declared -- nothing to check" when there
# is nothing. So calling it unconditionally, for every tier and task, is
# correct for a tier that needs no stack and is the only thing that protects
# one that does. A tier added later is covered without anyone remembering.
echo
echo "############ precondition: declared api_servers are listening ############"
SRV_RC=0
for T in $TIERS; do
  for TASK in cube_lifting cube_stack spill_wipe nut_assembly; do
    python3 scripts/verify_servers.py "$TASK" "$T" \
        --root "${CAPX_DIR:-/data/cap-x}" || SRV_RC=1
  done
done
if [ $SRV_RC -ne 0 ]; then
  echo "[tiers] REFUSING: a tier in this grid declares an api_server that is"
  echo "[tiers] not listening. Start the stack:"
  echo "[tiers]   bash scripts/start_vision_servers.sh"
  echo "[tiers]   bash scripts/start_molmo_server.sh"
  echo "[tiers] Collecting anyway gives 'Connection refused' x30 per cell at"
  echo "[tiers] 30 min-2 h each, and any cell that DOES commit commits a false"
  echo "[tiers] zero -- 2026-10-15, A3, one cell recorded 0/30 with no stack."
  exit 1
fi

# --- PRECONDITION. Do not skip this to save four minutes. -------------------
# Artifact #13 was the seeding patch reaching one simulator in four while the
# probe checked the one task where it worked. Both guards added afterwards say
# S1. If a reduced-API config binds a simulator the patch does not touch, this
# grid repeats #13 on a fresh set of cells and will look just as plausible.
echo "############ precondition: simulators behind every tier ############"
python3 scripts/verify_tier_simulators.py "${CAPX_DIR:-/data/cap-x}" \
  --tiers "$(echo S1 $TIERS | tr ' ' ',')" --strict
RC=$?
if [ $RC -ne 0 ]; then
  echo "[tiers] REFUSING TO COLLECT. A config in this grid binds a simulator the"
  echo "[tiers] seeding patch does not cover, or a binding could not be resolved."
  echo "[tiers] Fix patch_seeding.TARGETS, re-apply, re-run this check."
  exit 1
fi

for T in $TIERS; do
  echo
  echo "############ tier $T -- 21 models x 4 tasks ############"
  date -u
  # Same store: a cell's path is (wave, model, tier, task), so tiers do not
  # collide and every analysis already filters on tier. A separate store per
  # tier would fragment the corpus for no gain.
  OPENCAP_STORE="$STORE" \
  OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/tier_$T" \
  PILOT_TIERS="$T" \
    bash scripts/run_arm.sh baseline
  echo "TIER_${T}_RC=$?"
done

echo
echo "############ failure mix by tier ############"
python3 scripts/tier_report.py $TIERS

echo
echo "[tiers] Next, and NOT automatic: export the failing programs per tier,"
echo "[tiers] repair, and tag. The comparison that matters is the tag mix at"
echo "[tiers] S1 vs S3 vs S4 -- not the success rates."
