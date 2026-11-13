#!/usr/bin/env bash
# A7: the BARE tier. The rung the scaffold ladder is missing, and the only
# comparator that makes A3 (multi-turn) readable.
#
#     bash scripts/run_s2_arm.sh            # 21 models x 4 tasks = 84 cells
#
# WHY THIS EXISTS -- two reasons, and either alone would justify it.
#
# 1. A3 IS UNREADABLE WITHOUT IT. run_m1_arm.sh calls M1 "multi-turn,
#    privileged API" and says to compare it against S1. Both halves are wrong:
#    every `_multiturn` config is `privileged: false` on FrankaControlApi,
#    which is the S2 API. So M1 - S1 moves TWO axes at once (privilege and
#    turns) and M1 - S3 moves two others (API surface and turns). The
#    one-axis contrast for "what does the feedback loop buy" is M1 - S2, and
#    S2 has never been collected. 83 cells of A3 are already on the hub with
#    nothing to subtract them from.
#
# 2. PART A'S LADDER HAS A HOLE IN IT. The collected axis is
#
#      S1  privileged state + high-level primitives
#      S3  reduced API, with a worked example
#      S4  reduced API, no example
#
#    so the headline S1 -> S3 collapse conflates TWO changes: losing
#    privileged state access, and dropping to a lower-level API. S2 is bare
#    (non-privileged) but still high-level, so it separates them:
#
#      S1 -> S2   the cost of losing privileged state
#      S2 -> S3   the cost of the lower-level API
#
#    A reviewer asks this. Right now the answer is "we cannot say".
#
# S2 IS A WAVE 2 TIER AND NEEDS THE PERCEPTION SERVERS. This is the whole
# reason A3 burned 7h20m on 10-15 for two models: `run_m1_arm.sh` believed
# its own docstring, the chain ran it with the stack down, and 7 of 8 cells
# died 'Connection refused' x30 at 30 min to 2 h each. The 8th committed
# 0/30 WITH THE SERVERS DOWN. Do not let that happen to 84 more cells.
#
# ~9-11h at the Phase 5 rate (single-turn, same shape as one tier of A2).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

TIER="S2"
STORE="${OPENCAP_STORE:-hf:savoji/capx-ab-on-seeded}"

# --- PRECONDITION 1: the simulators behind S2 are seeded ---------------------
# Artifact #13 was the seeding patch reaching one simulator in four while the
# probe checked the one task where it worked. S2 configs have never been
# through this check, and an unseeded simulator makes every scene index a lie.
echo "############ precondition: simulators behind S2 ############"
python3 scripts/verify_tier_simulators.py "${CAPX_DIR:-/data/cap-x}" \
  --tiers "S1,S2" --strict
if [ $? -ne 0 ]; then
  echo "[s2] REFUSING TO COLLECT. An S2 config binds a simulator the seeding"
  echo "[s2] patch does not cover, or a binding could not be resolved. Fix"
  echo "[s2] patch_seeding.TARGETS, re-apply, re-run this check. Collecting"
  echo "[s2] anyway gives 84 cells whose scene ids do not mean what S1's do,"
  echo "[s2] and the whole point of this arm is the cell-by-cell pairing."
  exit 1
fi

# --- PRECONDITION 2: the perception stack, per task -------------------------
# verify_servers.py takes <task> <tier>. Called bare it exits on an argparse
# error, which reads as "servers down" -- that bug kept A2 from ever starting.
echo
echo "############ precondition: perception servers at S2 ############"
SRV_RC=0
for T in cube_lifting cube_stack spill_wipe nut_assembly; do
  python3 scripts/verify_servers.py "$T" "$TIER" \
      --root "${CAPX_DIR:-/data/cap-x}" || SRV_RC=1
done
if [ $SRV_RC -ne 0 ]; then
  echo "[s2] REFUSING: S2 is a WAVE 2 tier -- its configs declare SAM3 :8114,"
  echo "[s2] Contact-GraspNet :8115 and pyroki :8116. Start them:"
  echo "[s2]   bash scripts/start_vision_servers.sh"
  echo "[s2]   bash scripts/start_molmo_server.sh"
  echo "[s2] With the stack down this collects 84 cells of 'Connection"
  echo "[s2] refused' at 30 min to 2 h each, and any cell that does commit"
  echo "[s2] commits a FALSE ZERO. That is exactly what happened to A3 on"
  echo "[s2] 2026-10-15 and it cost 7h20m for two models."
  exit 1
fi

echo
echo "############ S2 (bare API, single turn) -- 21 models x 4 tasks ############"
date -u
# Same store as S1/S3/S4: a cell's path is (wave, model, tier, task), so the
# tiers do not collide and every analysis already filters on tier. Note that
# S2 is wave 2 while S1/S3/S4 are wave 1, so the path differs in the wave
# segment too -- that is by design and `completed()` does not care.
OPENCAP_STORE="$STORE" \
OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/tier_S2" \
PILOT_TIERS="$TIER" \
  bash scripts/run_arm.sh baseline
echo "TIER_S2_RC=$?"

echo
echo "############ what this unlocks ############"
python3 scripts/tier_report.py S1 S2 S3 S4
echo
echo "[s2] Two readouts, and the SECOND is the one this was run for:"
echo "[s2]   1. S1 -> S2 -> S3 separates losing privileged state from the"
echo "[s2]      lower-level API. Part A currently reports their SUM."
echo "[s2]   2. M1 - S2 is the one-axis multi-turn contrast. Pair cell by"
echo "[s2]      cell against capx-m1-seeded, NOT against S1 or S3."
echo "[s2] Success rate is the weak readout for both. The defect mix needs"
echo "[s2] the failing S2 programs exported, repaired and tagged, exactly as"
echo "[s2] S3/S4 were -- budget that pass before quoting an S2 tag number."
