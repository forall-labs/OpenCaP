#!/usr/bin/env bash
# A2: does naming the coordinate conventions move FRAME at the reduced API?
#
#     bash scripts/run_frame_arm.sh            # S3 and S4, 21 models x 4 tasks
#     bash scripts/run_frame_arm.sh S4         # one tier
#
# THE CLAIM UNDER TEST is the one the paper implicitly makes: that the defect
# taxonomy is ACTIONABLE. FRAME is 4% of failing programs at S1 and 47% at
# S3/S4; 19 of 21 models make at least one. Every form of it -- camera frame
# used as world, wxyz against xyzw, a 4x4 applied to a 3-vector -- is a
# CONVENTION the prompt never states, not a judgement about the task.
#
# That is why this is worth running when the two earlier interventions were
# not: the entry-point call and the defect-aware addendum both targeted
# failures the model could already see, and both came back null. This targets
# a fact the model cannot derive from the scene.
#
#     FRAME falls at S3/S4       the taxonomy is actionable, and the paper
#                                gains the constructive result it lacks
#     FRAME does not move        it is not actionable by prompting, and the
#                                word "actionable" comes out of the paper
#
# Either outcome changes what the paper says, which is why it is high in the
# order. A null here is publishable and should not be buried.
#
# 21 models x 4 tasks x 2 tiers = 168 cells, ~19h. One tier is ~9h.
#
# NEEDS THE PERCEPTION SERVERS: S3/S4 call SAM3, Contact-GraspNet and molmo.
# Unlike the worker sweep, do NOT run this with the servers down -- the
# perception calls fail and score as model defects (RESUME.md).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

TIERS="${*:-S3 S4}"

echo "############ precondition: the patch is applied ############"
# A gated patch that never applied is indistinguishable from an intervention
# that did nothing -- and "the intervention had no effect" is one of the two
# outcomes above, so this check is what keeps the null honest.
python3 scripts/patch_frame_hint.py "${CAPX_DIR:-/data/cap-x}"
if [ $? -ne 0 ]; then
  echo "[frame] REFUSING: the frame-hint patch is not applied."
  echo "[frame] Without it the arm collects a copy of the baseline and the"
  echo "[frame] null would be an artifact of the harness, not a result."
  exit 1
fi

echo
echo "############ precondition: perception servers ############"
# verify_servers.py takes <task> <tier>. Called bare it exits on an argparse
# error, which this script read as "servers down" and refused on EVERY run --
# A2 could never have started. Check each task at the first tier we will run.
FIRST_TIER="${TIERS%% *}"
SRV_RC=0
for T in cube_lifting cube_stack spill_wipe nut_assembly; do
  python3 scripts/verify_servers.py "$T" "$FIRST_TIER" \
      --root "${CAPX_DIR:-/data/cap-x}" || SRV_RC=1
done
if [ $SRV_RC -ne 0 ]; then
  echo "[frame] REFUSING: S3/S4 need SAM3 :8114, graspnet :8115, molmo :8122."
  echo "[frame] Start them:"
  echo "[frame]   bash scripts/start_vision_servers.sh"
  echo "[frame]   bash scripts/start_molmo_server.sh"
  echo "[frame] Without them the perception calls fail and score as model"
  echo "[frame] defects -- the whole S4 sweep was thrown away for this once."
  exit 1
fi

for T in $TIERS; do
  echo
  echo "############ frame_hint arm, tier $T ############"
  date -u
  PILOT_TIERS="$T" \
  OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/framehint_$T" \
    bash scripts/run_arm.sh frame_hint
  echo "FRAME_${T}_RC=$?"
done

echo
echo "[frame] Collected into hf:savoji/capx-framehint-seeded, tier-separated."
echo "[frame] The comparison is cell-by-cell against capx-ab-on-seeded at the"
echo "[frame] SAME tier. Success rate is the weak readout here -- the question"
echo "[frame] is whether FRAME defects fall, and that needs the failing"
echo "[frame] programs exported, repaired and tagged exactly as S3/S4 were."
echo "[frame] Budget that tagging pass before quoting a FRAME number."
