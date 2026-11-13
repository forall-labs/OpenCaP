#!/usr/bin/env bash
# A2's REAL readout: did FRAME defects fall, or only the success rate move?
#
#     bash scripts/run_frame_tagging.sh            # S3 and S4
#     bash scripts/run_frame_tagging.sh S4
#     bash scripts/run_frame_tagging.sh S3 --dry-run     # export + tag, no GPU
#
# WHY THIS IS A SEPARATE RUN FROM A2 ITSELF
# -----------------------------------------
# A2's claim is that the defect taxonomy is ACTIONABLE: that stating the
# coordinate conventions removes FRAME defects. The success rate is a PROXY for
# that, and a weak one -- a cell is one greedy program, and a prompt change
# re-draws it, so a cell going 0 -> 18 is one program swapped for another.
#
# The measurement that answers the actual claim is the tag mix among A2's
# FAILING programs, against the baseline's. Getting it needs the same pass
# S3/S4 went through:
#
#     1. export the failing programs from the frame-hint store
#     2. repair each to ceiling, verified, and minimise the repair
#     3. tag the minimal repair
#
# Step 2 needs the GPU and the perception stack; steps 1 and 3 do not.
#
# READ scripts/frame_report.py FIRST, and A8 before that. If a plain re-draw
# moves a 27/30 cell as far as the hint did, there is nothing here to explain
# and this pass should not be spent. A8 is the arm that answers it.
#
# ORDER WITHIN THE PASS. The baseline's own S3/S4 tags already exist
# (results/tags_reduced.json, 169 programs). This produces the arm's, into a
# SEPARATE file -- the two populations must not be merged, and the comparison
# is between them.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

DRY=""
TIERS=""
for A in "$@"; do
  case "$A" in
    --dry-run) DRY="--dry-run" ;;
    S3|S4)     TIERS="$TIERS $A" ;;
    *)         echo "[frametag] unknown argument: $A"; exit 2 ;;
  esac
done
TIERS="${TIERS:-S3 S4}"

ARM_STORE="${OPENCAP_FRAME_STORE:-hf:savoji/capx-framehint-seeded}"

# --- PRECONDITION: the stack, unless this is export+tag only ---------------
# Repairing means RUNNING candidate programs, which at S3/S4 means perception.
# A repair "verified" against a dead stack is a verified nothing, and it would
# enter the corpus as a measured distance.
if [ -z "$DRY" ]; then
  echo "############ precondition: declared api_servers are listening ############"
  SRV_RC=0
  FIRST_TIER="$(echo $TIERS | awk '{print $1}')"
  for T in cube_lifting cube_stack spill_wipe nut_assembly; do
    python3 scripts/verify_servers.py "$T" "$FIRST_TIER" \
        --root "${CAPX_DIR:-/data/cap-x}" || SRV_RC=1
  done
  if [ $SRV_RC -ne 0 ]; then
    echo "[frametag] REFUSING: repairs are VERIFIED BY RUNNING them, and at"
    echo "[frametag] S3/S4 that means SAM3, graspnet and pyroki must answer."
    echo "[frametag]   bash scripts/start_vision_servers.sh"
    echo "[frametag]   bash scripts/start_molmo_server.sh"
    echo "[frametag] Or pass --dry-run to export and tag without repairing."
    exit 1
  fi
fi

for T in $TIERS; do
  LOW="$(printf '%s' "$T" | tr '[:upper:]' '[:lower:]')"
  FAIL="results/failures_frame_${LOW}.json"

  echo
  echo "############ 1. export A2's failing programs, $T ############"
  # Its OWN file. results/failures_${LOW}.json is the BASELINE's export, and
  # overwriting it would silently replace the population every published S3/S4
  # tag was derived from.
  python3 scripts/export_failures.py "$FAIL" "$ARM_STORE" --tier "$T"
  if [ $? -ne 0 ]; then
    echo "[frametag] export failed for $T; not continuing to repairs."
    continue
  fi

  for TASK in cube_lifting cube_stack spill_wipe nut_assembly; do
    echo
    echo "############ 2. repair + minimise, $T $TASK ############"
    date -u
    # The acceptance bar is "no scene fails in ALL k draws" -- `score ==
    # trials` is wrong for grasping tasks, because plan_grasp resamples and
    # caps even CORRECT code near 90%. k defaults per task inside the batch.
    python3 scripts/spill_wipe_batch.py \
      --tier "$T" --task "$TASK" --failures "$FAIL" \
      --out "results/frame_batch_${LOW}_${TASK}.json" $DRY
    echo "FRAMETAG_${T}_${TASK}_RC=$?"
  done
done

echo
echo "############ 3. tag, and what to compare ############"
echo "[frametag] Tagging is a RATING step, not a script: the minimal repair is"
echo "[frametag] read and one tag is recorded per distinct cause."
echo "[frametag]   python3 scripts/tag_reduced.py add --tier S3 --pid <id> \\"
echo "[frametag]       --tag frame_convention --tag missing_action"
echo "[frametag]"
echo "[frametag] Then the comparison, arm against baseline, among FAILING"
echo "[frametag] programs only -- a tier with more failures does not"
echo "[frametag] automatically show larger shares:"
echo "[frametag]   python3 scripts/cross_tier_report.py"
echo "[frametag]"
echo "[frametag] THE RESULT IS THE FRAME SHARE, NOT THE RATE. Baseline S3/S4"
echo "[frametag] is 47% of failing programs (n=154). If the arm's share falls"
echo "[frametag] and the rate did not move, the taxonomy is actionable and the"
echo "[frametag] rate was the wrong instrument. If the share holds, the word"
echo "[frametag] 'actionable' comes out of the paper -- publish that."
echo "[frametag]"
echo "[frametag] Provenance: record WHO tagged these and whether they were"
echo "[frametag] blind, as artifacts/second_rater/ratings.json does. A rater"
echo "[frametag] who knows which arm a program came from is not a measurement"
echo "[frametag] of the arm."
