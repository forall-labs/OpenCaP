#!/usr/bin/env bash
# A3: does execution feedback help, and where can it even see the failure?
#
#     bash scripts/run_m1_arm.sh               # M1, 21 models x 4 tasks
#
# M1 is CaP-X's multi-turn tier: the model sees the execution
#
# CORRECTION (2026-10-27): this file used to call M1 "privileged API" and tell
# you to compare it against S1. Both are wrong, and the chain believed them.
# Every `_multiturn` config is `privileged: false` on FrankaControlApi -- the
# S2 API -- and declares sam3 :8114 (all four tasks), graspnet :8115 (cube
# tasks) and pyroki :8116. Consequences, both of which bit:
#   * run with the stack DOWN on 2026-10-15, A3 managed 2 of 21 models in
#     7h20m: 7 of 8 cells died 'Connection refused' x30 at 30 min-2 h each,
#     and the 8th committed 0/30 WITH NO STACK. That cell is a false zero and
#     must be deleted from capx-m1-seeded before any re-run (resume keeps it).
#   * the comparator is M1 - S2, not M1 - S1 (two axes) and not M1 - S3 (two
#     others). S2 was never collected; scripts/run_s2_arm.sh collects it.
# result and may revise. It is 'wave 1' in opencap.cells, which is about whether
# the MODEL needs vision -- M1 still declares three perception services.
#
# WHAT THIS TESTS, AND THE LIMIT OF IT
# ------------------------------------
# Phase 1 found the multi-turn loop is an ERROR-RECOVERY mechanism, not a
# refinement one: it engages only when execution THROWS. With a correct
# harness it went entirely unexercised -- MAX_TURNS=10, 810 episodes, every
# one single-turn -- because models that emit runnable-but-wrong code never
# re-enter it.
#
# The hand tags put a number on how much there is to see:
#
#     programs whose defect RAISES at runtime
#       S1      19/70   = 27%
#       S3/S4  101/169  = 60%
#
# So the sharp test is multi-turn AT THE REDUCED API, where most failures are
# the kind the loop can actually catch. THAT TIER DOES NOT EXIST. CaP-X ships
# _multiturn (privileged) and _multiturn_vdm_reduced_api (M4, which needs the
# VDM vision server and adds a modality confound). There is no text-only
# reduced-API multi-turn config.
#
# This runs M1 as shipped, which is still worth having: it closes H2 from the
# roadmap -- interaction gain, never tested -- and it gives the prediction a
# number. With only 27% of S1 defects raising, expect a SMALL effect; that is
# the mechanism claim being quantified rather than a disappointment.
#
# If the owner wants the sharp version, it needs a config composed from the
# two shipped YAMLs. That is a tier CaP-X never evaluated, so it breaks
# comparability with their published numbers -- a deliberate trade, not a
# detail. Decide before building it, not after.
#
# 21 models x 4 tasks = 84 cells, ~9h. NEEDS THE PERCEPTION SERVERS.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

echo "############ precondition: declared api_servers are listening ############"
# Checks what the tier's OWN config declares, which is the only claim that has
# not been wrong about M1. A tier declaring nothing makes this a no-op.
SRV_RC=0
for TASK in cube_lifting cube_stack spill_wipe nut_assembly; do
  python3 scripts/verify_servers.py "$TASK" M1 \
      --root "${CAPX_DIR:-/data/cap-x}" || SRV_RC=1
done
if [ $SRV_RC -ne 0 ]; then
  echo "[m1] REFUSING: M1 declares sam3 :8114, graspnet :8115 and pyroki :8116."
  echo "[m1]   bash scripts/start_vision_servers.sh"
  echo "[m1]   bash scripts/start_molmo_server.sh"
  echo "[m1] Run on 2026-10-15 without them, this did 2 of 21 models in 7h20m"
  echo "[m1] and committed one false zero. Do not repeat it."
  exit 1
fi

echo
echo "############ M1 (multi-turn, BARE/S2 API) ############"
echo "[m1] 84 cells. Expect a small effect: only 27% of S1 defects raise,"
echo "[m1] and the loop fires on nothing else."
date -u

OPENCAP_STORE="${OPENCAP_STORE:-hf:savoji/capx-m1-seeded}" \
OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/m1" \
PILOT_TIERS="M1" \
  bash scripts/run_arm.sh baseline
RC=$?
echo "M1_RC=$RC"

echo
echo "[m1] Compare against capx-ab-on-seeded at S2, cell by cell -- same"
echo "[m1] models, same tasks, same scenes, one axis different."
echo "[m1] Report turns_used: if the loop never engaged, the success numbers"
echo "[m1] are not a test of feedback, they are a re-run of S1."
exit $RC
