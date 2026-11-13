#!/usr/bin/env bash
# Artifact #15, job 3: do the worst-truncating models write programs with
# endings when given room?
#
#     bash scripts/run_bigcap_arm.sh                 # 3 models x 4 tasks
#     bash scripts/run_bigcap_arm.sh qwen35-2b       # one model
#
# THE QUESTION
# ------------
# 33.4% of requests from the <=2B band hit the 2,048-token cap; 0% from >=12B.
# The cap therefore varies along the same axis as every by-band result in this
# project. The eight programs we can SEE were cut do not move the trend
# (claims_register, "not the token cap"), but looks_truncated() only catches a
# cut that is visible in the code -- and qwen35-0.8b truncates on 69.9% of its
# requests while only 3 of its 4 corpus programs are visibly cut.
#
# This arm closes the other end: give the three worst models room and see
# whether their defect tags change. If missing_action and syntax fall away, the
# small end of the taxonomy is partly the cap. If the tags survive, the small
# models genuinely produce those defects.
#
# MAX_NEW_TOKENS IS A FROZEN CONTROL. It is gated, not edited: raising the
# constant in opencap/config.py would silently change the condition every future
# cell is collected under. store.py refuses a raised-cap cell into a
# standard-cap store and the reverse, because the two share a shard path and
# artifact #15 is exactly that they are indistinguishable once written.
#
# 3 models x 4 tasks = 12 cells. Budget ~5h: these are the slow rungs, and they
# are slow PRECISELY because their requests run to the cap -- raising it makes
# them slower, not faster. Do not read a long cell as a hang.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

# The three models above 20% truncation, worst first. The other five with a
# nonzero rate (qwen35-4b 18.8, qwen35-27b 17.9, qwen25-coder-1.5b 14.1,
# qwen3vl-8b 8.2, qwen35-9b 7.6) are a follow-on, not this run.
MODELS="${*:-qwen35-0.8b qwen35-2b qwen3vl-2b}"
CAP="${OPENCAP_MAX_NEW_TOKENS:-8192}"
STORE="${OPENCAP_STORE:-hf:savoji/capx-lowtrunc-bigcap}"

case "$STORE" in
  *-bigcap*) ;;
  *) echo "[bigcap] REFUSING: $STORE is not a raised-cap store."
     echo "[bigcap] store.py would refuse the write anyway; failing here is cheaper."
     exit 1 ;;
esac

echo "[bigcap] models: $MODELS"
echo "[bigcap] max_new_tokens=$CAP (default 2048) -> $STORE"
echo "[bigcap] 12 cells, resumable, completed cells skip."

OPENCAP_MAX_NEW_TOKENS="$CAP" \
OPENCAP_STORE="$STORE" \
OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/bigcap" \
PILOT_MODELS="$MODELS" \
  bash scripts/run_arm.sh baseline
RC=$?
echo "BIGCAP_RC=$RC"

echo
echo "############ truncation under the raised cap ############"
# The point of the run: did the cap stop binding? Compare against the
# greedy-era rate for the same models (33.4% for the <=2B band).
python3 scripts/truncation_audit.py --cap "$CAP" --since "$(date -u +%Y-%m-%d)"

echo
echo "[bigcap] Next, and it is NOT automatic: export the failing programs from"
echo "[bigcap] $STORE, repair them, and re-tag. The question is whether"
echo "[bigcap] missing_action / missing_coverage / syntax survive the room."
exit $RC
