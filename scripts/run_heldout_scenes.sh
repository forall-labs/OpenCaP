#!/usr/bin/env bash
# A1: do the repairs hold on scenes they were never written against?
#
#     bash scripts/run_heldout_scenes.sh            # references + 30 repairs
#     bash scripts/run_heldout_scenes.sh --n 40     # a bigger sample
#
# ~2h, LLM-free: no engine, no weights, no vision servers, no IK server. A GPU
# only because MuJoCo renders through EGL. Trials 31-60 are scenes nothing in
# this project has touched; see docs/experiment-plan.md section A1.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

OUT="${OPENCAP_HELDOUT_OUT:-results/heldout_scenes.json}"
LOG="${OPENCAP_LOGDIR:-/tmp}/heldout_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[heldout] out=$OUT log=$LOG gpu=$OPENCAP_GPU"

python3 scripts/run_heldout_scenes.py --out "$OUT" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "HELDOUT_RC=$RC"

echo
echo "############ held-out report ############"
python3 scripts/heldout_report.py "$OUT" 2>&1 | tee -a "$LOG"
echo
echo "[heldout] commit $OUT."
exit $RC
