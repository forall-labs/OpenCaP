#!/usr/bin/env bash
# The IK-seed leak A/B, on the H200 box.
#
#     bash scripts/run_leak_ab.sh probe     # nut_assembly, 4 arms  (~1.7h)
#     bash scripts/run_leak_ab.sh panel     # four tasks, 3 arms    (~4h)
#     bash scripts/run_leak_ab.sh probe A,B # a subset of the arms
#
# Run the probe FIRST and read it. If scheduling order moves no more scenes
# than a plain replicate does, the panel is not worth four hours -- the
# question is answered and the re-draw policy stands.
#
# LLM-free: no vLLM, no weights, no engine. It runs the simulator only, so it
# coexists with a sweep on another card -- but NOT with one on the same card,
# because launch.py binds a fixed IK port (:8116) and two of them fight. That
# is not a hypothetical: it produced a 26/30 that was charged to the code.
#
# This sources env_server.sh. phase5_launch.sh did not, and its outer-shell
# stages ran with no HF_TOKEN and no CAPX_DIR -- a 403 on the store and a
# verifier looking in Colab's /content/cap-x.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

STAGE="${1:-probe}"
ARMS="${2:-}"
case "$STAGE" in
  probe|panel) ;;
  *) echo "usage: $0 {probe|panel} [ARMS]"; exit 2 ;;
esac

OUT="${OPENCAP_LEAK_OUT:-results/leak_ab.json}"
LOG="${OPENCAP_LOGDIR:-/tmp}/leak_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ).log"
ARGS=(--stage "$STAGE" --out "$OUT")
[ -n "$ARMS" ] && ARGS+=(--arms "$ARMS")

echo "[leak] stage=$STAGE out=$OUT"
echo "[leak] log=$LOG"
# Redirected to a file AND checked by status -- never piped. Piping a run into
# tail replaces its exit status with tail's, which is how a failing test got
# pushed to main (STATUS 2026-09-14).
python3 scripts/run_leak_ab.py "${ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "LEAK_RC=$RC"

echo
echo "############ leak report ############"
python3 scripts/leak_report.py "$OUT" 2>&1 | tee -a "$LOG"

echo
echo "[leak] records are in $OUT -- commit them; the report reads them."
echo "[leak] trace dirs stay on the box; nothing here writes to HF."
exit $RC
