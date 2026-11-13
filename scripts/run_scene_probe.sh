#!/usr/bin/env bash
# Is cube_stack scene 13 a defect the models share, or a scene that flips?
#
#     bash scripts/run_scene_probe.sh            # 8 draws, 3 controls
#     bash scripts/run_scene_probe.sh --draws 12
#
# LLM-free: no vLLM, no weights, no engine. It coexists with another job on a
# different card, but NOT with a sweep on the same card -- launch.py binds a
# fixed IK port (:8116) and two of them fight, which has already produced a
# 26/30 that was charged to the code.
#
# Sources env_server.sh: phase5_launch.sh did not, and its outer-shell stages
# ran with no HF_TOKEN and no CAPX_DIR.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

OUT="${OPENCAP_PROBE_OUT:-results/scene_probe.json}"
LOG="${OPENCAP_LOGDIR:-/tmp}/scene_probe_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[probe] out=$OUT log=$LOG"

# Redirected AND status-checked, never piped into something that eats it:
# piping a run into tail is how a failing test reached main.
python3 scripts/run_scene_probe.py --out "$OUT" "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "PROBE_RC=$RC"

echo
echo "############ scene probe report ############"
python3 scripts/scene_probe_report.py "$OUT" 2>&1 | tee -a "$LOG"

echo
echo "[probe] commit $OUT -- the report reads it and the register will too."
exit $RC
