#!/usr/bin/env bash
# HANDOFF §0.1 / §7 experiment 1 -- the gate that precedes everything else.
#
# Re-runs the OVERLAP_RUNG (qwen3-8b, S1, all three tasks) on this host so the
# paired difference against the Colab-collected reference can be measured. If
# that difference is significant, results from the two machines cannot be
# pooled without a host offset.
#
# Two deliberate choices, both about not changing more than one thing:
#
#   OPENCAP_PREIMPORT_NUMPY=1 -- capx-ab-on is the numpy-preimport-ON arm
#     (§2.1). The handoff's own §7 snippet omits this; running without it
#     would compare an OFF arm here against an ON arm there and attribute the
#     patch's effect to the host. For qwen3-8b the patch is believed to be a
#     no-op (STATUS.md: only 4B/cube_lifting moved in the A/B), but "believed
#     to be a no-op" is not a reason to vary it inside the control.
#
#   A separate store -- never write these cells into capx-ab-on. The reference
#     baseline has to stay a single-host corpus to remain the reference.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

export OPENCAP_SCRATCH="${OPENCAP_SCRATCH:-/data/capx-out/host}"
export OPENCAP_STORE="${OPENCAP_STORE:-hf:savoji/capx-host-check}"
export OPENCAP_PREIMPORT_NUMPY=1
unset OPENCAP_PROMPT_FIX OPENCAP_CALL_ENTRYPOINT
mkdir -p "$OPENCAP_SCRATCH"

echo "[gate] store=$OPENCAP_STORE scratch=$OPENCAP_SCRATCH"
echo "[gate] arm: numpy preimport ON, prompt fix OFF, entrypoint OFF"

PILOT_MODELS="qwen3-8b" \
PILOT_TIERS="S1" \
PILOT_TASKS="cube_lifting cube_stack spill_wipe" \
PILOT_TRIALS=30 \
bash scripts/pilot.sh
echo "GATE_RC=$?"
