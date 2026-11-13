#!/usr/bin/env bash
# HANDOFF §7 experiment 2 -- complete the prompt-fix arm.
#
# Tests whether the defect catalogue is ACTIONABLE or merely descriptive: the
# addendum (scripts/patch_prompt_fix.py, gated on OPENCAP_PROMPT_FIX=1) names
# six defect classes that a verified repair had to fix, and the question is
# whether naming them changes anything.
#
# 12 models x 3 tasks for parity with capx-ab-on; 2 cells already exist and
# are skipped by the store's resume. Models ascend by size so the cheap cells
# land first and a budget can be re-estimated from real measurements rather
# than from one extrapolated cell (§7, "Budget from measurement").
#
# RUN THE HOST GATE FIRST. Every cell here is paired against capx-ab-on, which
# was collected on Colab. If scripts/host_effect_report.py shows an effect,
# these pairings cross a host boundary and the arm needs a host offset before
# any of it can be read.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

export OPENCAP_SCRATCH="${OPENCAP_SCRATCH:-/data/capx-out/pfix}"
export OPENCAP_STORE="${OPENCAP_STORE:-hf:savoji/capx-promptfix}"
export OPENCAP_PREIMPORT_NUMPY=1    # capx-ab-on is the numpy-ON arm (§2.1)
export OPENCAP_PROMPT_FIX=1         # the intervention under test
unset OPENCAP_CALL_ENTRYPOINT       # a different arm; not mixed in here
mkdir -p "$OPENCAP_SCRATCH"

echo "[arm] store=$OPENCAP_STORE scratch=$OPENCAP_SCRATCH"
echo "[arm] numpy preimport ON, prompt fix ON, entrypoint OFF"

PILOT_MODELS="${PILOT_MODELS:-qwen3-0.6b qwen25-coder-1.5b qwen3-1.7b qwen25-coder-3b qwen3-4b qwen3vl-4b gemma3-4b qwen35-4b qwen25-coder-7b qwen3-8b gemma3-12b qwen3-14b}" \
PILOT_TIERS="S1" \
PILOT_TASKS="cube_lifting cube_stack spill_wipe" \
PILOT_TRIALS=30 \
bash scripts/pilot.sh
echo "ARM_RC=$?"
