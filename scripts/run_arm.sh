#!/usr/bin/env bash
# One arm of the S1 panel, on this host, into its own store.
#
#     bash scripts/run_arm.sh <arm>          # baseline | numpy_off | promptfix | entrypoint
#
# Replaces the per-arm scripts. They differed only in three environment flags
# and a store name, and keeping four near-identical copies is how the arms
# drifted apart in the first place: run_host_gate.sh set OPENCAP_PREIMPORT_NUMPY
# and the handoff's own snippet did not, which would have charged a patch
# effect to the host.
#
# Every arm runs the SAME 21 models x 3 tasks x 30 frozen scenes. Arms are
# compared cell-by-cell against `baseline`, so an arm collected over a
# different panel is not a comparison, it is two different experiments.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

ARM="${1:-}"
FRAME=0
case "$ARM" in
  baseline)   STORE_DEFAULT=hf:savoji/capx-ab-on-seeded
              NUMPY=1; PFIX=0; ENTRY=0 ;;
  numpy_off)  STORE_DEFAULT=hf:savoji/capx-ab-off-seeded
              NUMPY=0; PFIX=0; ENTRY=0 ;;
  promptfix)  STORE_DEFAULT=hf:savoji/capx-promptfix-seeded
              NUMPY=1; PFIX=1; ENTRY=0 ;;
  entrypoint) STORE_DEFAULT=hf:savoji/capx-entrypoint-seeded
              NUMPY=1; PFIX=0; ENTRY=1 ;;
  # The first intervention with a mechanism behind it: FRAME is 4% of failing
  # programs at S1 and 47% at S3/S4, and every form of it is a CONVENTION the
  # prompt never states rather than a judgement about the task. See
  # scripts/patch_frame_hint.py.
  frame_hint) STORE_DEFAULT=hf:savoji/capx-framehint-seeded
              NUMPY=1; PFIX=0; ENTRY=0; FRAME=1 ;;
  *) echo "usage: $0 {baseline|numpy_off|promptfix|entrypoint|frame_hint}"; exit 2 ;;
esac

export OPENCAP_STORE="${OPENCAP_STORE:-$STORE_DEFAULT}"
export OPENCAP_SCRATCH="${OPENCAP_SCRATCH:-/data/capx-out/$ARM}"
export OPENCAP_WORKERS="${OPENCAP_WORKERS:-2}"   # 5 workers behind a batch-1 shim
                                             # is what produced the 200s-timeout
                                             # empty-program cells
mkdir -p "$OPENCAP_SCRATCH"

# Flags are SET or UNSET explicitly, never inherited. An arm that picks up a
# stray flag from the shell is indistinguishable from a real effect.
if [ "$NUMPY" = 1 ]; then export OPENCAP_PREIMPORT_NUMPY=1; else unset OPENCAP_PREIMPORT_NUMPY; fi
if [ "$PFIX"  = 1 ]; then export OPENCAP_PROMPT_FIX=1;      else unset OPENCAP_PROMPT_FIX; fi
if [ "$ENTRY" = 1 ]; then export OPENCAP_CALL_ENTRYPOINT=1; else unset OPENCAP_CALL_ENTRYPOINT; fi
if [ "$FRAME" = 1 ]; then export OPENCAP_FRAME_HINT=1;      else unset OPENCAP_FRAME_HINT; fi

case "$OPENCAP_STORE" in
  *capx-ab-on|*capx-ab-off|*capx-promptfix|*capx-entrypoint|*capx-scaling-traces|*-h200)
    echo "[arm] REFUSING: $OPENCAP_STORE is a retired corpus."
    echo "[arm] Colab-era stores are cross-host; the -h200 stores were collected"
    echo "[arm] before artifact #13, when cube_lifting, spill_wipe and"
    echo "[arm] nut_assembly drew a random scene every run. Seeded stores carry"
    echo "[arm] the -seeded suffix."
    exit 1 ;;
esac

MODELS="\
qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b qwen3-32b \
qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b \
gemma3-4b gemma3-12b gemma3-27b \
qwen35-0.8b qwen35-2b qwen35-4b qwen35-9b qwen35-27b \
qwen25-coder-1.5b qwen25-coder-3b qwen25-coder-7b"

echo "[arm] $ARM -> $OPENCAP_STORE"
echo "[arm] numpy=${NUMPY} promptfix=${PFIX} entrypoint=${ENTRY} frame_hint=${FRAME} workers=$OPENCAP_WORKERS"
echo "[arm] 21 models x 4 tasks = 84 cells; resumable, completed cells skip."

PILOT_MODELS="${PILOT_MODELS:-$MODELS}" \
PILOT_TIERS="${PILOT_TIERS:-S1}" \
PILOT_TASKS="${PILOT_TASKS:-cube_lifting cube_stack spill_wipe nut_assembly}" \
PILOT_TRIALS=30 \
bash scripts/pilot.sh
RC=$?
echo "ARM_RC=$RC"

echo "[arm] scanning $OPENCAP_STORE"
python3 scripts/scan_corpus.py "$OPENCAP_STORE"
exit $RC
