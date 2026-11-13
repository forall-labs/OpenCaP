#!/usr/bin/env bash
# Re-collect the whole S1 panel on H200, into a NEW store.
#
# WHY ALL 63 CELLS AND NOT JUST THE BROKEN ONES
# ---------------------------------------------
# 36 of the 63 cells in capx-ab-on came from Colab and carry no vllm_version
# or harness_commit -- that half of the corpus cannot name the stack that
# produced it. Two of them (gemma3-12b) are timeout-contaminated, and four
# H200 cells are too. Re-running only those 6 would fix the corpus but waste
# the opportunity underneath it.
#
# Re-running all 63 into an empty store buys three things for one run:
#   1. a complete single-host corpus with full provenance, no host caveat;
#   2. a 36-cell CROSS-HOST comparison (Colab vs H200) -- twelve models
#      instead of the one rung the original gate covered;
#   3. a 23-cell SAME-HOST reproducibility check (H200 vs H200), which is the
#      harness noise floor on this hardware and is what separates "the host
#      changed the answer" from "this cell is just noisy".
# Without (3), any disagreement in (2) is unattributable.
#
# The old store is FROZEN in opencap/store.py and will refuse writes. That is
# deliberate: a cell's shard path is (wave, model, tier, task), so re-running
# into the same store replaces it in place with nothing left to compare.
#
# OPENCAP_WORKERS=2 is the timeout fix. Five sim workers queueing behind a
# batch-size-1 shim pushed CaP-X's HTTP client past its 200s read timeout and
# produced 102 empty-program episodes that scored as honest zeros.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

export OPENCAP_SCRATCH="${OPENCAP_SCRATCH:-/data/capx-out/rerun}"
export OPENCAP_STORE="${OPENCAP_STORE:-hf:savoji/capx-ab-on-h200}"
export OPENCAP_PREIMPORT_NUMPY=1     # capx-ab-on is the numpy-ON arm
export OPENCAP_WORKERS="${OPENCAP_WORKERS:-2}"
unset OPENCAP_PROMPT_FIX OPENCAP_CALL_ENTRYPOINT
mkdir -p "$OPENCAP_SCRATCH"

case "$OPENCAP_STORE" in
  *capx-ab-on|*capx-ab-off|*capx-promptfix|*capx-entrypoint)
    echo "[rerun] REFUSING: $OPENCAP_STORE is a published corpus."
    echo "[rerun] Send re-runs to a new store and compare before promoting."
    exit 1 ;;
esac

# All 21 models already in the panel: the 18 grid rungs plus the three
# Qwen2.5-Coder rungs, so the new corpus is a drop-in replacement.
MODELS="\
qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b qwen3-32b \
qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b \
gemma3-4b gemma3-12b gemma3-27b \
qwen35-0.8b qwen35-2b qwen35-4b qwen35-9b qwen35-27b \
qwen25-coder-1.5b qwen25-coder-3b qwen25-coder-7b"

echo "[rerun] store=$OPENCAP_STORE workers=$OPENCAP_WORKERS"
echo "[rerun] 21 models x 3 tasks = 63 cells; resumable, completed cells skip."

PILOT_MODELS="${PILOT_MODELS:-$MODELS}" \
PILOT_TIERS="S1" \
PILOT_TASKS="${PILOT_TASKS:-cube_lifting cube_stack spill_wipe}" \
PILOT_TRIALS=30 \
bash scripts/pilot.sh
RC=$?
echo "RERUN_RC=$RC"

echo
echo "[rerun] scanning the new corpus for the defects that motivated this run"
python3 scripts/scan_corpus.py "$OPENCAP_STORE"
echo "[rerun] now compare before promoting anything:"
echo "  python3 scripts/host_compare_report.py hf:savoji/capx-ab-on $OPENCAP_STORE"
exit $RC
