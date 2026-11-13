#!/usr/bin/env bash
# Re-validate the harness controls on THIS hardware, before any arm runs.
#
#     bash scripts/run_controls.sh [batch|prefix|replicate]...   # default: all
#
# WHY THIS RUNS FIRST
# -------------------
# Three constants in opencap/config.py are not conventions, they are empirical
# findings -- and every one was measured on Colab:
#
#   BATCH_MAX = 1              micro-batching was the single largest source of
#                              run-to-run variation (spread 5 trials -> 1)
#   DISABLE_PREFIX_CACHING     same cell: 27/30 with a fresh engine, 0/30 with
#                              a warm one
#   GENERATION_NOISE ~ 1/30    flip rate at BATCH_MAX=1
#
# Artifact #12 showed that decode behaviour itself differs across GPU
# architectures. A control validated on other hardware is an assumption here.
# If BATCH_MAX=1 turns out to be unnecessary on H200 we are paying throughput
# for nothing; if it is INSUFFICIENT, every arm is mis-configured and the whole
# re-run is wasted. Either way it is cheaper to find out in 18 cells than in
# 189.
#
# One control is already answered and is NOT re-run: within-host generation
# determinism. The Phase 3 re-collection produced 27 same-host cell pairs and
# all 27 emitted IDENTICAL programs, so temperature-0 generation is reproducible
# on a fixed host. The 1.26 discordant states/cell that remain are simulation,
# not generation.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

WHICH="${*:-batch prefix replicate}"
want() { [[ " $WHICH " == *" $1 "* ]]; }

# A deliberately mixed subset: floor, mid-range and ceiling. Mid-range cells are
# where flips concentrate -- 33 of 34 same-host discordant states came from 6 of
# 27 cells, all mid-range -- so a control measured only on 0/30 and 30/30 cells
# would report a noise floor of zero and prove nothing.
CTRL_MODELS="${CTRL_MODELS:-qwen3-0.6b qwen25-coder-3b gemma3-27b qwen35-9b qwen3-32b}"
# cube_stack is here deliberately. The first control run used
# "cube_lifting spill_wipe" -- and artifact #13 then showed BOTH of those drew
# a random scene every run, while cube_stack was the one task whose seeding
# patch actually applied. So that run measured scene randomness and reported it
# as a harness noise floor. With every task seeded the distinction is moot, but
# a control subset that excludes the task with the strongest guarantees is how
# the mistake happened, so the subset now spans them.
CTRL_TASKS="${CTRL_TASKS:-cube_lifting cube_stack spill_wipe}"

run_ctrl() {   # name, store, extra env
  local name="$1" store="$2"; shift 2
  echo; echo "############ control: $name ############"; date -u
  env "$@" \
    OPENCAP_STORE="$store" \
    OPENCAP_SCRATCH="/data/capx-out/ctrl_$name" \
    OPENCAP_PREIMPORT_NUMPY=1 \
    PILOT_MODELS="$CTRL_MODELS" PILOT_TIERS="S1" PILOT_TASKS="$CTRL_TASKS" \
    PILOT_TRIALS=30 \
    bash scripts/pilot.sh
  echo "${name}_RC=$?"
}

# 1. Is BATCH_MAX=1 still necessary? Same cells at batch 4.
want batch && run_ctrl batch hf:savoji/capx-ctrl-batch-seeded \
  OPENCAP_BATCH_MAX=4 OPENCAP_WORKERS=2

# 2. Does prefix caching still contaminate? Same cells with it ENABLED.
#    This condition is deliberately broken; its store is never a baseline.
want prefix && run_ctrl prefix hf:savoji/capx-ctrl-prefix-seeded \
  OPENCAP_ALLOW_PREFIX_CACHE=1 OPENCAP_WORKERS=2

# 3. Noise floor: a third replicate of cells already collected twice, so the
#    per-cell flip rate rests on three points rather than two.
want replicate && run_ctrl replicate hf:savoji/capx-ctrl-replicate-seeded \
  OPENCAP_WORKERS=2

echo; echo "############ reading the controls ############"
for s in batch prefix replicate; do
  want "$s" || continue
  echo; echo "--- $s vs canonical baseline ---"
  python3 scripts/host_compare_report.py \
      hf:savoji/capx-ab-on-seeded "hf:savoji/capx-ctrl-$s-seeded" || true
done
cat <<'NOTE'

How to read these. Every control here is a SAME-HOST comparison, so the
same-host group in each report is the whole result and the noise floor to read
it against is the 1.26 discordant states/cell already measured.

  batch      discordance at the floor -> BATCH_MAX=1 is unnecessary here and
             costs throughput for nothing. Well above it -> keep it, and the
             Colab finding transfers.
  prefix     expected to be LARGE. If it is not, --deterministic may be doing
             nothing on this vLLM build, which would mean the baseline was
             never actually protected. That is the more alarming outcome.
  replicate  a third point per cell. Report the per-cell flip rate, not the
             mean -- the mean is dominated by a handful of mid-range cells and
             says nothing about whether a given delta is real.
NOTE
