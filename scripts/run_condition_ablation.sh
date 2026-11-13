#!/usr/bin/env bash
# What does a collection condition do to a published number?
#
#     bash scripts/run_condition_ablation.sh              # baseline + all conditions
#     bash scripts/run_condition_ablation.sh prefix batch # a subset
#     bash scripts/run_condition_ablation.sh --list
#
# WHY THIS EXISTS
# ---------------
# This project has found sixteen measurement artifacts and inverted ten
# headline findings. That is currently a set of war stories: each artifact was
# found opportunistically and its effect reported from whatever cell exposed
# it -- 0.000 -> 1.000 on one cube_lifting cell for the numpy import, 27/30 ->
# 0/30 for prefix caching, 33% of <=2B requests for the token cap.
#
# Anecdotes do not make a methods paper. What does is the same small grid run
# under each condition, so every artifact gets an effect size on a COMMON
# denominator and the question "would this have been visible in the published
# metric?" has an answer rather than an opinion.
#
# The deliverable is one table: condition -> success rate -> delta against
# baseline -> discordant cells. It is the ablation study nobody runs on their
# own benchmark.
#
# THE GRID IS DELIBERATELY SMALL. 5 models x 2 tasks = 10 cells per condition.
# The point is the DELTA between conditions on identical cells, not a precise
# success rate, and a wide grid would buy precision on the wrong quantity
# while costing a day per condition.
#
# Models span the ladder because several artifacts are size-dependent: the
# token cap bites the small end (33% of <=2B requests, 0% of >=12B) and the
# multi-turn loop only engages where execution throws.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

MODELS="${OPENCAP_ABL_MODELS:-qwen35-0.8b qwen3-1.7b qwen3-4b qwen3-8b gemma3-27b}"
TASKS="${OPENCAP_ABL_TASKS:-cube_lifting cube_stack}"
TIER="${OPENCAP_ABL_TIER:-S1}"

# name | store suffix | env assignments | what it reproduces
#
# Store names satisfy opencap/store.py's guards: a sampled cell may only go to a
# '-sampled' store and a raised-cap cell only to a '-bigcap' one, because each
# pair shares a shard path and one would replace the other in place.
read -r -d '' CONDITIONS <<'EOT' || true
baseline|abl-baseline||the corpus conditions, for the delta to be against
numpy_off|abl-numpyoff|OPENCAP_PREIMPORT_NUMPY=|artifact #3: one missing import took a correct cell 0.000 -> 1.000
batch8|abl-batch8|OPENCAP_BATCH_MAX=8|artifact #5: micro-batching, the largest variance source measured
prefix_on|abl-prefixon|OPENCAP_DISABLE_PREFIX_CACHING=|artifact #4: a warm cache took a cell 27/30 -> 0/30
workers1|abl-workers1|OPENCAP_WORKERS=1|2026-10-13: parallelism moves the whole noise floor
bigcap|abl-bigcap-8192|OPENCAP_MAX_NEW_TOKENS=8192|artifact #15: 33% of <=2B requests stop at the 2048 cap
sampled|abl-sampled-t07|OPENCAP_TEMPERATURE=0.7 OPENCAP_SAMPLE_INDEX=1|the decode condition every cross-model claim assumes
EOT

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "$CONDITIONS" | while IFS='|' read -r n s e w; do
    [ -n "$n" ] && printf '  %-11s %-18s %s\n' "$n" "${e:-(none)}" "$w"
  done
  exit 0
fi

WANT="${*:-}"
LOG="${OPENCAP_LOGDIR:-/tmp}/condition_ablation_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[abl] models: $MODELS"
echo "[abl] tasks: $TASKS   tier: $TIER"
echo "[abl] log: $LOG"

{
RC=0
printf '%s\n' "$CONDITIONS" | while IFS='|' read -r NAME SUFFIX ENVS WHY; do
  [ -z "$NAME" ] && continue
  if [ -n "$WANT" ] && [[ " $WANT " != *" $NAME "* ]]; then continue; fi
  echo
  echo "############ condition: $NAME ############"
  echo "[abl] $WHY"
  echo "[abl] env: ${ENVS:-(corpus defaults)}"
  date -u
  # Each condition gets its OWN store. A cell path is (wave, model, tier,
  # task) with no room for the condition, so two conditions in one store
  # would overwrite each other in place and leave a mixture -- the defect
  # this whole ablation exists to measure.
  # shellcheck disable=SC2086
  env $ENVS \
    OPENCAP_STORE="hf:savoji/capx-$SUFFIX" \
    OPENCAP_SCRATCH="${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/abl_$NAME" \
    PILOT_MODELS="$MODELS" \
    PILOT_TASKS="$TASKS" \
    PILOT_TIERS="$TIER" \
    bash scripts/run_arm.sh baseline
  r=$?; [ $r -ne 0 ] && RC=$r
  echo "ABL_${NAME}_RC=$r"
done

echo
echo "############ the table ############"
python3 scripts/condition_report.py
echo "ABL_RC=$RC"
} 2>&1 | tee "$LOG"
echo "[abl] log: $LOG"
