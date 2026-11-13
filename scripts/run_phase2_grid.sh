#!/usr/bin/env bash
# HANDOFF §1 / §7 experiment 1 -- complete the S1 grid.
#
# 18 models x 4 tasks at S1 = 72 cells; 27 already exist in capx-ab-on and are
# skipped by the store's resume, so this runs 45. Four model families complete
# (Qwen3, Qwen3-VL, Gemma 3, Qwen3.5), nothing above 32B.
#
# WRITES INTO capx-ab-on ON PURPOSE. These are baseline cells under identical
# conditions -- same tier, same arm flags, same 30 frozen scenes -- not an
# intervention. The one thing that must stay true is that every cell in that
# store was collected the same way, which is why the flags below are pinned
# rather than inherited.
#
#   RUN scripts/run_gate0_oracle.sh FIRST. nut_assembly has never run in this
#   harness. If its oracle does not reach the ceiling, every model result on
#   that task is uninterpretable and the 9 nut_assembly cells here are wasted.
#
# Models ascend by size within each family so cheap cells land first and the
# budget can be re-estimated from measurement (§7). Per-cell cost is a property
# of MODEL OUTPUT, not of the task: measured spread on one H200 was 52s to
# 2896s for cells of identical nominal size.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

export OPENCAP_SCRATCH="${OPENCAP_SCRATCH:-/data/capx-out/grid}"
export OPENCAP_STORE="${OPENCAP_STORE:-hf:savoji/capx-ab-on}"
export OPENCAP_PREIMPORT_NUMPY=1     # capx-ab-on is the numpy-ON arm (§2.1)
unset OPENCAP_PROMPT_FIX             # baseline arm: no intervention
unset OPENCAP_CALL_ENTRYPOINT
mkdir -p "$OPENCAP_SCRATCH"

# Ascending by size inside each family. qwen3-32b / qwen3vl-32b / gemma3-27b /
# qwen35-27b run at bf16 (84GB / 71GB) -- they fit on a 141GB H200 and keeping
# them unquantised keeps the whole curve one precision. On an 80GB card set
# them back to fp8 in models.py AND run the qwen3-8b-fp8 control first.
DEFAULT_MODELS="\
qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b qwen3-32b \
qwen3vl-2b qwen3vl-4b qwen3vl-8b qwen3vl-32b \
gemma3-4b gemma3-12b gemma3-27b \
qwen35-0.8b qwen35-2b qwen35-4b qwen35-9b qwen35-27b"

# --- Gate 0 preflight: drop ungated tasks, never abort the run -------------
# A task whose human oracle has not reached the ceiling produces model results
# nobody can attribute -- but that is a reason to skip THAT task, not to lose
# the other 36 cells. Tasks already represented in the corpus are grandfathered;
# only tasks new to this harness need a gate.
TASKS_IN="${PILOT_TASKS:-cube_lifting cube_stack spill_wipe nut_assembly}"
TASKS_OUT=$(python3 - "$TASKS_IN" <<'PY'
import sys
sys.path.insert(0, ".")
GATED = {"nut_assembly"}          # new to this harness; everything else has a baseline
STORE = "hf:savoji/capx-oracle"
keep, drop = [], []
for task in sys.argv[1].split():
    if task not in GATED:
        keep.append(task); continue
    try:
        from opencap.store import open_store
        from opencap.cells import Cell
        st = open_store(STORE)
        cell = Cell(model="oracle", tier="S1", task=task)
        eps = st.read_cell(cell) if cell in st.completed() else []
        ok = sum(1 for e in eps if e.get("success"))
        (keep if eps and ok == len(eps) else drop).append(task)
        if eps and ok != len(eps):
            print(f"[grid] gate0: {task} oracle scored {ok}/{len(eps)}", file=sys.stderr)
    except Exception as e:
        print(f"[grid] gate0: cannot read {STORE} for {task}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        drop.append(task)
for t in drop:
    print(f"[grid] SKIPPING {t}: no passing Gate 0 oracle. Run "
          f"scripts/run_gate0_oracle.sh {t}, then re-run this script -- "
          f"completed cells are skipped, so nothing is repeated.", file=sys.stderr)
print(" ".join(keep))
PY
)
if [ -z "$TASKS_OUT" ]; then
  echo "[grid] no task passed Gate 0 -- nothing to run"; exit 1
fi
export PILOT_TASKS="$TASKS_OUT"

echo "[grid] tasks: $PILOT_TASKS"
echo "[grid] store=$OPENCAP_STORE scratch=$OPENCAP_SCRATCH"
echo "[grid] arm: numpy preimport ON, prompt fix OFF, entrypoint OFF"
echo "[grid] completed cells are skipped by the store; this is resumable."

PILOT_MODELS="${PILOT_MODELS:-$DEFAULT_MODELS}" \
PILOT_TIERS="S1" \
PILOT_TASKS="$PILOT_TASKS" \
PILOT_TRIALS=30 \
bash scripts/pilot.sh
echo "GRID_RC=$?"
