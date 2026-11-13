#!/usr/bin/env bash
# HANDOFF §1 step 0 / §7 experiment 0 -- Gate 0 for a task never run here.
#
# Executes CaP-X's OWN human oracle program for a task at S1. No LLM, no
# serving, deterministic. It answers a question that must be settled before any
# model result on that task can be interpreted:
#
#     does this task score correctly in OUR harness?
#
# A task whose hand-written reference solution does not reach the ceiling is a
# task where a model's 0/30 means nothing -- it could be the model, the config,
# the reward, or a missing dependency, and the score cannot tell you which.
# This project has already retracted one tier (S4) whose prompt promised camera
# observations the environment could not provide; nothing objected at the time
# because nothing checked (§8).
#
# It also produces the ground-truth reference that the distance-to-solution
# metric is measured against, so it is a prerequisite twice over.
#
#     bash scripts/run_gate0_oracle.sh [task ...]     # default: nut_assembly
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

TASKS=${*:-nut_assembly}
TRIALS="${GATE0_TRIALS:-30}"
WORKERS="${OPENCAP_WORKERS:-5}"
OUT="${OPENCAP_SCRATCH:-/data/capx-out}/gate0"
GATE0_STORE="${GATE0_STORE:-hf:savoji/capx-oracle}"
mkdir -p "$OUT"

rc_all=0
for task in $TASKS; do
  cfg=$(python3 -c "from opencap.capx_bridge import oracle_config_path; print(oracle_config_path('$task'))")
  echo
  echo "=== Gate 0: $task ==="
  echo "[gate0] oracle config: $cfg"
  if [ ! -f "$CAPX_DIR/$cfg" ]; then
    echo "[gate0] FAIL: oracle config does not exist in the checkout"; rc_all=1; continue
  fi
  # Start from an EMPTY run directory. CaP-X names trial dirs by outcome, so a
  # run into a directory that already holds an earlier draw overwrites the
  # same-outcome trials and adds the differing ones; the parse then reads a
  # mixture of draws (the shared-scratch trap, HANDOFF s07). The phase 5
  # nut_assembly directory held four draws and 78 trial dirs.
  rm -rf "$OUT/$task" "$OUT/oracle/$task"

  # Same environment the corpus was collected under. The oracle is LLM-free,
  # but it still executes generated-code-shaped Python in the same namespace,
  # so the numpy preimport has to match or this is not the same environment.
  ( cd "$CAPX_DIR" && \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENCAP_PREIMPORT_NUMPY=1 \
    uv run capx/envs/launch.py \
      --config-path "$cfg" \
      --use-oracle-code True \
      --total-trials "$TRIALS" \
      --num-workers "$WORKERS" \
      --record-video False \
      --output-dir "$OUT/$task" ) 2>&1 | tail -20

  python3 - "$OUT/$task" "$task" "$TRIALS" "$GATE0_STORE" <<'PY'
import sys, pathlib, uuid
sys.path.insert(0, ".")
from opencap.results import cross_check, parse_trials
from opencap.cells import Cell
from opencap.schema import Episode
from opencap.store import open_store
out, task, trials = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
store_spec = sys.argv[4]
# CaP-X rewrites --output-dir to <dir>/<model_slug>/<run>, so the path we
# asked for usually does not exist. It was first in this list and
# parse_trials() raised FileNotFoundError on it before the real directory
# was ever tried -- a passing gate and a failing one both died here.
# launch.py inserts the model slug ("oracle") into the output path, so the run
# lands in $OUT/oracle/<task>. Look there and ONLY there. This used to glob
# oracle/* -- every task's directory -- and keep whichever parsed last, so on
# 2026-09-15 all four seeded oracle cells were committed from spill_wipe's
# 30/30 trials: cube_stack's own summary said 29 and nut_assembly's said 16.
runs = [out, out.parent / "oracle" / task]
ts, run_dir = [], None
for r in runs:
    if not r.is_dir():
        continue
    parsed = parse_trials(r)
    if parsed:
        ts, run_dir = parsed, r
if not ts:
    print(f"[gate0] {task}: NO TRIALS PARSED -- treat as failure"); sys.exit(1)
ok = sum(1 for t in ts if t.success)
print(f"[gate0] {task}: {ok}/{len(ts)} succeeded")
# The directory parse must agree with CaP-X's own summary, as it must for
# every other cell the corpus commits. Every store write asserts verified=True;
# this is the check that makes the assertion true.
agree, why = cross_check(run_dir)
if not agree:
    print(f"[gate0] {task}: parse disagrees with summaries.txt ({why}) -- not committing")
    sys.exit(1)

# The oracle run IS run data, so it belongs in HF like every other cell --
# partitioned by cell, never on a timer. It is the ground-truth reference the
# distance metric is measured against, and a Gate 0 that lives only in a
# terminal cannot be re-checked six weeks later.
# model="oracle" keeps it out of any model-keyed analysis by construction.
if len(ts) != trials:
    print(f"[gate0] {len(ts)}/{trials} trials parsed -- not committing a short cell")
else:
    cell = Cell(model="oracle", tier="S1", task=task)
    # Provenance matters here as much as anywhere: an oracle run is the
    # reference a whole task is judged against, and a reference that cannot
    # name the machine it came from is the defect this project spent Phase 3
    # correcting. The first version of this script recorded none.
    import os, subprocess
    def _gpu():
        try:
            return subprocess.run(["nvidia-smi", "--query-gpu=name",
                                   "--format=csv,noheader"], capture_output=True,
                                  text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            return ""
    def _commit():
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    gpu, commit = _gpu(), _commit()
    host_kind = os.environ.get("OPENCAP_HOST_KIND") or (
        "h200" if "H200" in gpu else ("h100" if "H100" in gpu else "unknown"))
    eps = [Episode(episode_id=f"gate0__{task}__{t.trial}__{uuid.uuid4().hex[:8]}",
                   model="oracle", tier="S1", task=task, state_id=t.trial,
                   wave=cell.wave, success=t.success, reward=t.reward,
                   weight_revision="human_oracle_code",
                   host_gpu=gpu, host_kind=host_kind,
                   harness_commit=commit).to_trace()
           for t in ts]
    try:
        open_store(store_spec).write_cell(cell, eps, verified=True)
        print(f"[gate0] committed {len(eps)} episodes to {store_spec}")
    except Exception as e:
        # Never let a push failure masquerade as a gate failure, or vice versa.
        print(f"[gate0] WARNING: could not commit to {store_spec}: "
              f"{type(e).__name__}: {e}")

if ok < trials:
    print(f"[gate0] {task} FAILED the gate ({ok}/{trials}).")
    print(f"[gate0] run_phase2_grid.sh will SKIP this task and run the others --")
    print( "[gate0] do not halt the grid for it. A model result on an ungated task")
    print( "[gate0] cannot be attributed, but the other tasks are unaffected.")
    print( "[gate0] Fix the config/reward/deps, re-run this, then re-run the grid;")
    print( "[gate0] completed cells are skipped so nothing is repeated.")
    sys.exit(1)
print(f"[gate0] {task} PASSES -- the task scores correctly in this harness.")
PY
  [ $? -ne 0 ] && rc_all=1
done
echo
echo "GATE0_RC=$rc_all"
exit $rc_all
