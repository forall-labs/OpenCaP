#!/usr/bin/env bash
# The whole gap-closing programme, in one detached chain.
#
#     bash scripts/run_all_experiments.sh                  # everything, ~49h
#     bash scripts/run_all_experiments.sh A1 B1            # a subset
#     bash scripts/run_all_experiments.sh --list
#
# Run it detached. The chain outlives a dropped session, every job is
# independently resumable, and nothing here aborts the chain on a failure --
# a job that dies is recorded and the next one starts.
#
#     setsid nohup bash scripts/run_all_experiments.sh \
#         > /data/capx-logs/allexp.log 2>&1 &
#
# (setsid, not tmux: the tmux server died twice in Phase 7 and took the run
# with it.)
#
# WHY THESE, AND WHY THIS ORDER
# -----------------------------
# Part A -- what open-weight models get wrong at robot code, and how it
# changes with API abstraction -- is the paper. Part B, the measurement
# artifacts, is a second paper only if B1 earns it. Full reasoning in
# docs/experiment-plan.md; HANDOFF section 09 is the operational version.
#
# A1 is first because it is the cheapest and the only one that can invalidate
# an instrument: if the repairs are fitted to their thirty scenes, every
# distance in the corpus is an in-sample number.
#
# A BAD A1 RESULT DOES NOT INVALIDATE THE REST. A2, A3 and A4 collect new
# data whose worth does not depend on it, so the chain is safe to leave
# unattended. A1 changes how the corpus is described, not whether the rest is
# worth having.
#
# THE SERVER STATE DIFFERS PER JOB, which is the only real complication in
# running these together:
#
#     A1  held-out scenes      servers DOWN    LLM-free
#     B1  condition ablation   servers DOWN    needs the engine
#     --- start SAM3, graspnet, molmo ---
#     A7  S2 bare tier         servers UP      the missing ladder rung
#     A8  draw stability, cube_lifting/S3      the bound A2 actually needs
#     A3  M1 multi-turn        servers UP      <-- MOVED, see below
#     A2  frame-convention     servers UP      S3/S4
#     A4  draw stability S3    servers UP      S3
#
# A3 WAS IN THE SERVERS-DOWN GROUP AND THAT WAS A BUG. run_m1_arm.sh called
# M1 "multi-turn, privileged API", so this chain ran it with the stack down on
# 2026-10-15; every `_multiturn` config is privileged:false and declares sam3,
# graspnet and pyroki. Seven of eight cells died 'Connection refused' x30 at
# 30 min-2 h each, and the eighth committed 0/30 with no stack -- a false zero
# that must be deleted before A3 re-runs (scripts/delete_cell.py).
#
# A8 EXISTS BECAUSE A4 MEASURED THE WRONG CELLS. A4 ran cube_stack at S3,
# where every greedy cell is 0/30, so it bounds how far a re-draw moves a cell
# AT THE FLOOR -- where a cell has nowhere to fall. All ten of A2's movers are
# on other tasks, several at 27/30, and none of them is bounded by that number.
# A8 re-runs the same five-draw design on cube_lifting/S3, where the baseline
# is 27/30, and that is what decides whether A2 is a result.
#
# A7 RUNS BEFORE A3 ON PURPOSE. M1 is the S2 API plus multi-turn, and S2 has
# never been collected, so 83 finished A3 cells currently have nothing to be
# subtracted from. A7 is also the rung Part A's ladder is missing: S1 -> S3
# conflates losing privileged state with dropping to a lower-level API, and
# S2 separates them.
#
# The servers-down jobs are ordered first so the stack comes up ONCE, in the
# middle, rather than being started and stopped around each job. Starting a
# 27B engine or three perception services changes what else fits on the card,
# and B1 measures exactly that kind of thing.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source scripts/env_server.sh

JOBS_DOWN="A1 B1"             # servers must be DOWN (or are irrelevant)
JOBS_UP="A7 A8 A3 A2 A4"      # servers must be UP
# A7 before A3: A3 has no comparator until S2 exists.
# A8 is the re-draw bound A2 needs and A4 did not provide.

describe() {
  case "$1" in
    A1) echo "held-out scenes        ~2h   LLM-free   can invalidate the distance metric" ;;
    B1) echo "condition ablation     ~8h   engine     gates whether Part B is a paper" ;;
    A7) echo "S2 bare tier           ~10h  SERVERS    unlocks A3; splits the S1->S3 collapse" ;;
    A8) echo "draw stability OFF the floor ~11h SERVERS  decides A2; A4 bounded only 0/30 cells" ;;
    A3) echo "M1 multi-turn          ~9h   SERVERS    compare to S2, NOT S1; read len(turns)" ;;
    A2) echo "frame-convention hint  ~19h  SERVERS    a null is publishable, do not bury it" ;;
    A4) echo "draw stability at S3   ~11h  SERVERS    DONE, and it bounds only 0/30 cells -- see A8" ;;
  esac
}

run_job() {
  case "$1" in
    A1) bash scripts/run_heldout_scenes.sh ;;
    B1) bash scripts/run_condition_ablation.sh ;;
    A7) bash scripts/run_s2_arm.sh ;;
    A8) OPENCAP_SAMPLED_TASK=cube_lifting OPENCAP_SAMPLED_TIER=S3 \
          bash scripts/run_sampled_arm.sh ;;
    A3) bash scripts/run_m1_arm.sh ;;
    A2) bash scripts/run_frame_arm.sh ;;
    A4) OPENCAP_SAMPLED_TASK=cube_stack OPENCAP_SAMPLED_TIER=S3 \
          bash scripts/run_sampled_arm.sh ;;
    *)  echo "[all] unknown job $1"; return 2 ;;
  esac
}

if [ "${1:-}" = "--list" ]; then
  for j in $JOBS_DOWN $JOBS_UP; do printf '  %-3s %s\n' "$j" "$(describe "$j")"; done
  exit 0
fi

WANT="${*:-$JOBS_DOWN $JOBS_UP}"
declare -A RC=()

card() {
  nvidia-smi -i "${OPENCAP_GPU}" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}

echo "############ gap-closing programme ############"
date -u
echo "[all] gpu=$OPENCAP_GPU  jobs: $WANT"
echo "[all] card at start:"; card
echo

# --- phase 1: servers down -------------------------------------------------
for J in $JOBS_DOWN; do
  [[ " $WANT " == *" $J "* ]] || continue
  echo; echo "######## $J -- $(describe "$J") ########"; date -u
  echo "[all] servers must be DOWN for this job; not starting them."
  run_job "$J"; RC[$J]=$?
  echo "[all] ${J}_RC=${RC[$J]}"
done

# --- the stack, once -------------------------------------------------------
NEED_UP=0
for J in $JOBS_UP; do [[ " $WANT " == *" $J "* ]] && NEED_UP=1; done
if [ "$NEED_UP" = 1 ]; then
  echo; echo "######## starting the perception stack ########"; date -u
  bash scripts/start_vision_servers.sh
  bash scripts/start_molmo_server.sh
  # Checked, not assumed: the whole S4 sweep was thrown away once because
  # only PyRoKi came up and the models scored 0/30 following a prompt that
  # documented a key the environment never produced.
  SRV=0
  for T in cube_lifting cube_stack spill_wipe nut_assembly; do
    python3 scripts/verify_servers.py "$T" S3 --root "${CAPX_DIR:-/data/cap-x}" \
      || SRV=1
  done
  if [ "$SRV" -ne 0 ]; then
    echo "[all] PERCEPTION STACK DID NOT COME UP. Skipping $JOBS_UP rather than"
    echo "[all] collecting arms whose perception failures would score as model"
    echo "[all] defects. The servers-down jobs above are unaffected."
    for J in $JOBS_UP; do [[ " $WANT " == *" $J "* ]] && RC[$J]=86; done
    NEED_UP=0
  fi
fi

# --- phase 2: servers up ---------------------------------------------------
if [ "$NEED_UP" = 1 ]; then
  for J in $JOBS_UP; do
    [[ " $WANT " == *" $J "* ]] || continue
    echo; echo "######## $J -- $(describe "$J") ########"; date -u
    run_job "$J"; RC[$J]=$?
    echo "[all] ${J}_RC=${RC[$J]}"
  done
fi

echo; echo "############ summary ############"; date -u
echo "[all] card at end:"; card
FAIL=0
for J in $WANT; do
  r="${RC[$J]:-skipped}"
  case "$r" in
    0)  note="ok" ;;
    86) note="SKIPPED -- perception stack did not come up" ;;
    *)  note="FAILED"; FAIL=1 ;;
  esac
  printf '  %-3s rc=%-8s %s\n' "$J" "$r" "$note"
done

cat <<'EOT'

[all] What to send back, per job:
  A1  results/heldout_scenes.json  + the report. READ IT FIRST: if the repairs
      drop and the four references do not, every distance in the corpus is an
      in-sample number. That is a result, not a failure.
  B1  the condition table. The success rate is the least important column --
      cells moved, worst cell, and "visible?" are the methods result. Note
      that the 0.95% floor is a workers>1 quantity, so say which condition it
      applies to rather than applying it to all seven.
  A2  the arm, AND whether FRAME actually moved. Success rate is the weak
      readout; the question needs the failing programs exported, repaired and
      tagged as S3/S4 were. Budget that pass before quoting a FRAME number.
      A null is publishable -- it means "actionable" comes out of the paper.
  A3  turns_used above all. If the loop never engaged, the success numbers are
      a re-run of S1 rather than a test of feedback.
  A4  per-model stability of the static FRAME proxy. No tagging needed.

[all] Commit results/*.json as they land. Traces go to HF automatically.
EOT
echo "ALL_RC=$FAIL"
exit $FAIL
