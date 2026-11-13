#!/usr/bin/env bash
# The whole project, rebuilt on one machine, in order.
#
#     bash scripts/run_all_h200.sh            # everything, resumable
#     bash scripts/run_all_h200.sh 4 5        # only these stages
#
# WHY A COMPLETE REBUILD AND NOT A PATCH
# --------------------------------------
# Artifact #12: with identical weights, prompt, seed, temperature and vLLM
# build, 17 of 34 cross-host cells emitted a DIFFERENT program. Within a host,
# 0 of 27. So results collected on Colab and on H200 are not two measurements
# of one thing -- and EVERY conclusion this project reached before 2026-09-11
# was computed over Colab data or a Colab/H200 mixture.
#
# The default status of a pre-cutover claim is UNVERIFIED, not "true until
# challenged". That includes the ones nobody ever doubted: three constants in
# opencap/config.py are themselves Colab measurements, which is why stage 1 runs
# before any arm.
#
# What is NOT re-run: anything already collected on H200 and passing the data
# quality scan. Stage 2 copies those forward rather than burning GPU to
# reproduce them.
#
# Nothing here aborts on failure. Every stage is independently resumable --
# completed cells are skipped by the store.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Default runs the ESSENTIAL set. Stages 5-7 (the intervention arms) are
# opt-in: both interventions came back negative on clean data -- the entry-point
# call moves no cell at all and the prompt addendum is net negative -- so
# re-collecting them at 84 cells each buys a more precise measurement of two
# effects we have already characterised. The baseline is what the failure
# analysis runs on, and it is the one that has to be right.
#
#     bash scripts/run_all_h200.sh              # essential: 0 1 2 3 4 8
#     bash scripts/run_all_h200.sh 5 6 7        # the arms, if wanted
STAGES="${*:-0 1 2 3 4 8}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
banner() { echo; echo "############ stage $1: $2 ############"; date -u; }
RC=0
run() { "$@"; local r=$?; [ $r -ne 0 ] && RC=$r; return 0; }

# --- 0: preconditions -------------------------------------------------------
if want 0; then
  banner 0 "bootstrap and preconditions"
  run bash scripts/bootstrap_server.sh
  run python3 -m unittest discover -s tests -q
  # state pairing is not a nicety: if trial index does not map to a fixed
  # initial state, state_id is meaningless and every paired test is invalid.
  # Same default as env_server.sh. This script does not source that file, so
  # the fallback here used to be $HOME/cap-x, which does not exist on this
  # box: the probe died on a path before reaching a single task, and the
  # stage went on -- a precondition check that never ran.
  run python3 scripts/verify_pairing_states.py "${CAPX_DIR:-/data/cap-x}" 30
  # The defect taxonomy tags a call to any name the harness does not inject as
  # `unbound_name`, from a list that was maintained by hand and was wrong:
  # goto_home_joint_position is bound, nine programs were tagged for calling
  # it, and that alone produced "INTERFACE rises with scale". Read the binding
  # site, do not trust the list.
  run python3 scripts/verify_provided_api.py "${CAPX_DIR:-/data/cap-x}" --strict
fi

# --- 1: re-validate the controls -------------------------------------------
# BATCH_MAX=1, DISABLE_PREFIX_CACHING and the noise floor were all measured on
# Colab. If one of them does not hold here, every arm below is mis-configured.
# 18 cells to find that out instead of 189.
if want 1; then
  banner 1 "re-validate harness controls"
  run bash scripts/run_controls.sh
  echo "[stage1] READ THE OUTPUT before continuing. If prefix caching shows no"
  echo "[stage1] effect, --deterministic may be a no-op on this build and the"
  echo "[stage1] baseline was never protected."
fi

# --- 2: migrate reusable H200 cells ----------------------------------------
# Nothing migrates this time. Every -h200 cell was collected before artifact
# #13, when cube_lifting, spill_wipe and nut_assembly drew a random scene on
# every run. Their cell totals remain valid unpaired, which is why those stores
# stay readable -- but a cell copied forward would carry scenes that nothing
# else in the new corpus shares, and state-level pairing is the whole point of
# re-collecting. cube_stack cells could be migrated in principle; three tasks
# out of four could not, and a half-migrated store is worse than a clean one.
if want 2; then
  banner 2 "Gate 0 oracles (all four tasks, now seeded)"
  GATE0_STORE=hf:savoji/capx-oracle-seeded \
    run bash scripts/run_gate0_oracle.sh cube_lifting cube_stack spill_wipe nut_assembly
fi

# --- 3: oracles -------------------------------------------------------------
# The human reference per task: LLM-free, and the ground truth the distance
# metric is measured against. nut_assembly FAILS its gate (~13/30) and is
# expected to -- the arms drop it automatically. Do not stop.
# THE priority. 21 models x 4 tasks x 30 FIXED scenes. This is the corpus the
# failure analysis, the repair loop and the distance metric all run on, and the
# first one in the project where state_id actually identifies a scene on every
# task.
if want 3; then
  banner 3 "baseline arm -- 84 cells, seeded, four tasks"
  run bash scripts/run_arm.sh baseline
fi

# --- 4-7: the arms ----------------------------------------------------------
# Same 21 models x 3 tasks x 30 scenes in every arm. Arms are compared
# cell-by-cell, so a differently-shaped arm is a different experiment -- the old
# numpy-off arm was 5 models and the old entrypoint arm was 1.
# Repairs for whatever the baseline produces. nut_assembly is new to the panel
# and has never been repaired, so expect a fresh batch there.
if want 4; then
  banner 4 "export the failing programs, and report what still needs a repair"
  # The ground truth a repair is written against. nut_assembly joined the panel
  # after references.json was built, so it has none -- and without it nobody can
  # write the insertion its 21 programs all fail at.
  run python3 scripts/export_oracle_code.py nut_assembly
  run python3 scripts/export_failures.py results/failures_seeded.json \
        hf:savoji/capx-ab-on-seeded
  # This line is the one that was missing. Stage 4 exported work and stage 8
  # checked existing work, and nothing reported the difference: 21 nut_assembly
  # programs were exported and then skipped in silence for a whole phase.
  run python3 scripts/repair_status.py
  echo "[stage4] repair each MISSING program, verify as oracle_code, iterate to"
  echo "[stage4] ceiling, minimise, re-verify. See HANDOFF section 05."
fi

# Opt-in. Both interventions are already characterised as negative.
if want 5; then banner 5 "numpy-off arm (opt-in)";   run bash scripts/run_arm.sh numpy_off;  fi
if want 6; then banner 6 "entrypoint arm (opt-in)";  run bash scripts/run_arm.sh entrypoint; fi
if want 7; then banner 7 "prompt-fix arm (opt-in)";  run bash scripts/run_arm.sh promptfix;  fi

# --- 8: re-verify every repair on this host ---------------------------------
# 43 of 128 verification records were produced on Colab pods. Verification is
# LLM-free so it cannot suffer decode drift, but simulation noise IS
# host-dependent: a repair accepted at 30/30 there is not evidence of 30/30
# here. New output files, so nothing is skipped as already-done.
#
# -seeded, not -h200: results/verify_h200.json already holds 119 records at
# ceiling from the unseeded rebuild, and the verifier skips a pid whose
# existing record is at ceiling. Pointed at the old file, this stage ran to
# completion in seconds having verified nothing -- a no-op that reports
# success. The seeded scenes are a different measurement and get their own file.
if want 8; then
  banner 8 "re-verify all repairs and minimal versions on this host"
  # Separate scratch per pass. verify_repairs.py now also clears a pid's run
  # dir before launching, but two passes over the same pids in one directory
  # is the shape of the §7 shared-scratch trap and should not be relied on
  # to be caught downstream.
  run python3 scripts/verify_repairs.py --repairs-dir results/repairs \
        --scratch "${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/verify_repairs" \
        --out results/verify_seeded.json
  run python3 scripts/verify_repairs.py --repairs-dir results/minimal \
        --scratch "${OPENCAP_SCRATCH_ROOT:-/data/capx-out}/verify_minimal" \
        --out results/minimal_verify_seeded.json
fi

# --- audit ------------------------------------------------------------------
echo; echo "############ provenance audit ############"
python3 scripts/audit_provenance.py \
  hf:savoji/capx-ab-on-seeded hf:savoji/capx-oracle-seeded || true

echo; echo "############ repair coverage ############"
python3 scripts/repair_status.py || true

echo; echo "############ claims register ############"
python3 scripts/claims_register.py || true

echo; echo "ALL_RC=$RC"; date -u
echo "Traces are already on HF. Commit results/ and your report, then push."
exit $RC
