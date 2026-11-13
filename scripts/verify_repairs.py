"""Re-execute repaired programs against the original 30 scene seeds.

A repair is a claim; this turns it into a measurement. Until a repaired program
has been run in the environment, its "distance to solution" is an opinion about
code.

MECHANISM -- no patch to CaP-X required. CodeExecEnvConfig already accepts an
`oracle_code` string from YAML, and `--use-oracle-code True` makes trial.py run
it verbatim instead of querying a model:

    config["use_oracle_code"] -> raw_code = env.oracle_code

So each repair is written into a generated copy of the task's own S1 config and
run through the normal launcher. No LLM is loaded, which is why this runs on a
T4 rather than an A100.

    python3 scripts/verify_repairs.py [--only PID] [--trials 30]

BAR: 30/30. Anything less is reported as such and the repair goes back for
another iteration -- except where the task's own best-known program cannot
reach 30/30 either (spill_wipe's reference scores 29/30), in which case the
shortfall is reported against that reference rather than silently accepted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.capx_bridge import config_path  # noqa: E402
from opencap.results import cross_check, is_complete, parse_trials  # noqa: E402

#: Best verified score for each task, from the corpus. A repair is not expected
#: to beat the best program anyone has produced for the task.
#: spill_wipe was set to 29 from coder-7b's single observation; a repaired
#: qwen3-14b then scored 30/30, so the task is fully solvable and 29 was one
#: noisy draw, not a ceiling.
#: What "fixed" means per task. A repair is only accepted at the task's own
#: ceiling, not merely "better than the model". `.get(task, trials)` defaults a
#: new task to a full sweep, which is the safe direction -- an unknown task
#: demands 30/30 rather than silently accepting a partial repair.
#: nut_assembly's ceiling should be confirmed from its Gate 0 oracle run
#: (hf:savoji/capx-oracle) rather than assumed; if the human oracle cannot
#: reach 30/30 on that task, no repair can either.
TASK_CEILING = {"cube_lifting": 30, "cube_stack": 30, "spill_wipe": 30,
                "nut_assembly": 30}

#: Scenes a task can flip with IDENTICAL code and identical seeds -- pure
#: simulation nondeterminism, measured on the seeded corpus (2026-09-16).
#:
#: Verification runs oracle_code with no LLM, and cube_stack was seeded
#: correctly all along, so its scenes never changed between draws. Yet 13
#: repairs read 29/30 on one draw and 30/30 on the next. That is the
#: simulator, not the code -- most likely a scene sitting on a contact
#: tolerance. Every other task showed one such case in 119.
#:
#: This is NOT a lowered acceptance bar. A repair still has to reach the
#: ceiling. What it buys is a re-draw before a result inside the band is
#: called a defect: without it, thirteen correct repairs were sent back for
#: another iteration, and three were "fixed" for a shortfall that was noise.
TASK_NOISE = {"cube_stack": 1}
#: How many extra draws to spend on a result inside the band before reporting
#: it short. One is enough: a real defect fails twice, noise usually does not.
NOISE_REDRAWS = 1


def build_config(capx: Path, task: str, code: str, out_dir: str,
                 trials: int, workers: int, dest: Path) -> None:
    """Write a copy of the task's S1 config with `oracle_code` injected."""
    import yaml
    base = yaml.safe_load((capx / config_path(task, "S1")).read_text())
    base.setdefault("env", {}).setdefault("cfg", {})["oracle_code"] = code
    base["output_dir"] = out_dir
    base["trials"] = trials
    base["num_workers"] = workers
    base["record_video"] = False
    dest.write_text(yaml.safe_dump(base, sort_keys=False, width=10**6))


def run_one(capx: Path, task: str, pid: str, code: str, trials: int,
            workers: int, scratch: Path) -> dict:

    out_dir = scratch / pid
    cfg = scratch / f"{pid}.yaml"
    # A verification must start from an EMPTY output directory. CaP-X names
    # trial dirs by outcome (trial_NN_sandboxrc_X_reward_Y_taskcompleted_Z),
    # so a re-run into an existing dir overwrites same-outcome trials and adds
    # differing ones -- the parse then reads a mixture of two runs. Stage 8 of
    # the rebuild ran its repair and minimal passes into one scratch dir and
    # every minimal record was contaminated by the repair run that preceded it
    # (31 trial dirs per pid). Same defect as the shared-scratch A/B arms (§7).
    import shutil
    for stale in (out_dir, out_dir.parent / "oracle" / out_dir.name):
        if stale.exists():
            shutil.rmtree(stale)
    build_config(capx, task, code, str(out_dir), trials, workers, cfg)
    # --no-sync: the environment is synced by bootstrap; re-resolving the
    # lockfile per record reaches github.com and a transient 500 became
    # "run did not complete" on two records of stage 8.
    cmd = ["uv", "run", "--no-sync", "capx/envs/launch.py",
           "--config-path", str(cfg),
           "--use-oracle-code", "True",
           "--total-trials", str(trials),
           "--num-workers", str(workers),
           "--record-video", "False"]
    # Match the conditions the corpus was collected under, explicitly rather
    # than inheriting from whatever shell invoked this. The originals ran in
    # the capx-ab-on arm with numpy pre-imported; verifying without it would
    # compare a repair against a different environment than its baseline.
    #
    # OPENCAP_CALL_ENTRYPOINT is deliberately NOT set: repairs for
    # `never_invoked` add the call to the CODE, as an L3 edit, so the measured
    # distance reflects the fix. Enabling the L1 scaffold patch here would
    # silently recover those programs and zero out a real edit.
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               OPENCAP_PREIMPORT_NUMPY="1")
    env.pop("OPENCAP_CALL_ENTRYPOINT", None)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=capx, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    # launch.py inserts the model slug ("oracle") into the output path.
    run_dir = out_dir.parent / "oracle" / out_dir.name
    if not run_dir.exists():
        run_dir = out_dir
    if not is_complete(run_dir):
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return {"pid": pid, "ok": False, "error": "run did not complete",
                "tail": tail, "wall": round(wall, 1)}
    trials_parsed = parse_trials(run_dir)

    # An API server that died takes trials down with it and looks exactly like
    # a broken repair -- a spill_wipe cell came back 0/30 purely because the
    # PyRoKi IK server was gone.
    #
    # This used to fire only when EVERY trial failed, which left the partial
    # case silent: on 2026-09-10 a repair scored 26/30 with four trials dying
    # on "Connection refused" from :8116, and the shortfall was attributed to
    # the code and sent back for another iteration. A second launch.py running
    # concurrently binds the same fixed IK port, so partial loss is the normal
    # shape of this failure, not the exceptional one. Count the affected
    # trials directly and refuse on any of them.
    n_unreachable = 0
    for d in sorted(run_dir.glob("trial_*")):
        f = d / "summary.txt"
        if not f.exists():
            continue
        try:
            txt = f.read_text()
        except OSError:
            continue
        if "Connection refused" in txt or "Max retries exceeded" in txt:
            n_unreachable += 1
    if n_unreachable:
        return {"pid": pid, "ok": False, "wall": round(wall, 1),
                "unreachable_trials": n_unreachable,
                "error": f"API server unreachable during {n_unreachable} of "
                         f"{len(trials_parsed)} trials (environment failure, "
                         f"not a code result). Do not run a verification "
                         f"concurrently with a sweep: both bind :8116."}

    ok, msg = cross_check(run_dir)
    return {
        "pid": pid, "ok": ok, "cross_check": msg, "wall": round(wall, 1),
        "successes": sum(1 for t in trials_parsed if t.success),
        "n": len(trials_parsed),
        "run_dir": str(run_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # Defaults come from the environment env_server.sh sets, so the pipeline
    # can call this with no arguments; the /content paths are Colab-only.
    ap.add_argument("--capx", default=os.environ.get("CAPX_DIR", "/content/cap-x"))
    ap.add_argument("--scratch",
                    default=os.path.join(os.environ.get("OPENCAP_SCRATCH_ROOT",
                                                        "/data/capx-out"), "verify"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="verify a single pid")
    ap.add_argument("--out", default="/content/verify/results.json")
    # A parallel track for MINIMISED repairs. The iterated repairs drifted well
    # away from the model's own code -- renamed variables, deleted lines that
    # were merely unused, rewrote blocks where one edit would do. Distance is
    # only meaningful as the MINIMUM edit that works, so the minimal versions
    # are verified separately and only adopted where they still reach ceiling.
    ap.add_argument("--repairs-dir", default="results/repairs",
                    help="directory of repair json files to verify")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    # The seeded export first, then the older one for repairs whose program
    # only the unseeded corpus produced. This was hard-wired to failures.json,
    # which holds none of the 21 nut_assembly programs: the first repair saved
    # for that task would have crashed stage 8 with a KeyError on its pid.
    fails: dict = {}
    for name in ("failures.json", "failures_seeded.json"):
        f = root / "results" / name
        if f.exists():
            fails.update({x["pid"]: x for x in json.loads(f.read_text())})
    reps: dict = {}
    for f in sorted(glob.glob(str(root / a.repairs_dir / "*.json"))):
        reps.update(json.loads(Path(f).read_text()))
    orphans = sorted(pid for pid in reps if pid not in fails)
    if orphans:
        # A repair for a program no export knows is unverifiable here: its
        # task is unknown. Say so and go on rather than die on the first one.
        print(f"[verify] {len(orphans)} repair(s) match no exported program and "
              f"are skipped: {' '.join(p[:12] for p in orphans)}")
        reps = {pid: r for pid, r in reps.items() if pid in fails}

    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out)
    # Create the parent up front. The record is written after each run, and a
    # missing directory turned 25 finished second-draw verifications into a
    # FileNotFoundError that a `| grep` on the console swallowed; the results
    # had to be rebuilt from the run dirs. Verify the path before spending the
    # GPU time, not after.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        done = {r["pid"]: r for r in json.loads(out_path.read_text())}

    todo = [(pid, r) for pid, r in reps.items()
            if r["code"] is not None and (a.only is None or pid == a.only)]

    # Order by ascending edit distance, so the cheap decisive checks land
    # before the expensive rewrites. If the loop is killed partway, what
    # survives is the informative half rather than an arbitrary slice.
    from opencap.repair import normalised_edit_distance
    todo.sort(key=lambda kv: normalised_edit_distance(
        fails[kv[0]]["code"], kv[1]["code"]))
    print(f"[verify] {len(todo)} repairs to run ({len(done)} already done)")
    for pid, rep in todo:
        task = fails[pid]["task"]
        # `ok` only means the cross-check passed -- an iterating repair that
        # recovered 29 of 30 also has ok=True. Skipping on `ok` alone would
        # skip every repair that still needs another attempt, which is exactly
        # the set a second pass exists to re-run.
        prev = done.get(pid)
        if prev and prev.get("ok") and \
                prev.get("successes", 0) >= TASK_CEILING.get(task, a.trials):
            print(f"[skip] {pid} already at ceiling {prev['successes']}/{prev['n']}")
            continue
        model = fails[pid]["models"][0]
        print(f"[run ] {pid} {model} {task}")
        res = run_one(Path(a.capx), task, pid, rep["code"], a.trials,
                      a.workers, scratch)
        # save_repair.py writes {code, note}; the older hand-built records
        # also carry `level`. Requiring it here made every repair saved
        # with the helper crash verification with a KeyError.
        res.update(task=task, model=model,
                   level=rep.get("level", "L3_edit"))
        ceiling = TASK_CEILING.get(task, a.trials)
        band = TASK_NOISE.get(task, 0)
        # A result inside the task's measured noise band is not yet evidence
        # of a defect. Re-draw before saying so, and keep the earlier draw
        # under `history` so the shortfall is not silently erased either.
        redraws = 0
        while (res.get("ok") and band
               and ceiling - band <= res["successes"] < ceiling
               and redraws < NOISE_REDRAWS):
            redraws += 1
            print(f"[redraw] {pid} {task}: {res['successes']}/{res['n']} is "
                  f"within the {band}-scene noise band for this task; "
                  f"re-drawing before calling it short")
            prev_draw = {k: res.get(k) for k in ("successes", "n", "wall", "ok")}
            res2 = run_one(Path(a.capx), task, pid, rep["code"], a.trials,
                           a.workers, scratch)
            res2.update(task=task, model=model,
                        level=rep.get("level", "L3_edit"))
            res2["history"] = (res.get("history") or []) + [prev_draw]
            res = res2
        if res.get("ok"):
            s, n = res["successes"], res["n"]
            verdict = ("PASS" if s >= ceiling else
                       f"SHORT of {ceiling} -- ITERATE")
            if redraws and s >= ceiling:
                verdict += f" (after {redraws} re-draw; first draw was noise)"
            print(f"[{'ok' if s >= ceiling else 'ITER'}] {pid} {model} {task}: "
                  f"{s}/{n} ({res['wall']}s)  {verdict}")
        else:
            print(f"[fail] {pid}: {res.get('error')} {res.get('tail','')[:200]}")
        done[pid] = res
        out_path.write_text(json.dumps(list(done.values()), indent=2))
    print(f"[verify] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
