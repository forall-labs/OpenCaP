"""Measure how much of a cell's score is the IK-seed leak rather than the code.

    python3 scripts/run_leak_ab.py --stage probe     # nut_assembly, 4 arms
    python3 scripts/run_leak_ab.py --stage panel     # all four tasks, 3 arms

WHAT IS BEING MEASURED
----------------------
The privileged API keeps its IK seed (`self.cfg`) on the API object, and that
object lives for the whole worker process while `run_parallel_with_setup` hands
trials out from a DYNAMIC queue. So a trial's first IK solve is seeded by
whichever trial its worker happened to run before it, and the vel-cost solver
can answer with a posture up to 0.5 rad/joint away (STATUS 2026-09-16).

That makes a trial's starting configuration a function of worker scheduling
order. Seeding fixed which SCENE each trial draws; it did not fix the state the
arm starts in. Phase 5 re-collected the entire corpus to obtain state-level
pairing, and if this leaks, the corpus still does not have it.

The effect is already known to be large where it was looked for: the human
nut_assembly reference scores 16/30 as written and 30/30 with one
`goto_home_joint_position()` prepended, and one program scored 10, 7, 8 and 11
of 30 on four five-worker draws against 6, 6 on two single-worker draws.
Nobody has measured it across the panel.

THE ARMS
--------
    A    workers=4, code unchanged     the condition the corpus was collected
                                       under -- the reference arm
    A2   workers=4, code unchanged     IDENTICAL to A. The replicate control:
                                       without it an A-vs-B difference cannot
                                       be told from ordinary re-draw variance,
                                       and this project has twice published a
                                       difference that was one draw from each
                                       of two overlapping distributions.
    B    workers=1, code unchanged     one worker, so trial order is fixed.
                                       The leak is still present -- each trial
                                       still inherits the previous trial's
                                       posture -- but it is now DETERMINISTIC.
                                       A vs B therefore isolates the part of
                                       the score that depends on scheduling.
    C    workers=4, start-home         the program resets the seed itself, on
                                       its first line. This is the fix whose
                                       effect is already verified on the
                                       oracle, and it needs no patch to CaP-X:
                                       verify_repairs runs arbitrary program
                                       text through `oracle_code`, which is
                                       exactly how the 21 nut_assembly repairs
                                       were written.

WHY THE TARGETS ARE REPAIRS AND NOT THE FAILING PROGRAMS
--------------------------------------------------------
Every program run here is a repair verified at 30/30 on this host. A program
at the floor cannot show the leak: 0/30 has nowhere to move. A program at the
ceiling can only move DOWN, so any movement at all is signal and needs no
threshold argument.

C IS NOT AVAILABLE ON EVERY TASK
--------------------------------
The cube_stack API has no home function and never resets cfg. Arm C is
therefore restricted to tasks whose API actually binds the primitive --
nut_assembly today. Stage 0 of run_all_h200.sh now prints exactly what each
task's API binds (scripts/verify_provided_api.py); if a home-like primitive
turns up on another task, add it to ARM_C_TASKS rather than assuming.

Resumable: a (pid, arm) already in the output file is skipped. Each arm gets
its OWN scratch directory -- two arms sharing one output path is the defect
that made the first A/B read a mixture of both (STATUS section 7).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

# ONE execution path. verify_repairs.run_one builds the config, clears the
# stale output dir, launches, checks for a dead IK server and cross-checks the
# parse against CaP-X's own summary. Re-implementing any of that here is how
# the four per-arm sweep scripts drifted apart (scripts/run_arm.sh, header).
from opencap.results import parse_trials  # noqa: E402
from verify_repairs import run_one  # noqa: E402

#: The primitive that resets the IK seed. Prepended as its own first line.
HOME_CALL = "goto_home_joint_position()"

#: Tasks whose API binds it. See the module docstring -- do not widen this
#: without reading stage 0's output.
ARM_C_TASKS = ("nut_assembly",)

ARMS = {
    "A":  {"workers": 4, "prepend": None,      "why": "corpus condition"},
    "A2": {"workers": 4, "prepend": None,      "why": "replicate of A"},
    "B":  {"workers": 1, "prepend": None,      "why": "fixed trial order"},
    "C":  {"workers": 4, "prepend": HOME_CALL, "why": "program resets the seed"},
}

STAGES = {
    "probe": {"tasks": ("nut_assembly",), "arms": ("A", "A2", "B", "C")},
    "panel": {"tasks": ("cube_lifting", "cube_stack", "spill_wipe",
                        "nut_assembly"),
              "arms": ("A", "A2", "B")},
}


def load_programs() -> dict:
    """Minimal repairs where one exists, else the iterated repair."""
    progs: dict = {}
    for sub in ("repairs", "minimal"):
        for f in sorted(glob.glob(str(ROOT / "results" / sub / "*.json"))):
            progs.update(json.loads(Path(f).read_text()))
    return {pid: r for pid, r in progs.items() if r.get("code")}


def load_failures() -> dict:
    fails: dict = {}
    for name in ("failures.json", "failures_seeded.json"):
        f = ROOT / "results" / name
        if f.exists():
            fails.update({x["pid"]: x for x in json.loads(f.read_text())})
    return fails


def load_references(tasks) -> tuple[dict, dict]:
    """The human reference per task, as a program the A/B can run.

    THE POSITIVE CONTROL. Every other program here is a repair at 30/30, so if
    the whole experiment comes back null there is no way to tell "no leak" from
    "the measurement does not work". The nut_assembly reference is the one
    program whose response to this exact intervention is already known: 16/30
    as written, 30/30 with goto_home_joint_position() prepended, measured over
    an 11-line ablation on 2026-09-16.

    So arm C on the reference MUST come back at or near ceiling while arm A
    does not. If it does not, stop and fix the harness before reading anything
    else in the file -- the rest of the run is not interpretable.
    """
    path = ROOT / "results" / "references.json"
    if not path.exists():
        print("[leak] references.json missing -- running WITHOUT the positive "
              "control. A null result will not be interpretable.")
        return {}, {}
    refs = json.loads(path.read_text())
    progs, meta = {}, {}
    for task, r in refs.items():
        if task not in tasks or not r.get("code"):
            continue
        pid = f"reference_{task}"
        progs[pid] = {"code": r["code"], "level": "reference"}
        meta[pid] = {"task": task, "models": ["oracle"],
                     "known": f"{r.get('successes')}/{r.get('n')}"}
    return progs, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), default="probe")
    ap.add_argument("--capx", default=os.environ.get("CAPX_DIR", "/data/cap-x"))
    ap.add_argument("--scratch-root",
                    default=os.path.join(
                        os.environ.get("OPENCAP_SCRATCH_ROOT", "/data/capx-out"),
                        "leak_ab"))
    ap.add_argument("--out", default=str(ROOT / "results" / "leak_ab.json"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset, e.g. A,B")
    ap.add_argument("--only", default=None, help="a single pid")
    ap.add_argument("--no-reference", action="store_true",
                    help="drop the human reference (the positive control)")
    a = ap.parse_args()

    stage = STAGES[a.stage]
    arms = tuple(a.arms.split(",")) if a.arms else stage["arms"]
    for arm in arms:
        if arm not in ARMS:
            print(f"[leak] unknown arm {arm!r}; known: {', '.join(sorted(ARMS))}")
            return 2

    progs, fails = load_programs(), load_failures()
    if not a.no_reference:
        ref_progs, ref_meta = load_references(stage["tasks"])
        progs.update(ref_progs)
        fails.update(ref_meta)
        for pid, m in sorted(ref_meta.items()):
            print(f"[leak] positive control: {pid} (known stock score "
                  f"{m['known']}); arm C must beat arm A here")
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: list = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {(r["pid"], r["arm"]) for r in done}

    todo = []
    for pid, rep in sorted(progs.items()):
        f = fails.get(pid)
        if f is None:
            # A program no export knows has no task, so it cannot be run. Say
            # so; do not drop it in silence.
            print(f"[leak] {pid} matches no exported program -- skipped")
            continue
        if f["task"] not in stage["tasks"]:
            continue
        if a.only and pid != a.only:
            continue
        for arm in arms:
            if ARMS[arm]["prepend"] and f["task"] not in ARM_C_TASKS:
                continue
            if (pid, arm) in seen:
                continue
            todo.append((pid, arm, f, rep))

    n_progs = len({t[0] for t in todo})
    print(f"[leak] stage={a.stage} arms={','.join(arms)}")
    print(f"[leak] {len(todo)} runs over {n_progs} programs "
          f"({len(seen)} already recorded)")
    if not todo:
        print("[leak] nothing to do")
        return 0

    for i, (pid, arm, f, rep) in enumerate(todo, 1):
        spec = ARMS[arm]
        code = rep["code"]
        if spec["prepend"]:
            # Prepended as the FIRST line, before the program's own imports.
            # It is a call into the harness, not Python that needs anything
            # defined, and putting it first is what the oracle ablation
            # measured (16/30 -> 30/30).
            code = spec["prepend"] + "\n" + code
        # Per-arm scratch. Two arms writing into one directory is how the
        # first A/B came to read a mixture of both arms' trial dirs.
        scratch = Path(a.scratch_root) / arm
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"[run ] {i}/{len(todo)} {pid} {f['models'][0]} {f['task']} "
              f"arm={arm} workers={spec['workers']}")
        res = run_one(Path(a.capx), f["task"], pid, code, a.trials,
                      spec["workers"], scratch)
        rec = {
            "pid": pid, "task": f["task"], "model": f["models"][0],
            "arm": arm, "workers": spec["workers"],
            "prepend": spec["prepend"], "why": spec["why"],
            "ok": res.get("ok"), "successes": res.get("successes"),
            "n": res.get("n"), "wall": res.get("wall"),
            "error": res.get("error"),
        }
        # Per-trial outcomes, because the whole point is a PAIRED comparison.
        # Cell totals cannot tell "the same three scenes fail in both arms"
        # from "three different scenes fail in each", and those are opposite
        # conclusions about the leak.
        if res.get("run_dir"):
            rec["trials"] = [{"trial": t.trial, "success": t.success,
                              "reward": t.reward}
                             for t in parse_trials(res["run_dir"])]
        done.append(rec)
        out_path.write_text(json.dumps(done, indent=2))
        got = f"{res.get('successes')}/{res.get('n')}" if res.get("ok") else \
            f"FAILED: {res.get('error')}"
        print(f"[done] {pid} arm={arm} {got}")

    print(f"[leak] {len(done)} records -> {out_path}")
    print("[leak] now: python3 scripts/leak_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
