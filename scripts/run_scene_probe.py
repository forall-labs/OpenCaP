"""Is a scene a defect the program caused, or a scene that flips on its own?

    python3 scripts/run_scene_probe.py                    # cube_stack 13, defaults
    python3 scripts/run_scene_probe.py --draws 8 --scene 13

WHAT IS BEING SEPARATED
-----------------------
The first defect class this project ever verified was `stale_observation`: five
programs, four model families, ALL failing trial 13 of cube_stack, ALL at reward
exactly 0.05, all fixed by one line. That was read as something the models
shared -- they cached the target pose before an action that could move it.

The leak A/B (2026-09-20) then found that cube_stack scene 13 is the ONLY
bistable scene in the corpus: 17 of 42 cube_stack programs flip it between two
runs of byte-identical code on identical seeds, while all 3,654 other
cube_stack observations return reward exactly 1.000.

Those two facts have never been separated. If scene 13 flips on its own, then
"the repair fixed trial 13" may be a coin that landed heads, and one verified
defect class rests partly on chance.

THE DESIGN
----------
For each program: run the ORIGINAL and its MINIMAL repair, N draws each, and
count how often the scene passes.

    original fails N/N, repair passes N/N      the defect is real and the
                                               repair fixes it
    both land near N/2                         the scene is a coin flip and
                                               the class is an artifact
    original fails N/N, repair passes ~N/2     the repair helps but the scene
                                               is also unstable -- report both

CONTROLS ARE NOT OPTIONAL HERE. Programs never tagged `stale_read` are run
too. If the scene flips at the same rate for them, then "the repair fixed
scene 13" says nothing about the repair. Pass --no-controls only if you want
an uninterpretable result.

All 30 trials are run and the scene of interest is read out of them. Running
fewer trials would be cheaper, but the scene a trial draws is a function of its
index, and changing --total-trials changes a variable this probe exists to hold
still.

LLM-free. Resumable on (pid, version, draw). Same execution path as the
verifier -- see scripts/run_leak_ab.py for why that matters.
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

from opencap.results import parse_trials  # noqa: E402
from verify_repairs import run_one  # noqa: E402

try:
    from worker_chains import chains as _chains, finish_events as _finish_events
except Exception:                                    # pragma: no cover
    _chains = _finish_events = None

#: The tag whose evidence this probe re-examines.
TAG = "stale_read"


def load(sub: str) -> dict:
    out: dict = {}
    for f in sorted(glob.glob(str(ROOT / "results" / sub / "*.json"))):
        out.update(json.loads(Path(f).read_text()))
    return out


def pick(task: str, n_controls: int) -> tuple[list, list]:
    """(tagged, control) pids for this task, from the seeded tagging."""
    tags = json.loads((ROOT / "results/tags_seeded.json").read_text())
    same_task = [t for t in tags if t["task"] == task]
    tagged = [t["pid"] for t in same_task if TAG in t["tags"]]
    # Controls: same task, never carried the tag. Sorted by distance so the
    # picks are deterministic rather than whatever order the file happens to
    # be in -- a control set that changes between runs is not a control.
    others = sorted((t for t in same_task if TAG not in t["tags"]),
                    key=lambda t: (t.get("distance") or 0, t["pid"]))
    return tagged, [t["pid"] for t in others[:n_controls]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--scene", type=int, default=13)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--capx", default=os.environ.get("CAPX_DIR", "/data/cap-x"))
    ap.add_argument("--scratch-root",
                    default=os.path.join(
                        os.environ.get("OPENCAP_SCRATCH_ROOT", "/data/capx-out"),
                        "scene_probe"))
    ap.add_argument("--out", default=str(ROOT / "results" / "scene_probe.json"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4,
                    help="sim workers. 4 is the corpus condition; 1 fixes "
                         "trial order and is the alternation test")
    ap.add_argument("--pids", default=None,
                    help="comma-separated pids to probe instead of the tagged "
                         "set plus controls (role is recorded as 'targeted')")
    a = ap.parse_args()

    fails = {x["pid"]: x for x in
             json.loads((ROOT / "results/failures_seeded.json").read_text())}
    minimal = load("minimal")
    if a.pids:
        # A targeted probe: one named program, usually to test a mechanism
        # rather than a tag. Controls do not apply -- the comparison is the
        # same program under a different condition, not against other
        # programs -- so they are not silently added.
        tagged, controls = [p.strip() for p in a.pids.split(",")], []
        print(f"[probe] TARGETED: {', '.join(tagged)} (no controls; the "
              f"comparison is this program against itself)")
    else:
        tagged, controls = pick(a.task, 0 if a.no_controls else a.controls)
    if a.no_controls:
        print("[probe] WARNING: running without controls. If the scene flips "
              "at the same\n[probe] rate for untagged programs, this result "
              "cannot be interpreted.")

    role_name = "targeted" if a.pids else "tagged"
    plan = [(pid, role_name) for pid in tagged] + \
           [(pid, "control") for pid in controls]
    print(f"[probe] task={a.task} scene={a.scene} draws={a.draws}")
    print(f"[probe] {len(tagged)} tagged '{TAG}', {len(controls)} controls")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {(r["pid"], r["version"], r["draw"], r.get("workers", 4))
            for r in done}

    todo = []
    for pid, role in plan:
        f = fails.get(pid)
        if f is None:
            print(f"[probe] {pid} is not in the seeded export -- skipped")
            continue
        versions = {"original": f["code"]}
        rep = minimal.get(pid)
        if rep and rep.get("code"):
            versions["repair"] = rep["code"]
        else:
            # Say it. A program with no minimal repair contributes only half
            # the comparison, and a silently halved plan is how a stage came
            # to export work nobody ever checked.
            print(f"[probe] {pid} has no minimal repair -- original only")
        for version, code in versions.items():
            for draw in range(a.draws):
                if (pid, version, draw, a.workers) in seen:
                    continue
                todo.append((pid, role, version, code, draw, f))

    print(f"[probe] {len(todo)} runs ({len(seen)} already recorded)")
    for i, (pid, role, version, code, draw, f) in enumerate(todo, 1):
        # Per (version, draw) scratch: a re-run into a directory that already
        # holds trial dirs makes the parse read a mixture of two runs.
        scratch = Path(a.scratch_root) / f"w{a.workers}_{version}_{draw}"
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"[run ] {i}/{len(todo)} {pid} {f['models'][0]} {version} draw={draw}")
        res = run_one(Path(a.capx), a.task, pid, code, a.trials,
                      a.workers, scratch)
        rec = {"pid": pid, "role": role, "version": version, "draw": draw,
               "task": a.task, "model": f["models"][0], "scene": a.scene,
               # Recorded per record, not per file: a probe at workers=1 and
               # one at workers=4 are two conditions and must never be pooled
               # by a reader who only has the filename.
               "workers": a.workers,
               "ok": res.get("ok"), "successes": res.get("successes"),
               "n": res.get("n"), "error": res.get("error"),
               "wall": res.get("wall")}
        if res.get("run_dir"):
            trials = parse_trials(res["run_dir"])
            for t in trials:
                if t.trial == a.scene:
                    rec["scene_success"] = t.success
                    rec["scene_reward"] = t.reward
            # Every failing trial, because the question "do trials 9 and 13
            # fail together" cannot be asked of one scene's outcome. The
            # original probe recorded only the scene of interest and the
            # alternation had to be dug out of the run dirs by hand.
            rec["failed_trials"] = sorted(t.trial for t in trials
                                          if not t.success)
            # THE PER-WORKER ORDER, IN THE RECORD.
            #
            # The 2026-10-15 mechanism claim -- "trial 9 fails only when trial
            # 5 ran before it on its worker", 14 of 18 against 0 of 22 -- is
            # reconstructed from trial-directory birth times, which live in pod
            # scratch and are not committed. So the only evidence for the
            # project's current best mechanism hypothesis cannot be checked
            # from the repository, and the next person to look has to take it
            # on trust or re-run.
            #
            # Storing the chains makes the rule checkable from the committed
            # file, and makes every future draw a free out-of-sample test of
            # it. Reconstruction is a MODEL of the dynamic queue, not a
            # reading of it, so it is recorded under its own key and never
            # mixed with the measured outcomes.
            if _finish_events and _chains and a.workers > 1:
                try:
                    ev = _finish_events(res["run_dir"])
                    rec["worker_chains"] = _chains(ev, a.workers)
                    rec["finish_order"] = [t for _, t, _ in ev]
                except Exception as exc:             # pragma: no cover
                    rec["worker_chains_error"] = f"{type(exc).__name__}: {exc}"
        done.append(rec)
        out_path.write_text(json.dumps(done, indent=2))
        print(f"[done] {pid} {version} draw={draw} cell={res.get('successes')}/"
              f"{res.get('n')} scene{a.scene}="
              f"{rec.get('scene_success')} r={rec.get('scene_reward')}")

    print(f"[probe] {len(done)} records -> {out_path}")
    print("[probe] now: python3 scripts/scene_probe_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
