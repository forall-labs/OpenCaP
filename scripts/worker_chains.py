"""Which trials ran on the same worker, in what order? Rebuilt from a finished run.

    python3 scripts/worker_chains.py '/data/capx-out/scene_probe/w4_original_*'
    python3 scripts/worker_chains.py --pid 81ee85d0f933 \
        '/data/capx-out/scene_probe/original_*'

WHY
---
CaP-X's parallel runner (capx/utils/parallel_eval.py, run_parallel_with_setup)
builds one sim environment per worker and then has each worker pull the next
trial id off a shared FIFO queue whenever it is free. A worker's environment
persists across the trials it runs, so "what ran before trial k on its worker"
is the per-worker carry-over Phase 6 looked for. Nothing records that
assignment.

It can be rebuilt. Trials 1..W start first; every finish frees one worker,
which pulls the next id. So the j-th finish event, in time order, hands trial
W+j to the worker that just finished. A trial's result directory is created
when the trial finishes, so its birth time (mtime where the filesystem has no
birth time) gives the order.

LIMITS, stated so the output is not over-read: two trials finishing within a
few milliseconds can be swapped, which reassigns everything downstream of
them; and the directory is written at the end of the trial, which is close to
but not exactly when the worker pulls its next id. A rule this rebuilds should
be checked on runs it was not found on.

Used 2026-10-13 on qwen3-4b's cube_stack program (STATUS): a within-worker
predecessor rule fit the eight new workers=4 draws and failed on the eight
2026-09-21 ones -- trial 13 fails when trial 9 passes even on another worker.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

TRIAL = re.compile(r"trial_(\d+)_sandboxrc_(\d+)_reward_([\d.]+)_taskcompleted_(\d)")


def finish_events(run_dir: str):
    """-> [(finish_time, trial, success)] sorted by time."""
    ev = []
    for d in os.listdir(run_dir):
        m = TRIAL.match(d)
        if not m:
            continue
        st = os.stat(os.path.join(run_dir, d))
        t = getattr(st, "st_birthtime", None) or st.st_mtime
        ev.append((t, int(m.group(1)), m.group(4) == "1"))
    return sorted(ev)


def chains(ev, workers: int) -> list[list[int]]:
    """Per-worker trial order under the dynamic-queue model."""
    chain = {k: [k] for k in range(1, workers + 1)}
    owner = {k: k for k in range(1, workers + 1)}
    nxt = workers + 1
    for _, k, _ in ev:
        if nxt > len(ev):
            break
        owner[nxt] = owner[k]
        chain[owner[k]].append(nxt)
        nxt += 1
    return [chain[k] for k in sorted(chain)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="probe scratch dirs (globs ok)")
    ap.add_argument("--pid", default="*", help="program id under oracle/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trials", default="9,13",
                    help="trials whose predecessor to print")
    a = ap.parse_args()
    watch = [int(t) for t in a.trials.split(",") if t]

    paths = sorted(p for g in a.runs for p in glob.glob(g))
    if not paths:
        print("no run dirs matched")
        return 1
    for run in paths:
        found = sorted(glob.glob(os.path.join(run, "oracle", a.pid)))
        if not found:
            continue
        if len(found) > 1:
            print(f"{run}: {len(found)} programs -- pass --pid")
            continue
        ev = finish_events(found[0])
        ok = {k: s for _, k, s in ev}
        cs = chains(ev, a.workers)
        pred = {c[i + 1]: c[i] for c in cs for i in range(len(c) - 1)}
        fails = sorted(k for k, s in ok.items() if not s)
        preds = "  ".join(f"pred({t})={pred.get(t)}" for t in watch)
        print(f"{os.path.basename(run):18s} fails={fails}  {preds}")
        for c in cs:
            print("    " + " ".join(f"{k}{'' if ok.get(k, True) else 'x'}"
                                    for k in c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
