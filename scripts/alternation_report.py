"""Do two trials share a failure only one of them can have per run?

    python3 scripts/alternation_report.py results/scene_probe_w1.json [...]

THE OBSERVATION
---------------
qwen3-4b's cube_stack program (81ee85d0f933) scores 28/30 on every draw. Trial
3 always fails. The SECOND failure is trial 9 on draws 0, 1, 4, 7 and trial 13
on draws 2, 3, 5, 6 -- never both, never neither, across eight draws. Under
independence that pattern has probability 1/256.

Two trials sharing something only one of them can have per run is the shape of
the worker-state mechanism Phase 6 looked for and could not see: it tested
programs at ceiling, which can only move down, and said so. This is the first
mid-range program anyone pointed at it.

THE TEST
--------
At `--workers 1` the trial order is fixed, so a per-worker carry-over has
nothing left to vary. The prediction is that the alternation STOPS: either both
trials fail every draw, or the same one does.

    alternation persists at workers=1  -> not worker state. Something in the
                                          scene pair, or the solver, and the
                                          Phase 6 refutation stands untouched.
    alternation stops at workers=1     -> worker carry-over DOES move outcomes
                                          below ceiling, and Phase 6's null is
                                          a ceiling result only -- which it
                                          already says, but nobody has shown
                                          the other half.

n = 1 program either way. This prints the per-draw failure sets and the
exclusivity count; it does not promote one program to a finding.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


def load(paths):
    recs = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            print(f"[alt] {f} missing -- skipped")
            continue
        recs += json.loads(f.read_text())
    return [r for r in recs if r.get("ok") and "failed_trials" in r]


def main() -> int:
    paths = sys.argv[1:] or [str(ROOT / "results/scene_probe.json"),
                             str(ROOT / "results/scene_probe_w1.json")]
    recs = load(paths)
    if not recs:
        print("no records with failed_trials -- re-run run_scene_probe.py "
              "(older records did not store them)")
        return 1

    by = defaultdict(list)
    for r in recs:
        by[(r["pid"], r["version"], r.get("workers", 4))].append(r)

    print("=" * 76)
    print("TRIAL-FAILURE ALTERNATION".center(76))
    print("=" * 76)
    for (pid, version, workers), rs in sorted(by.items()):
        rs.sort(key=lambda r: r["draw"])
        sets = [set(r["failed_trials"]) for r in rs]
        always = set.intersection(*sets) if sets else set()
        ever = set.union(*sets) if sets else set()
        varying = sorted(ever - always)
        print(f"\n{rs[0]['model']}  {pid}  {version}  workers={workers}  "
              f"{len(rs)} draws")
        print(f"  cell scores     {[r['successes'] for r in rs]}")
        print(f"  fails ALWAYS    {sorted(always) or '-'}")
        print(f"  fails SOMETIMES {varying or '-'}")
        for r in rs:
            print(f"    draw {r['draw']}: {sorted(r['failed_trials'])}")
        # exclusivity: among the varying trials, is exactly one present per draw?
        if len(varying) >= 2:
            counts = [len(set(r["failed_trials"]) & set(varying)) for r in rs]
            exact_one = sum(1 for c in counts if c == 1)
            print(f"  of {len(rs)} draws, exactly ONE of {varying} failed in "
                  f"{exact_one}")
            if exact_one == len(rs):
                print(f"  -> MUTUALLY EXCLUSIVE across every draw. Under "
                      f"independence this is ~1/{2**len(rs)}.")
            else:
                print(f"  -> NOT mutually exclusive (counts per draw: {counts})")
        elif len(varying) == 1:
            print(f"  -> one varying trial only; nothing to be exclusive with")
        else:
            print(f"  -> no variation between draws")

    # --- by worker count ---------------------------------------------------
    # The question this answers is not "which trial" but "does parallelism
    # matter, and how". Printed as a table because with three counts the
    # shape -- flat, monotone, or thresholded -- is the discriminator between
    # trial interleaving and contention.
    VARY = {"original": {9, 13}, "repair": {13}}
    per: dict = defaultdict(lambda: [0, 0])
    for (pid, version, workers), rs in by.items():
        tset = VARY.get(version)
        if tset is None:
            continue
        for r in rs:
            per[workers][1] += 1
            per[workers][0] += bool(set(r["failed_trials"]) & tset)
    counts = sorted(per, reverse=True)
    print("\nDRAWS WITH A VARYING TRIAL FAILING, BY WORKER COUNT")
    print("  (pooled over the program and its repair; this is the quantity")
    print("   that decides ordering vs contention, not which trial failed)\n")
    print(f"  {'workers':>9}{'draws':>8}{'with a varying failure':>26}")
    for w in counts:
        h, n = per[w]
        print(f"  {w:>9}{n:>8}{f'{h}/{n}  {h/n:.0%}' if n else '--':>26}")
    if len(counts) < 2:
        print("\n  Only one worker count here, which answers nothing. Both a")
        print("  parallel and a serial condition are needed:")
        print("    bash scripts/run_worker_sweep.sh 1 4")
    elif len(counts) == 2:
        print("\n  Two counts: enough to show parallelism matters, not enough")
        print("  to say why. A middle count separates interleaving from")
        print("  contention:  bash scripts/run_worker_sweep.sh 2")
    else:
        print("\n  Three or more counts: read the shape. Flat above 1 means")
        print("  interleaving; rising with workers means contention or load;")
        print("  nothing until a high count means a load threshold.")
        print("  scripts/claims_register.py prints that reading with intervals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
