"""Read results/leak_ab.json and say whether the IK-seed leak moves scores.

    python3 scripts/leak_report.py [results/leak_ab.json]

HOW TO READ IT
--------------
Every program here is a repair verified at 30/30 on this host, so the only
direction it can move is down and any movement is signal.

The comparison that matters is A vs B, and it is PAIRED: same code, same
seeded scenes, same host, differing only in whether trials are handed to four
workers or one. A2 is the same condition as A, run twice, and it is the
yardstick -- an A-vs-B difference that does not exceed the A-vs-A2 difference
is re-draw variance, not the leak. This project has twice published a
difference that was one draw from each of two overlapping distributions, so
the replicate is not optional and its numbers are printed first.

C is the fix, where the task's API allows it: the program calls
goto_home_joint_position() on its first line. C below A would mean the fix
costs something; C at ceiling where A is not means the shortfall was the leak.

DISCORDANT SCENES, not score differences, are the quantity. Two arms scoring
28/30 can agree on all thirty scenes or disagree on four, and those are
opposite conclusions.
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

from opencap.analysis import mcnemar  # noqa: E402


def outcomes(rec) -> dict:
    return {t["trial"]: t["success"] for t in rec.get("trials", [])}


def compare(recs, left, right):
    """Per-program (discordant, n_shared, mcnemar) for two arms."""
    out = []
    for pid, by_arm in sorted(recs.items()):
        a, b = by_arm.get(left), by_arm.get(right)
        if not a or not b or not a.get("ok") or not b.get("ok"):
            continue
        oa, ob = outcomes(a), outcomes(b)
        shared = set(oa) & set(ob)
        if not shared:
            continue
        disc = sum(1 for s in shared if oa[s] != ob[s])
        out.append({
            "pid": pid, "task": a["task"], "model": a["model"],
            "left": a["successes"], "right": b["successes"],
            "disc": disc, "n": len(shared),
            "mc": mcnemar(oa, ob),
        })
    return out


def summarise(rows, label):
    if not rows:
        print(f"  {label}: no comparable pairs")
        return None
    moved = [r for r in rows if r["disc"]]
    tot_disc = sum(r["disc"] for r in rows)
    tot_n = sum(r["n"] for r in rows)
    print(f"  {label}: {len(rows)} programs, {len(moved)} with any discordant "
          f"scene, {tot_disc} discordant of {tot_n} ({tot_disc/tot_n:.2%})")
    return tot_disc / tot_n


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/leak_ab.json"
    if not path.exists():
        print(f"{path} missing -- run scripts/run_leak_ab.py first")
        return 1
    raw = json.loads(path.read_text())
    failed = [r for r in raw if not r.get("ok")]
    recs: dict = defaultdict(dict)
    for r in raw:
        if r.get("ok"):
            recs[r["pid"]][r["arm"]] = r

    print("=" * 72)
    print("IK-SEED LEAK A/B".center(72))
    print("=" * 72)
    print(f"\n{len(raw)} records, {len(recs)} programs, {len(failed)} runs that "
          f"did not complete")
    for r in failed:
        print(f"  DID NOT RUN {r['pid']} arm={r['arm']}: {r.get('error')}")

    print("\nDISCORDANT SCENES (paired by trial index = seeded scene)\n")
    base = summarise(compare(recs, "A", "A2"),
                     "A  vs A2  (same condition twice -- the yardstick)")
    leak = summarise(compare(recs, "A", "B"),
                     "A  vs B   (four workers vs one -- the leak)")
    summarise(compare(recs, "A", "C"),
              "A  vs C   (stock vs program-side start-home -- the fix)")

    if base is not None and leak is not None:
        print()
        if leak <= base:
            print("  READS AS: NO LEAK. Scheduling order moves no more scenes "
                  "than re-running\n           the same condition does. The "
                  "re-draw policy in verify_repairs\n           is the right "
                  "response, and cube_stack's residual is closed.")
        else:
            # A ratio against a zero baseline is not a number, it is a
            # division by zero wearing a decimal point. Say the two rates.
            ratio = (f"{leak/base:.1f}x" if base > 0
                     else "against a replicate that moved NOTHING")
            print(f"  READS AS: LEAK PRESENT. Scheduling order moves "
                  f"{leak:.2%} of scenes\n           {ratio} "
                  f"(replicate {base:.2%}). Every cell total in the corpus\n"
                  f"           carries it, and 'state-paired' is not yet true: "
                  f"the scene is\n           fixed, the start state is not.")
        print("\n  CAVEAT: these are repairs at ceiling, so this is a LOWER "
              "bound. A program\n          sitting mid-range has more scenes "
              "able to move in either direction.")

    print("\nPER PROGRAM, where any arm disagreed\n")
    print(f"  {'model':<20}{'task':<14}{'A':>4}{'A2':>4}{'B':>4}{'C':>4}"
          f"{'A|B disc':>10}{'p':>8}")
    print("  " + "-" * 68)
    ab = {r["pid"]: r for r in compare(recs, "A", "B")}
    shown = 0
    for pid, by_arm in sorted(recs.items(),
                              key=lambda kv: -(ab.get(kv[0], {}).get("disc", 0))):
        got = {k: v.get("successes") for k, v in by_arm.items()}
        if len({v for v in got.values() if v is not None}) <= 1 and \
                not ab.get(pid, {}).get("disc"):
            continue
        r = ab.get(pid)
        one = by_arm[next(iter(by_arm))]
        def cell(k):
            v = got.get(k)
            return f"{v:>4}" if v is not None else "   -"
        disc = r["disc"] if r else 0
        pval = f"{r['mc'].p_value:.3f}" if r else ""
        print(f"  {one['model']:<20}{one['task']:<14}"
              f"{cell('A')}{cell('A2')}{cell('B')}{cell('C')}"
              f"{disc:>10}{pval:>8}")
        shown += 1
    if not shown:
        print("  every program scored identically in every arm")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
