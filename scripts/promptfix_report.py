"""Did the defect catalogue turn into a working prompt fix?

    python3 scripts/promptfix_report.py [baseline_store] [treatment_store]

Two questions, and the second is the real test:

  1. SUCCESS  did the score move? Paired on scene seed, exact McNemar.
  2. DEFECTS  did the SPECIFIC defects the addendum targets disappear from the
              generated code? A success gain with the targeted defects still
              present would mean the prompt helped for some other reason, and
              the catalogue would not be what fixed it.

Question 2 is answered by re-running the structural detectors over the new
programs -- the same checks that produced the tags, applied to fresh output.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.editclass import (PROVIDED, _defined_funcs, _called_names,  # noqa: E402
                             _loop_depth, _rebound, _z_values)
from opencap.models import BY_KEY  # noqa: E402
from opencap.repair import program_of  # noqa: E402
from opencap.store import open_store  # noqa: E402

#: Detectors that read the GENERATED code directly, so they work on new output
#: with no repair available. Each corresponds to a line of the addendum.
def detect(code: str) -> set[str]:
    hits = set()
    if _rebound(code):
        hits.add("api_rebound")
    defs = _defined_funcs(code)
    if defs and not (defs & _called_names(code)):
        hits.add("never_invoked")
    # z_approach on a move that is not a grasp is the harmful case; a grasp
    # with no z_approach at all is the other one. Both are approach_param.
    for line in code.splitlines():
        if "goto_pose" in line and "lift" in line.lower() and "z_approach" in line:
            hits.add("approach_param")
        if "z_approach=0.0" in line.replace(" ", ""):
            hits.add("approach_param")
    return hits


def load(spec: str):
    s = open_store(spec)
    out = {}
    for c in s.completed():
        if c.tier != "S1":
            continue
        out[(c.model, c.task)] = list(s.read_cell(c))
    return out


def main() -> int:
    base = load(sys.argv[1] if len(sys.argv) > 1 else "hf:savoji/capx-ab-on")
    trt = load(sys.argv[2] if len(sys.argv) > 2 else "hf:savoji/capx-promptfix")
    keys = sorted(set(base) & set(trt),
                  key=lambda k: (BY_KEY[k[0]].params_b if k[0] in BY_KEY else 0, k[1]))
    if not keys:
        print("no paired cells yet")
        return 0

    # Aggregate only over tasks every model has. Mixing an easy task that one
    # model ran with a hard task another ran makes the overall delta a
    # statement about the panel's composition, not about the fix. This is the
    # same defect as the unbalanced profile panel (4B had 5 cells vs 3).
    # Off-panel cells are still shown -- dropped from the mean, not hidden.
    models = {m for m, _ in keys}
    by_task: dict[str, set] = defaultdict(set)
    for m, t in keys:
        by_task[t].add(m)
    balanced = {t for t, ms in by_task.items() if ms == models}
    offpanel = [k for k in keys if k[1] not in balanced]
    keys = [k for k in keys if k[1] in balanced]
    if not keys:
        print("no task is present for all %d models -- no balanced panel yet"
              % len(models))
        for m, t in offpanel:
            print(f"  off-panel: {m} {t}")
        return 0

    print(f"{'model':<19}{'size':>6}  {'task':<14}{'base':>7}{'fix':>7}"
          f"{'delta':>8}{'p':>9}  defects removed")
    print("-" * 92)
    tb = tt = tn = 0
    dfix: Counter = Counter()
    # A prompt line can also CREATE the defect it was meant to prevent --
    # "pass z_approach when approaching" invites approach_param on lifts.
    # Removed-only accounting would score that as a clean win.
    dstay: Counter = Counter()
    dnew: Counter = Counter()
    for k in keys:
        m, task = k
        a, b = base[k], trt[k]
        da = {e["state_id"]: bool(e["success"]) for e in a}
        db = {e["state_id"]: bool(e["success"]) for e in b}
        sh = sorted(set(da) & set(db))
        if not sh:
            continue
        sa, sb, n = sum(da[i] for i in sh), sum(db[i] for i in sh), len(sh)
        mc = mcnemar({i: da[k2] for i, k2 in enumerate(sh)},
                     {i: db[k2] for i, k2 in enumerate(sh)})
        d_before = detect(program_of(a[0]))
        d_after = detect(program_of(b[0]))
        gone = d_before - d_after
        for g in gone:
            dfix[g] += 1
        for g in d_before & d_after:
            dstay[g] += 1
        for g in d_after - d_before:
            dnew[g] += 1
        tb += sa; tt += sb; tn += n
        sz = f"{BY_KEY[m].params_b:g}B" if m in BY_KEY else "?"
        print(f"{m:<19}{sz:>6}  {task:<14}{sa:>4}/{n:<3}{sb:>4}/{n:<3}"
              f"{(sb-sa)/n:>+8.3f}{mc.p_value:>9.4f}  "
              f"{','.join(sorted(gone)) if gone else ('—' if not d_before else 'none of '+','.join(sorted(d_before)))}")

    print("-" * 92)
    print(f"overall  base {tb}/{tn} = {tb/tn:.3f} {wilson(tb, tn)}")
    print(f"         fix  {tt}/{tn} = {tt/tn:.3f} {wilson(tt, tn)}")
    print(f"         delta {(tt-tb)/tn:+.3f} over {len(keys)} paired cells "
          f"({len(models)} models x {len(balanced)} task(s), balanced)")
    if offpanel:
        print(f"\n{len(offpanel)} cell(s) excluded from the mean -- task not "
              f"run for every model:")
        for m, t in sorted(offpanel):
            print(f"  {m} {t}")
    if dfix:
        print("\ntargeted defects removed from the generated code:")
        for d, n in dfix.most_common():
            print(f"  {d:<20}{n} cell(s)")
    else:
        print("\nno targeted defect disappeared -- any score change is not the "
              "catalogue doing the work")

    if dstay:
        print("\nstill present after the fix (addendum did not land):")
        for d, n in dstay.most_common():
            print(f"  {d:<20}{n} cell(s)")
    if dnew:
        print("\nINTRODUCED by the addendum (a prompt line backfired):")
        for d, n in dnew.most_common():
            print(f"  {d:<20}{n} cell(s)")
    if dfix and not dnew and tt == tb:
        print("\nNOTE: targeted defects removed with no score change. The "
              "addendum did what it says; those defects were not what was "
              "capping the score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
