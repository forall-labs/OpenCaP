"""Compare the two A/B arms: how much failure is import compliance, by scale.

Reports per cell:
  success off -> on            the artifact share
  failure-mode histogram       WHICH failures the import explains

The point is not the aggregate. `api_misuse` cells should move and
`physical_failure` cells should not -- if that holds, the taxonomy is
separating instruction-following from robot control, and the artifact share is
attributable rather than a lump.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.failures import classify_root_cause  # noqa: E402
from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.store import open_store  # noqa: E402


def load(spec: str):
    store = open_store(spec)
    out = {}
    for cell in store.completed():
        eps = store.read_cell(cell)
        out[(cell.model, cell.tier, cell.task)] = eps
    return out


def main() -> int:
    off = load(sys.argv[1] if len(sys.argv) > 1 else "hf:savoji/capx-ab-off-seeded")
    on = load(sys.argv[2] if len(sys.argv) > 2 else "hf:savoji/capx-ab-on-seeded")
    keys = sorted(set(off) & set(on))
    if not keys:
        print("no overlapping cells yet")
        return 1

    print(f"{'model':<11}{'tier':<5}{'task':<14}{'off':>8}{'on':>8}{'delta':>8}"
          f"  {'p(McNemar)':>10}{'flip':>6}  dominant failure (off)")
    print("-" * 100)
    tot_off = tot_on = tot_n = 0
    n_cells = 0
    for k in keys:
        m, ti, ta = k
        a, b = off[k], on[k]
        na, nb = len(a), len(b)
        sa = sum(e["success"] for e in a)
        sb = sum(e["success"] for e in b)
        tot_off += sa; tot_on += sb; tot_n += na; n_cells += 1
        # paired on state_id -- same seeds in both arms
        da = {e["state_id"]: bool(e["success"]) for e in a}
        db = {e["state_id"]: bool(e["success"]) for e in b}
        mc = mcnemar(da, db)
        # Our own taxonomy, not CaP-X's failure_label: "api_misuse" covers
        # both a missing import and a genuinely wrong API call, and only the
        # first is a scaffold artifact.
        dom = Counter(classify_root_cause(e) for e in a
                      if not e["success"]).most_common(1)
        dom_s = f"{dom[0][0]} ({dom[0][1]})" if dom else "-"
        # Totals hide direction: a 25/25 cell with two states flipping
        # opposite ways is not an unchanged cell, it is the residual
        # nondeterminism floor. The effect has to clear that floor.
        shared_st = set(da) & set(db)
        n_flip = sum(1 for s_id in shared_st if da[s_id] != db[s_id])
        print(f"{m:<11}{ti:<5}{ta:<14}{sa:>4}/{na:<3}{sb:>4}/{nb:<3}"
              f"{(sb-sa)/max(na,1):>+8.3f}  {mc.p_value:>10.4f}{n_flip:>6}  {dom_s}")

    print("-" * 100)
    print(f"overall  off {tot_off}/{tot_n} = {tot_off/tot_n:.3f} {wilson(tot_off, tot_n)}")
    print(f"         on  {tot_on}/{tot_n} = {tot_on/tot_n:.3f} {wilson(tot_on, tot_n)}")
    print(f"         artifact share of measured failure: "
          f"{(tot_on-tot_off)/max(tot_n-tot_off,1):.1%} of off-arm failures "
          f"recovered by one import, over {n_cells} paired cell(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
