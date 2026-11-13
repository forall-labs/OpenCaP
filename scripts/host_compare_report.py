"""Compare two collections of the same panel, split by what actually differs.

    python3 scripts/host_compare_report.py hf:savoji/capx-ab-on hf:savoji/capx-ab-on-h200

The original host gate covered ONE rung (qwen3-8b) on three tasks and found 0
discordant states of 90. That licensed pooling for that rung and, by
extrapolation, for twenty others. A full re-collection tests the extrapolation
directly.

Two groups, and reporting them together would waste the run:

  CROSS-HOST   reference cell came from Colab. Differences here confound host
               with everything else that changed (serving stack, timeouts).
  SAME-HOST    reference cell was already H200. Differences here are run-to-run
               noise on this hardware, since nothing varied but the run.

The same-host group is what makes the cross-host group readable: a 2-scene
disagreement means one thing if same-host cells are perfectly reproducible and
something else entirely if they are not. Cells known to be contaminated are
reported separately again -- they are expected to differ and including them in
either group would inflate it.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.repair import program_of  # noqa: E402
from opencap.store import open_store  # noqa: E402

EMPTY_LIMIT = 0.20


def load(spec):
    s = open_store(spec)
    return {(c.model, c.task): list(s.read_cell(c))
            for c in s.completed() if c.tier == "S1"}


def host_of(eps):
    return Counter(e.get("host_kind", "?") for e in eps).most_common(1)[0][0]


def empty_frac(eps):
    n = len(eps) or 1
    return sum(1 for e in eps if not (program_of(e) or "").strip()) / n


def main() -> int:
    a_spec = sys.argv[1] if len(sys.argv) > 1 else "hf:savoji/capx-ab-on"
    b_spec = sys.argv[2] if len(sys.argv) > 2 else "hf:savoji/capx-ab-on-h200"
    A, B = load(a_spec), load(b_spec)
    keys = sorted(set(A) & set(B),
                  key=lambda k: (BY_KEY[k[0]].params_b if k[0] in BY_KEY else 0, k[1]))
    if not keys:
        print("no cells in common yet")
        return 0

    groups = {"CROSS-HOST (colab -> h200)": [], "SAME-HOST (h200 -> h200)": [],
              "CONTAMINATED in reference": []}
    for k in keys:
        a, b = A[k], B[k]
        da = {e["state_id"]: bool(e["success"]) for e in a}
        db = {e["state_id"]: bool(e["success"]) for e in b}
        sh = sorted(set(da) & set(db))
        if not sh:
            continue
        mc = mcnemar({i: da[s] for i, s in enumerate(sh)},
                     {i: db[s] for i, s in enumerate(sh)})
        row = (k, sum(da[s] for s in sh), sum(db[s] for s in sh), len(sh), mc)
        if empty_frac(a) > EMPTY_LIMIT:
            groups["CONTAMINATED in reference"].append(row)
        elif host_of(a) == "h200":
            groups["SAME-HOST (h200 -> h200)"].append(row)
        else:
            groups["CROSS-HOST (colab -> h200)"].append(row)

    verdict_ok = True
    for name, rows in groups.items():
        if not rows:
            continue
        print(f"\n=== {name}: {len(rows)} cells ===")
        print(f"{'model':<19}{'task':<14}{'ref':>7}{'new':>7}{'disc':>6}{'p':>9}")
        disc_tot = ta = tb = tn = 0
        for (m, task), sa, sb, n, mc in rows:
            d = mc.b + mc.c
            disc_tot += d; ta += sa; tb += sb; tn += n
            mark = "  <-- differs" if d else ""
            print(f"{m:<19}{task:<14}{sa:>4}/{n:<2}{sb:>4}/{n:<2}{d:>6}{mc.p_value:>9.4f}{mark}")
        print(f"{'':33}{ta:>4}/{tn:<3}{tb:>4}/{tn:<3}{disc_tot:>5} discordant states")
        print(f"  ref {wilson(ta, tn)}   new {wilson(tb, tn)}")
        if name.startswith("SAME-HOST") and disc_tot:
            print(f"  NOTE: {disc_tot} discordant states with NOTHING varied but the run.")
            print( "  That is this hardware's noise floor -- read the cross-host")
            print( "  group against it, not against zero.")
        if name.startswith("CROSS-HOST") and disc_tot:
            verdict_ok = False

    cross = groups["CROSS-HOST (colab -> h200)"]
    if cross:
        d = sum(mc.b + mc.c for *_, mc in cross)
        same = sum(mc.b + mc.c for *_, mc in groups["SAME-HOST (h200 -> h200)"])
        print("\n" + "-" * 62)
        if d == 0:
            print("VERDICT: zero cross-host discordance across "
                  f"{len(cross)} cells. The original one-rung gate generalises; "
                  "the corpora are interchangeable and the new one can be promoted.")
        elif d <= same:
            print(f"VERDICT: {d} cross-host discordant states, at or below the "
                  f"{same}-state same-host noise floor. Consistent with noise "
                  "rather than a host effect.")
        else:
            print(f"VERDICT: {d} cross-host discordant states against a "
                  f"{same}-state same-host floor. This EXCEEDS noise -- do not "
                  "pool the two corpora until it is explained.")
    return 0 if verdict_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
