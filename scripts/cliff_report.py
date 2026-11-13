"""The abstraction cliff: A(m) = SR(S1) - SR(S4), per model and as a trend.

S1 gives the model privileged state and high-level primitives; S4 strips it to
low-level control with no in-context examples. A(m) is what the scaffold is
worth to model m.

  H1 (substitution) predicts A(m) SHRINKS with scale -- big models need the
     scaffold less, so scaffolding substitutes for parameters.
  H2 (multiplication) predicts it GROWS -- the scaffold pays off more the more
     capable the model.

Pairing is on (task, state_id), not state_id alone: state ids are per-task
trial indices, so pooling tasks under a bare index silently pairs cube_lifting
trial 7 against spill_wipe trial 7.

    python3 scripts/cliff_report.py hf:savoji/capx-ab-on
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.store import open_store  # noqa: E402


def load(spec):
    store = open_store(spec)
    out: dict[tuple[str, str], dict[tuple[str, int], bool]] = {}
    for c in store.completed():
        d = out.setdefault((c.model, c.tier), {})
        for e in store.read_cell(c):
            d[(c.task, e["state_id"])] = bool(e["success"])
    return out


def main() -> int:
    cells = load(sys.argv[1])
    models = sorted({m for (m, t) in cells},
                    key=lambda m: BY_KEY[m].params_b if m in BY_KEY else 0)
    rows, excluded = [], []
    print(f"{'model':<14}{'params':>8}{'SR(S1)':>10}{'SR(S4)':>10}"
          f"{'A(m)':>9}{'p':>10}{'paired n':>10}")
    print("-" * 71)
    for m in models:
        hi, lo = cells.get((m, "S1")), cells.get((m, "S4"))
        if not hi or not lo:
            continue
        shared = sorted(set(hi) & set(lo))
        if not shared:
            continue
        a = sum(hi[k] for k in shared)
        b = sum(lo[k] for k in shared)
        n = len(shared)
        mc = mcnemar({i: hi[k] for i, k in enumerate(shared)},
                     {i: lo[k] for i, k in enumerate(shared)})
        A = (a - b) / n
        # A rung at SR(S1)=0 has A(m)=0 for a structural reason, not an
        # empirical one: there is no cliff to fall off. Including it in the
        # trend manufactures "the cliff grows with scale" out of arithmetic.
        if a > 0:
            rows.append((BY_KEY[m].params_b, A))
        else:
            excluded.append(m)
        print(f"{m:<14}{BY_KEY[m].params_b:>8.1f}{a/n:>10.3f}{b/n:>10.3f}"
              f"{A:>+9.3f}{mc.p_value:>10.4f}{n:>10}")
    if excluded:
        print(f"\nexcluded from the trend (SR(S1)=0, so A(m)=0 by construction, "
              f"not by measurement): {', '.join(excluded)}")
    if len(rows) < 2:
        print(f"\nonly {len(rows)} rung(s) can show a cliff; need >=2 to read "
              "a trend. Reporting per-rung values only.")
        return 0
    print("-" * 71)
    # Sign of the change across the ladder is the whole question; a slope fit
    # on 3-5 points is not worth more than stating the endpoints.
    lo_p, lo_A = rows[0]
    hi_p, hi_A = rows[-1]
    print(f"A({lo_p:g}B) = {lo_A:+.3f}   ->   A({hi_p:g}B) = {hi_A:+.3f}"
          f"   change {hi_A - lo_A:+.3f}")
    if hi_A < lo_A:
        print("shrinks with scale -> consistent with H1 (scaffold substitutes "
              "for parameters)")
    elif hi_A > lo_A:
        print("grows with scale -> consistent with H2 (scaffold multiplies "
              "capability)")
    else:
        print("flat across the ladder -> neither H1 nor H2")
    print("\nA(m) is bounded by SR(S1): a rung at 0/30 on S1 cannot show a "
          "cliff, so read A(m) only where SR(S1) > 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
