"""The ablation table: what each collection condition does to the number.

    python3 scripts/condition_report.py

One row per condition, on a grid of identical cells. The quantity is the DELTA
against the baseline -- the success rate itself is from 10 cells and is not
the point.

TWO COLUMNS MATTER MORE THAN THE RATE.

  cells moved      how many of the shared cells changed score at all. A
                   condition that shifts the pooled rate by two points while
                   moving one cell by sixty is not a small effect; it is a
                   large effect on one cell, and the pooled number hides it.
  worst cell       the largest single-cell swing. Artifact #4 was found this
                   way: 27/30 -> 0/30 on one cell, invisible in a mean.

VISIBILITY is the methods question. A condition whose effect is smaller than
the project's own noise floor would NOT have been caught by any check here,
and that is the finding -- not that it is harmless, but that nothing would
have told you.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.store import (StoreUnreachable, open_store,  # noqa: E402
                         store_exists)

#: name -> store suffix. Mirrors run_condition_ablation.sh; a condition added
#: there and not here is reported as missing rather than skipped.
CONDITIONS = [
    ("baseline",  "abl-baseline"),
    ("numpy_off", "abl-numpyoff"),
    ("batch8",    "abl-batch8"),
    ("prefix_on", "abl-prefixon"),
    ("workers1",  "abl-workers1"),
    ("bigcap",    "abl-bigcap-8192"),
    ("sampled",   "abl-sampled-t07"),
]

#: Below this, the project's own checks would not have flagged the difference.
#: 0.95% is the measured post-determinism-fix noise floor (opencap/config.py).
NOISE = 0.0095


def read(suffix: str) -> dict | None:
    """Cells, or None if the store does not exist at all.

    `store_exists` FIRST, because opening an HF store creates it: this script
    created seven empty repos the first time it was run against an ablation
    that had not started. None means "never collected"; {} means "the store is
    there and holds nothing", and those are different things to report -- the
    second has already been mistaken for a verdict over zero cells once in
    this project.
    """
    spec = f"hf:savoji/capx-{suffix}"
    if not store_exists(spec):
        return None
    try:
        s = open_store(spec)
        out = {}
        for c in s.completed():
            eps = list(s.read_cell(c))
            out[(c.model, c.tier, c.task)] = {
                "n": len(eps),
                "k": sum(1 for e in eps if e.get("success")),
                "by_state": {e.get("state_id"): bool(e.get("success"))
                             for e in eps if e.get("state_id") is not None},
            }
        return out
    except Exception as e:
        # NOT {}: an empty dict is reported as "exists but holds no cells",
        # which is a claim about the collection. A failed read is not.
        raise StoreUnreachable(
            f"{spec} exists but could not be read: {type(e).__name__}: {e}"
        ) from e


def main() -> int:
    data = {name: read(sfx) for name, sfx in CONDITIONS}
    missing = [n for n, d in data.items() if d is None]
    empty = [n for n, d in data.items() if d == {}]
    if missing:
        print(f"not collected (store absent): {', '.join(missing)}")
    if empty:
        print(f"store exists but holds NO cells: {', '.join(empty)} -- that is "
              f"not a result over zero cells, it is an uncollected condition")
    base = data.get("baseline") or {}
    if not base:
        print("no baseline cells -- run `bash scripts/run_condition_ablation.sh "
              "baseline` first.\nEvery row here is a delta against it; without "
              "it there is nothing to report.")
        return 1

    print("=" * 78)
    print("COLLECTION CONDITIONS: EFFECT ON THE PUBLISHED NUMBER".center(78))
    print("=" * 78)
    print(f"\nbaseline grid: {len(base)} cells, "
          f"{sum(c['n'] for c in base.values())} episodes\n")
    print(f"{'condition':<12}{'cells':>6}{'success':>18}{'delta':>9}"
          f"{'moved':>7}{'worst cell':>12}{'visible?':>10}")
    print("-" * 78)

    bk = sum(c["k"] for c in base.values())
    bn = sum(c["n"] for c in base.values())
    brate = bk / bn if bn else 0.0

    for name, _ in CONDITIONS:
        d = data.get(name) or {}
        if not d:
            label = "store absent" if data.get(name) is None else "0 cells"
            print(f"{name:<12}{'--':>6}{label:>18}")
            continue
        shared = sorted(set(d) & set(base))
        if not shared:
            print(f"{name:<12}{0:>6}{'no shared cells':>18}")
            continue
        k = sum(d[c]["k"] for c in shared)
        n = sum(d[c]["n"] for c in shared)
        kb = sum(base[c]["k"] for c in shared)
        nb = sum(base[c]["n"] for c in shared)
        rate, rbase = k / n, kb / nb
        delta = rate - rbase
        moved = sum(1 for c in shared
                    if d[c]["k"] / d[c]["n"] != base[c]["k"] / base[c]["n"])
        worst = max((abs(d[c]["k"] / d[c]["n"] - base[c]["k"] / base[c]["n"])
                     for c in shared), default=0.0)
        iv = wilson(k, n)
        vis = "yes" if abs(delta) > NOISE or worst > 0.10 else "NO"
        print(f"{name:<12}{len(shared):>6}"
              f"{f'{k}/{n} {iv.point:.3f}':>18}{delta:>+9.3f}"
              f"{moved:>5}/{len(shared)}{worst:>11.0%}{vis:>10}")

    print(f"\nbaseline success {bk}/{bn} = {brate:.3f}; "
          f"visibility threshold {NOISE:.2%} pooled or 10% on one cell\n")

    # Paired, per condition: the pooled rate can sit still while cells move in
    # both directions, which is a bigger finding than a shift.
    print("PAIRED BY SCENE, against baseline\n")
    print(f"{'condition':<12}{'discordant scenes':>20}{'p':>10}  cells moving both ways")
    print("-" * 78)
    for name, _ in CONDITIONS:
        if name == "baseline":
            continue
        d = data.get(name) or {}
        shared = sorted(set(d) & set(base))
        if not shared:
            continue
        n01 = n10 = tot = 0
        up = down = 0
        for c in shared:
            a, b = base[c]["by_state"], d[c]["by_state"]
            keys = set(a) & set(b)
            tot += len(keys)
            d01 = sum(1 for s in keys if a[s] and not b[s])
            d10 = sum(1 for s in keys if not a[s] and b[s])
            n01 += d01
            n10 += d10
            if d01 and not d10:
                down += 1
            elif d10 and not d01:
                up += 1
            elif d01 and d10:
                up += 1
                down += 1
        if not tot:
            continue
        mc = mcnemar({i: True for i in range(n01)} | {-i - 1: False for i in range(n10)},
                     {i: False for i in range(n01)} | {-i - 1: True for i in range(n10)})
        print(f"{name:<12}{f'{n01 + n10}/{tot}':>20}{mc.p_value:>10.3g}"
              f"  {down} worse, {up} better")

    print("\nA condition marked NO under 'visible?' is the methods result: its")
    print("effect is below what any check in this project would have caught, so")
    print("nothing would have told you it was there. That is not the same as")
    print("harmless -- read it next to the worst-cell column.\n")
    return 0


def _guarded(fn):
    """Run main(), but never let an unreachable hub print as a null result.

    `store_exists` raises StoreUnreachable when the hub could not answer --
    a dead token makes every private store 401, i.e. indistinguishable from
    never created. Printing "not collected" then would be a measurement
    artifact dressed as a finding, so this exits 2 and says what to fix.
    """
    def run() -> int:
        try:
            return fn()
        except StoreUnreachable as e:
            print(f"CANNOT READ THE STORES -- this says NOTHING about what was\n"
                  f"collected. Nothing below would have been a result.\n\n{e}")
            return 2
    return run

if __name__ == "__main__":
    raise SystemExit(_guarded(main)())
