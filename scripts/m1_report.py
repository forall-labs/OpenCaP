"""A3: did the multi-turn loop engage, and what did it buy?

    python3 scripts/m1_report.py

TWO THINGS HAVE TO BE TRUE BEFORE THE SUCCESS NUMBERS MEAN ANYTHING, and the
handoff got the first one wrong.

1. THE LOOP MUST HAVE ENGAGED. If every episode is one turn, M1 is a re-run of
   its base tier with a longer prompt and nothing about feedback was tested.
   HANDOFF said to "report turns_used above all". `turns_used` is declared in
   opencap/schema.py with a default of 0 and was assigned by NOTHING, so it reads
   0 in all 2490 collected M1 episodes. Followed literally, that kills A3 on a
   field nobody ever wrote.

   The engagement signal is `len(turns)`, which is populated -- and it is >1 in
   785 of 2490 episodes. This script reads that and refuses to read
   `turns_used` on any episode whose turn list contradicts it, because the
   stored 0 is not a measurement.

2. THE COMPARATOR MUST MOVE ONE AXIS. Every `_multiturn` config is
   `privileged: false` on FrankaControlApi -- the S2 API. So

       M1 - S1   moves privilege AND turns
       M1 - S3   moves API surface AND turns   (what opencap/analysis.py pairs)
       M1 - S2   moves turns                   <-- the contrast

   S2 is collected by scripts/run_s2_arm.sh. Until it exists this prints the
   two-axis contrasts, LABELLED as such, and refuses to call either one an
   interaction gain.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import mcnemar, wilson  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.store import (StoreUnreachable, open_store,  # noqa: E402
                         store_exists)

M1 = "hf:savoji/capx-m1-seeded"
LADDER = "hf:savoji/capx-ab-on-seeded"
BANDS = (("<=2B", 0, 2), ("3-4B", 2, 4), ("7-9B", 4, 9), (">=12B", 9, 1e9))


def band_of(model: str) -> str:
    p = BY_KEY[model].params_b if model in BY_KEY else 0
    for name, lo, hi in BANDS:
        if lo < p <= hi:
            return name
    return BANDS[0][0]


def turns_of(ep: dict) -> int:
    """How many turns this episode actually took.

    `turns_used` is a dead field (always 0). The turn list is the measurement.
    """
    t = ep.get("turns")
    return len(t) if isinstance(t, list) else 0


def read(spec: str, tier: str | None = None) -> dict:
    """{(model, tier, task): {state_id: episode}}"""
    out: dict = {}
    s = open_store(spec)
    for c in s.completed():
        if tier and c.tier != tier:
            continue
        out[(c.model, c.tier, c.task)] = {
            e.get("state_id"): e for e in s.read_cell(c)
            if e.get("state_id") is not None
        }
    return out


def engagement(m1: dict) -> None:
    print("=" * 76)
    print("DID THE LOOP ENGAGE?".center(76))
    print("=" * 76)
    hist: Counter = Counter()
    dead = 0
    by_band: dict = defaultdict(lambda: [0, 0])     # [multi-turn eps, eps]
    for (model, _t, _task), eps in m1.items():
        for e in eps.values():
            n = turns_of(e)
            hist[n] += 1
            if e.get("turns_used") not in (None, n):
                dead += 1
            b = by_band[band_of(model)]
            b[0] += 1 if n > 1 else 0
            b[1] += 1
    total = sum(hist.values())
    multi = sum(v for k, v in hist.items() if k > 1)
    print(f"\n  episodes {total}   with MORE THAN ONE turn {multi} "
          f"({multi / max(total, 1):.1%})")
    print("  turns per episode: " + ", ".join(
        f"{k}:{v}" for k, v in sorted(hist.items())))
    if dead:
        print(f"\n  the stored `turns_used` field disagrees with the turn list "
              f"on {dead} of {total} episodes.")
        print( "  It is a DEAD FIELD -- declared with a default of 0 and never")
        print( "  assigned. Read len(turns). run_cell.py now derives it from")
        print( "  the same read so future runs cannot disagree.")
    print(f"\n  {'band':<8}{'episodes':>10}{'multi-turn':>12}{'share':>9}")
    print("  " + "-" * 37)
    for name, _, _ in BANDS:
        k, n = by_band.get(name, [0, 0])
        if n:
            print(f"  {name:<8}{n:>10}{k:>12}{k / n:>8.0%}")
    if multi == 0:
        print("\n  THE LOOP NEVER ENGAGED. A3 is a re-run of its base tier and")
        print( "  the success numbers test nothing about feedback. Stop here.")
    else:
        print("\n  The loop engaged. The success numbers below are about")
        print( "  feedback -- for the share of episodes that took a second turn.")


def contrast(m1: dict, lad: dict, tier: str, label: str) -> None:
    """Paired cell-by-cell M1 vs one tier of the ladder."""
    print("\n" + "=" * 76)
    print(f"M1 vs {tier}  --  {label}".center(76))
    print("=" * 76)
    rows = []
    pooled_hi: dict = {}
    pooled_lo: dict = {}
    up = down = same = 0
    for (model, _t, task), hi in sorted(m1.items()):
        lo = lad.get((model, tier, task))
        if not lo:
            continue
        shared = sorted(set(hi) & set(lo))
        if not shared:
            continue
        kh = sum(1 for st in shared if hi[st].get("success"))
        kl = sum(1 for st in shared if lo[st].get("success"))
        for st in shared:
            pooled_hi[(model, task, st)] = bool(hi[st].get("success"))
            pooled_lo[(model, task, st)] = bool(lo[st].get("success"))
        if kh > kl:
            up += 1
        elif kh < kl:
            down += 1
        else:
            same += 1
        if kh != kl:
            rows.append((model, task, kl, kh, len(shared)))
    n_paired = up + down + same
    print(f"\n  paired cells {n_paired}   M1 higher {up}   {tier} higher {down}"
          f"   equal {same}")
    if not rows:
        print("  no cell differs.")
    else:
        print(f"\n  {'model':<20}{'task':<14}{tier:>8}{'M1':>8}{'of':>6}")
        print("  " + "-" * 56)
        for model, task, kl, kh, n in sorted(rows, key=lambda r: r[3] - r[2]):
            print(f"  {model:<20}{task:<14}{kl:>8}{kh:>8}{n:>6}")

    # THE UNIT OF INDEPENDENCE IS THE CELL, NOT THE SCENE. One cell is one
    # program run on 30 scenes, so 30 scene outcomes are 30 looks at a single
    # draw. A scene-level McNemar over pooled cells therefore reports a
    # precision the design does not have -- the same mis-specification that
    # produced, and then withdrew, the Phase 11 worker verdict. Both are
    # printed, with the cell-level sign test first because it is the honest
    # one.
    if up + down:
        import math
        k = min(up, down)
        n = up + down
        sign_p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        print(f"\n  CELL-LEVEL sign test over the {n} cells that moved: "
              f"p = {sign_p:.4g}")
        print( "  This is the test the design supports: one cell, one draw.")
    mc = mcnemar(pooled_hi, pooled_lo)
    print(f"\n  scene-level, pooled (NOT independent): M1 only {mc.b}, "
          f"{tier} only {mc.c}, exact p = {mc.p_value:.4g}")
    print( "  NOT an independent-trials p. 30 scenes share one program, so")
    print( "  this overstates precision; it is here to show direction only.")
    print(f"\n  TWO AXES MOVE HERE -- turns AND "
          f"{'privilege' if tier == 'S1' else 'API surface'}. This is not an")
    print( "  interaction gain and must not be quoted as one. The one-axis")
    print( "  contrast is M1 - S2; run scripts/run_s2_arm.sh.")


def main() -> int:
    for spec in (M1, LADDER):
        if not store_exists(spec):
            print(f"{spec} does not exist -- nothing collected. Not creating it.")
            return 1
    m1 = read(M1)
    lad = read(LADDER)
    tiers = {t for (_m, t, _k) in lad}

    print(f"M1 cells {len(m1)}   ladder tiers present: {', '.join(sorted(tiers))}")
    engagement(m1)

    if "S2" in tiers:
        contrast(m1, lad, "S2", "THE ONE-AXIS CONTRAST: turns only")
        print("\n  S2 is collected, so this is the reportable interaction gain.")
    else:
        print("\n" + "!" * 76)
        print("  S2 IS NOT COLLECTED. The one-axis comparator for M1 does not")
        print("  exist, so nothing below is an interaction gain. Both contrasts")
        print("  move two axes. Collect it: bash scripts/run_s2_arm.sh")
        print("!" * 76)
    for tier, label in (("S1", "two axes: turns AND privilege"),
                        ("S3", "two axes: turns AND API surface")):
        if tier in tiers:
            contrast(m1, lad, tier, label)
    return 0


def _guarded(fn):
    """An unreachable hub must never print as a null result."""
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
