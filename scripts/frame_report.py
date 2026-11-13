"""A2: does naming the coordinate conventions move the number, and by more
than a re-draw would?

    python3 scripts/frame_report.py            # S3 and S4
    python3 scripts/frame_report.py S4

THE CLAIM UNDER TEST is that the defect taxonomy is ACTIONABLE. FRAME is 4% of
failing programs at S1 and 47% at S3/S4, and every form of it -- camera frame
used as world, wxyz against xyzw, a 4x4 applied to a 3-vector -- is a
CONVENTION the prompt never states. So stating it should help, if the tag
means what the paper says it means.

WHAT THIS REPORT IS CAREFUL ABOUT, in order of how badly it has gone wrong
before:

1. THE UNIT IS THE CELL, NOT THE SCENE. A cell is ONE greedy program run on 30
   scenes. A prompt change re-draws the program, so a cell going 0 -> 18 is one
   program replaced by another, not 18 independent wins. Pooling scenes gives a
   p-value the design cannot support -- that is how the Phase 11 worker verdict
   was produced and withdrawn. The cell-level sign test comes first here; the
   scene-level count is printed for direction only, labelled.

2. A MOVE MUST CLEAR WHAT A RE-DRAW ALONE DOES. A4 collected five sampled
   draws per cell for exactly this. It is quoted here with its own limit
   stated: A4 ran cube_stack at S3, where every cell is 0/30 at greedy, so it
   bounds re-draw movement AT THE FLOOR. It does NOT bound a cell sitting at
   27/30, and most of A2's movers are such cells on other tasks. Where that is
   the case this says so instead of borrowing the number.

3. SUCCESS RATE IS THE WEAK READOUT. Whether FRAME defects FELL needs the
   failing programs exported, repaired and tagged, exactly as S3/S4 were. This
   prints the rate half and says what it cannot answer.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import mcnemar  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.store import (StoreUnreachable, open_store,  # noqa: E402
                         store_exists)

ARM = "hf:savoji/capx-framehint-seeded"
BASE = "hf:savoji/capx-ab-on-seeded"
#: A4's arm, and the ONLY (task, tier) at which a re-draw bound was measured.
A4_TASK, A4_TIER = "cube_stack", "S3"


def sign_p(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def read(spec: str) -> dict:
    s = open_store(spec)
    out: dict = {}
    for c in s.completed():
        out[(c.model, c.tier, c.task)] = {
            e.get("state_id"): bool(e.get("success")) for e in s.read_cell(c)
            if e.get("state_id") is not None
        }
    return out


def redraw_bound() -> tuple[float, int, str] | None:
    """(max |delta| seen, cells, note) from A4's draws, or None if absent."""
    specs = [f"hf:savoji/capx-nut-sampled-{A4_TASK}-{A4_TIER.lower()}-k{k}"
             for k in range(1, 6)]
    present = [sp for sp in specs if store_exists(sp)]
    if not present:
        return None
    per: dict = {}
    for sp in present:
        s = open_store(sp)
        for c in s.completed():
            if c.task != A4_TASK or c.tier != A4_TIER:
                continue
            k = sum(1 for e in s.read_cell(c) if e.get("success"))
            per.setdefault(c.model, []).append(k)
    if not per:
        return None
    spread = max((max(v) - min(v)) for v in per.values())
    return float(spread), len(per), f"{len(present)} draws"


def main() -> int:
    tiers = sys.argv[1:] or ["S3", "S4"]
    for spec in (ARM, BASE):
        if not store_exists(spec):
            print(f"{spec} does not exist -- nothing collected. Not creating it.")
            return 1
    arm, base = read(ARM), read(BASE)

    bound = redraw_bound()
    print("=" * 76)
    print("A2: THE FRAME-CONVENTION HINT".center(76))
    print("=" * 76)

    all_rows = []
    for tier in tiers:
        keys = [k for k in arm if k[1] == tier and k in base]
        if not keys:
            print(f"\n[{tier}] no paired cells.")
            continue
        up = down = same = 0
        ka = kb = na = 0
        rows = []
        ph: dict = {}
        pl: dict = {}
        for k in sorted(keys):
            hi, lo = arm[k], base[k]
            shared = sorted(set(hi) & set(lo))
            if not shared:
                continue
            a = sum(1 for s in shared if hi[s])
            b = sum(1 for s in shared if lo[s])
            ka += a
            kb += b
            na += len(shared)
            for s in shared:
                ph[(k[0], k[2], s)] = hi[s]
                pl[(k[0], k[2], s)] = lo[s]
            if a > b:
                up += 1
            elif a < b:
                down += 1
            else:
                same += 1
            if a != b:
                rows.append((k[0], k[2], b, a))
        print(f"\n[{tier}]  {up + down + same} paired cells   "
              f"baseline {kb}/{na}   hint {ka}/{na}   net {ka - kb:+d}")
        print(f"       cells up {up}   down {down}   unchanged {same}   "
              f"sign test p = {sign_p(up, down):.4g}")
        mc = mcnemar(ph, pl)
        print(f"       scene-level (NOT independent): hint only {mc.b}, "
              f"baseline only {mc.c}, p = {mc.p_value:.4g}")
        if rows:
            print(f"\n       {'model':<20}{'task':<14}{'base':>6}{'hint':>6}"
                  f"{'delta':>7}  clears a re-draw?")
            print("       " + "-" * 66)
            for model, task, b, a in sorted(rows, key=lambda r: r[3] - r[2]):
                d = a - b
                if bound and (task, tier) == (A4_TASK, A4_TIER):
                    verdict = "YES" if abs(d) > bound[0] else "no"
                elif bound:
                    verdict = "UNBOUNDED (A4 ran %s/%s)" % (A4_TASK, A4_TIER)
                else:
                    verdict = "no A4 data"
                print(f"       {model:<20}{task:<14}{b:>6}{a:>6}{d:>+7}  "
                      f"{verdict}")
            all_rows += [(tier, *r) for r in rows]

    print("\n" + "=" * 76)
    print("THE RE-DRAW BOUND".center(76))
    print("=" * 76)
    if not bound:
        print("\n  A4's stores are absent. Nothing here bounds a program swap,")
        print("  so no cell above is a result yet.")
    else:
        spread, ncells, note = bound
        print(f"\n  A4, {A4_TASK} at {A4_TIER}, {note}, {ncells} cells:")
        print(f"  the widest spread any cell shows across draws is "
              f"{spread:.0f} scenes of 30.")
        print( "\n  READ THE LIMIT. Every greedy cell of that arm is 0/30, so")
        print( "  this bounds how far a re-draw moves a cell AT THE FLOOR. It")
        print( "  does NOT bound a cell at 27/30, and it says nothing about")
        print(f"  tasks other than {A4_TASK}.")
        other = [r for r in all_rows if (r[2], r[0]) != (A4_TASK, A4_TIER)]
        if other:
            print(f"\n  {len(other)} of {len(all_rows)} movers are outside "
                  f"that arm, including the largest:")
            for tier, model, task, b, a in sorted(
                    other, key=lambda r: -abs(r[4] - r[3]))[:4]:
                print(f"    {tier} {model} {task}  {b} -> {a}")
            print( "  To bound those, re-run A4 on a task and tier where the")
            print( "  baseline is NOT at the floor:")
            print( "    OPENCAP_SAMPLED_TASK=cube_lifting OPENCAP_SAMPLED_TIER=S3 \\")
            print( "      bash scripts/run_sampled_arm.sh")

    print("\n" + "=" * 76)
    print("WHAT THIS CANNOT ANSWER".center(76))
    print("=" * 76)
    print("\n  Whether FRAME DEFECTS FELL. That is the claim -- the rate is a")
    print( "  proxy for it. It needs A2's failing programs exported, repaired")
    print( "  to ceiling, minimised and tagged, exactly as S3/S4 were. A null")
    print( "  on the rate with FRAME unchanged, and a null on the rate with")
    print( "  FRAME halved, are different papers.")
    return 0


def _guarded(fn):
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
