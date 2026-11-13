"""Does the abstraction level change WHAT fails, or only HOW MUCH?

    python3 scripts/tier_report.py [S1 S3 S4]

This is the readout of the scaffold axis, and it is deliberately two reports:

  RATE   success per band per tier. The obvious number, and the least
         interesting one -- CaP-X already publishes an abstraction gap.

  MIX    the defect-tag distribution AMONG FAILING PROGRAMS, per tier. This is
         the one nobody else can compute, because it needs a verified minimal
         repair behind every tag. If S1 failures are software integration and
         S4 failures are geometry, a single success rate has been conflating
         two different problems with two different fixes.

READ THE MIX CONDITIONALLY ON FAILING. If S4 floors the ladder, its failure
count rises and every tag count rises with it; the quantity of interest is the
SHARE of failing programs carrying each tag, not the count. A tier whose
failures are the same mix, only more of them, is a different finding from a
tier whose failures are a different kind.

Tagging a new tier is not automatic: it needs failing programs exported,
repaired to ceiling and minimised, exactly as S1 was. Until that exists this
prints the rate half and says so rather than implying the mix is unchanged.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402
from opencap.editclass import FAMILY  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.store import (StoreUnreachable, open_store,  # noqa: E402
                         store_exists)

CANON = "hf:savoji/capx-ab-on-seeded"
BANDS = (("<=2B", 0, 2), ("3-4B", 2, 4), ("7-9B", 4, 9), (">=12B", 9, 1e9))


def band_of(model: str) -> str:
    p = BY_KEY[model].params_b if model in BY_KEY else 0
    for name, lo, hi in BANDS:
        if lo < p <= hi:
            return name
    return BANDS[0][0]


def tags_for(tier: str) -> list | None:
    """Per-tier tagging, if it has been produced."""
    for name in (f"tags_{tier.lower()}.json",
                 "tags_seeded.json" if tier == "S1" else None):
        if not name:
            continue
        p = ROOT / "results" / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def main() -> int:
    tiers = sys.argv[1:] or ["S1", "S3", "S4"]
    # store_exists first: opening an HF store CREATES it, so a report naming a
    # store nobody collected brings an empty repo into being -- and an empty
    # repo reads like a collected zero to the next person.
    if not store_exists(CANON):
        print(f"{CANON} does not exist -- nothing collected. Not creating it.")
        return 1
    try:
        s = open_store(CANON)
        cells = list(s.completed())
    except Exception as e:
        print(f"cannot read {CANON}: {type(e).__name__}: {e}")
        return 1

    present = {c.tier for c in cells}
    missing = [t for t in tiers if t not in present]
    if missing:
        print(f"[tiers] not collected yet: {', '.join(missing)} "
              f"(store holds {', '.join(sorted(present))})")
    tiers = [t for t in tiers if t in present]
    if not tiers:
        print("[tiers] nothing to report")
        return 1

    # --- RATE ---------------------------------------------------------------
    print("=" * 74)
    print("SUCCESS RATE BY BAND AND TIER".center(74))
    print("=" * 74)
    rate: dict = defaultdict(lambda: [0, 0])
    for c in cells:
        if c.tier not in tiers:
            continue
        eps = list(s.read_cell(c))
        b = band_of(c.model)
        rate[(c.tier, b)][0] += sum(1 for e in eps if e.get("success"))
        rate[(c.tier, b)][1] += len(eps)
    print(f"\n{'band':<8}" + "".join(f"{t:>22}" for t in tiers))
    print("-" * 74)
    for name, _, _ in BANDS:
        row = ""
        for t in tiers:
            k, n = rate.get((t, name), [0, 0])
            if not n:
                row += f"{'--':>22}"
                continue
            iv = wilson(k, n)
            row += f"{f'{k}/{n} {iv.point:.3f}':>22}"
        print(f"{name:<8}{row}")

    # --- MIX ----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("DEFECT MIX AMONG FAILING PROGRAMS".center(74))
    print("=" * 74)
    have = {t: tags_for(t) for t in tiers}
    untagged = [t for t, v in have.items() if not v]
    if untagged:
        print(f"\n  NOT TAGGED: {', '.join(untagged)}. A tier's mix needs its")
        print( "  failing programs exported, repaired to ceiling and minimised,")
        print( "  exactly as S1 was. Until that happens this is a rate result")
        print( "  only -- do NOT report the mix as unchanged, it is unmeasured.")
    tagged = [t for t in tiers if have[t]]
    if not tagged:
        return 0

    print(f"\n{'tag':<22}{'family':<12}" + "".join(f"{t:>12}" for t in tagged))
    print("-" * 74)
    alltags: set = set()
    share: dict = {}
    for t in tagged:
        recs = have[t]
        n = len(recs) or 1
        c = Counter()
        for r in recs:
            for tag in set(r["tags"]):
                c[tag] += 1
        share[t] = {k: v / n for k, v in c.items()}
        alltags |= set(c)
    for tag in sorted(alltags, key=lambda x: -max(share[t].get(x, 0) for t in tagged)):
        row = "".join(f"{share[t].get(tag, 0):>11.0%} " for t in tagged)
        print(f"{tag:<22}{FAMILY.get(tag, '?'):<12}{row}")

    print(f"\n{'FAMILY':<22}{'':<12}" + "".join(f"{t:>12}" for t in tagged))
    print("-" * 74)
    fams: dict = {}
    for t in tagged:
        recs = have[t]
        n = len(recs) or 1
        c = Counter()
        for r in recs:
            for f in set(r["families"]):
                c[f] += 1
        fams[t] = {k: v / n for k, v in c.items()}
    for fam in sorted({f for t in tagged for f in fams[t]},
                      key=lambda x: -max(fams[t].get(x, 0) for t in tagged)):
        row = "".join(f"{fams[t].get(fam, 0):>11.0%} " for t in tagged)
        print(f"{fam:<22}{'':<12}{row}")
    print(f"\n  n programs: " + ", ".join(f"{t} {len(have[t])}" for t in tagged))
    print("  Shares are of FAILING programs, so a tier with more failures does")
    print("  not automatically show larger shares. A mix that moves is the")
    print("  result; a mix that holds while the rate falls is a different one.")
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
