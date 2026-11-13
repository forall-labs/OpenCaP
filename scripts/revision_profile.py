"""Does the multi-turn scaffold actually buy anything, per model scale?

CaP-X grants MAX_TURNS turns with execution feedback between them. That
affordance is only worth its cost if models use it -- and "used" means the
program changed, not that another turn was spent.

Three ways it can be wasted, and they need different remedies:
  never re-entered   one turn, then stop -- the model never asks for feedback
  spent verbatim     more turns, identical program -- feedback changed nothing
  spent revising     the affordance working as designed

    python3 scripts/revision_profile.py hf:savoji/capx-ab-off [hf:...on]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.models import BY_KEY  # noqa: E402
from opencap.store import open_store  # noqa: E402


def turn_codes(episode) -> list[str]:
    codes = [(t.get("code") or t.get("content") or "").strip()
             for t in (episode.get("turns") or [])]
    return [c for c in codes if c]


def collect(specs, tier="S1"):
    agg = defaultdict(lambda: defaultdict(int))
    for spec in specs:
        store = open_store(spec)
        for cell in store.completed():
            if tier and cell.tier != tier:
                continue
            for e in store.read_cell(cell):
                codes = turn_codes(e)
                a = agg[cell.model]
                a["eps"] += 1
                a["turns"] += len(codes)
                if len(codes) >= 2:
                    a["multi"] += 1
                    if len(set(codes)) == 1:
                        a["repeat"] += 1
                    elif e.get("success"):
                        a["revised_won"] += 1
    return agg


def main() -> int:
    agg = collect(sys.argv[1:])
    if not agg:
        print("no cells found")
        return 1
    print(f"{'model':<12}{'eps':>6}{'mean turns':>12}{'multi-turn':>12}"
          f"{'verbatim':>10}{'revised':>9}{'revised+won':>13}")
    print("-" * 74)
    for m in sorted(agg, key=lambda x: BY_KEY[x].params_b if x in BY_KEY else 0):
        a = agg[m]
        multi, rep = a["multi"], a["repeat"]
        revised = multi - rep
        print(f"{m:<12}{a['eps']:>6}{a['turns']/max(a['eps'],1):>12.2f}"
              f"{multi:>12}{rep:>10}{revised:>9}{a['revised_won']:>13}")
    print("\nA rung with 0 multi-turn episodes never re-entered the loop; a rung "
          "whose multi-turn episodes are all verbatim spent the turns for "
          "nothing. Neither is a longer-horizon-planning problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
