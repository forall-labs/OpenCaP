"""Known-good programs per task, taken from cells that actually passed.

A repair is judged by how little it changes the model's OWN program, so the
reference is used to understand what "working" requires -- never as a target to
converge on. Rewriting a failed program into the reference would make every
distance equal to distance-from-reference and measure nothing.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.repair import program_of, strip_markers  # noqa: E402
from opencap.store import open_store  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1])
    best = defaultdict(list)
    for spec in sys.argv[2:]:
        s = open_store(spec)
        for cell in s.completed():
            eps = list(s.read_cell(cell))
            n_ok = sum(1 for e in eps if e.get("success"))
            if not n_ok:
                continue
            ok = next(e for e in eps if e.get("success"))
            best[cell.task].append({
                "model": cell.model, "tier": cell.tier,
                "successes": n_ok, "n": len(eps),
                "code": strip_markers(program_of(ok)),
            })
    recs = {}
    for task, cands in best.items():
        cands.sort(key=lambda c: -c["successes"])
        recs[task] = cands[0]
        print(f"{task:<14}best={cands[0]['model']} "
              f"{cands[0]['successes']}/{cands[0]['n']}  "
              f"({len(cands)} passing cells available)")
    out.write_text(json.dumps(recs, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
