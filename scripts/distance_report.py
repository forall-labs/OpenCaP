"""Distance-to-solution per program: how far was it from working?

    python3 scripts/distance_report.py

Reads results/failures.json (distinct failed programs) and results/repairs/*.json
(the minimal repairs), and reports the normalised edit distance between them.

UNVERIFIED until each repair has been re-executed against the same 30 scene
seeds. `verified_successes` on a RepairResult is what promotes a distance from
a judgement to a measurement; this report prints the judgement and says so.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.models import BY_KEY  # noqa: E402
from opencap.repair import (edit_ops, is_parseable,  # noqa: E402
                          normalised_edit_distance)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    fails = {x["pid"]: x for x in json.loads((root / "results/failures.json").read_text())}
    reps: dict = {}
    for f in sorted(glob.glob(str(root / "results/repairs/*.json"))):
        reps.update(json.loads(Path(f).read_text()))

    rows, bad = [], []
    for pid, rep in reps.items():
        f = fails[pid]
        m = f["models"][0]
        if rep["code"] is None:
            rows.append((f["task"], 0.0, m, f["n_episodes"],
                         ",".join(f["root_causes"])[:15], 0, 0, 0, rep["level"]))
            continue
        if not is_parseable(rep["code"]):
            bad.append(pid)
            continue
        d = normalised_edit_distance(f["code"], rep["code"])
        o = edit_ops(f["code"], rep["code"])
        rows.append((f["task"], d, m, f["n_episodes"],
                     ",".join(f["root_causes"])[:15],
                     o["added"], o["removed"], o["changed"], rep["level"]))
    if bad:
        print(f"UNPARSEABLE REPAIRS (fix before trusting anything): {bad}")
        return 1

    print(f"{'task':<13}{'model':<19}{'sz':>6}{'n':>4}  {'cause':<16}"
          f"{'DIST':>6}{'+':>4}{'-':>4}{'~':>4}")
    print("-" * 82)
    for t, d, m, n, c, a, rm, ch, lv in sorted(rows):
        sz = f"{BY_KEY[m].params_b:g}B" if m in BY_KEY else "?"
        tag = " *" if lv == "L4_scene_limited" else ""
        print(f"{t:<13}{m:<19}{sz:>6}{n:>4}  {c:<16}{d:>6.3f}"
              f"{a:>4}{rm:>4}{ch:>4}{tag}")
    n_scene = sum(1 for r in rows if r[8] == "L4_scene_limited")
    print(f"\n* = no defect found ({n_scene} programs); the cell's majority passed. "
          "Distance is 0 by finding, not by repair.")
    print("\nSTATUS: UNVERIFIED. No repair has been re-executed yet. These are "
          "judgements about what each program needs, not measurements of what "
          "recovers it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
