"""Settle the sweep-order question with repeated measures.

Row-major vs column-major rastering on spill_wipe has been claimed, retracted,
re-claimed and re-retracted in this project, every time on single runs. Three
accidental replicates then showed +-5.3 scenes of run-to-run noise on IDENTICAL
code, which is why single runs could never settle it.

This runs ONE program N times in each order and reports both distributions.

    python3 scripts/sweep_order_test.py [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from scripts.verify_repairs import run_one  # noqa: E402

ROW = ("for y in np.linspace(y_min, y_max, 10):\n"
       "    for x in np.linspace(x_min, x_max, 10):")
COL = ("for x in np.linspace(x_min, x_max, 10):\n"
       "    for y in np.linspace(y_min, y_max, 10):")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capx", default="/content/cap-x")
    ap.add_argument("--scratch", default="/content/sweeptest")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="/content/sweeptest/results.json")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    reps = json.loads((root / "results/repairs/spill_wipe.json").read_text())
    col_code = reps["6a98c7c543d8"]["code"]          # verified 30/30, column-major
    assert COL in col_code, "column-major raster not found in reference repair"
    row_code = col_code.replace(COL, ROW)
    assert row_code != col_code

    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for order, code in (("column", col_code), ("row", row_code)):
        for i in range(a.repeats):
            pid = f"sweep_{order}_{i}"
            r = run_one(Path(a.capx), "spill_wipe", pid, code, 30,
                        a.workers, scratch)
            r["order"] = order
            s = r.get("successes")
            print(f"[{order:>6} {i}] {s}/{r.get('n')}  ({r.get('wall')}s)")
            out.append(r)
            Path(a.out).write_text(json.dumps(out, indent=2))

    for order in ("column", "row"):
        xs = sorted(r["successes"] for r in out
                    if r["order"] == order and r.get("ok"))
        if xs:
            print(f"{order:>6}-major: {xs}  mean={sum(xs)/len(xs):.1f}")
    print("\nSame program, same seeds, only the loop order differs. If the two "
          "distributions overlap, the effect is not separable from the "
          "+-5.3 scene noise measured on identical code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
