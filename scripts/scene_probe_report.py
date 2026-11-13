"""Read results/scene_probe.json: is the scene a defect, or a coin flip?

    python3 scripts/scene_probe_report.py [results/scene_probe.json]

Reports, per program, how often the scene passed across N draws of the ORIGINAL
and of the MINIMAL repair, with a Wilson interval on each rate -- because at
N=8 a point estimate of 6/8 and one of 8/8 are not distinguishable and printing
them bare invites reading a difference that is not there.

The control rows are the whole argument. If untagged programs flip the scene at
the same rate, then "the repair fixed scene 13" says nothing about the repair.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402


def rate(recs):
    got = [r for r in recs if r.get("ok") and "scene_success" in r]
    if not got:
        return None
    k = sum(1 for r in got if r["scene_success"])
    iv = wilson(k, len(got))
    return k, len(got), iv.lo, iv.hi


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/scene_probe.json"
    if not path.exists():
        print(f"{path} missing -- run scripts/run_scene_probe.py first")
        return 1
    raw = json.loads(path.read_text())
    scene = raw[0]["scene"] if raw else "?"
    task = raw[0]["task"] if raw else "?"
    by = defaultdict(lambda: defaultdict(list))
    role = {}
    model = {}
    for r in raw:
        by[r["pid"]][r["version"]].append(r)
        role[r["pid"]] = r["role"]
        model[r["pid"]] = r["model"]

    bad = [r for r in raw if not r.get("ok")]
    print("=" * 74)
    print(f"SCENE PROBE -- {task} scene {scene}".center(74))
    print("=" * 74)
    print(f"\n{len(raw)} runs, {len(by)} programs, {len(bad)} that did not complete")

    print(f"\n{'model':<20}{'role':<9}{'original':>22}{'repair':>22}")
    print("-" * 74)
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for pid in sorted(by, key=lambda p: (role[p] != "tagged", model[p])):
        cells = []
        for version in ("original", "repair"):
            r = rate(by[pid].get(version, []))
            if r is None:
                cells.append(f"{'--':>22}")
                continue
            k, n, lo, hi = r
            agg[role[pid]][version][0] += k
            agg[role[pid]][version][1] += n
            cells.append(f"{k}/{n} [{lo:.2f},{hi:.2f}]".rjust(22))
        print(f"{model[pid]:<20}{role[pid]:<9}{cells[0]}{cells[1]}")

    print(f"\n{'POOLED':<20}{'':9}{'original':>22}{'repair':>22}")
    print("-" * 74)
    for r in ("tagged", "control"):
        if r not in agg:
            continue
        cells = []
        for version in ("original", "repair"):
            k, n = agg[r][version]
            if not n:
                cells.append(f"{'--':>22}")
                continue
            iv = wilson(k, n)
            lo, hi = iv.lo, iv.hi
            cells.append(f"{k}/{n} [{lo:.2f},{hi:.2f}]".rjust(22))
        print(f"{'':<20}{r:<9}{cells[0]}{cells[1]}")

    print(f"\nHOW TO READ IT (scene {scene} pass rate)")
    print("  original ~0, repair ~1, controls ~1   the defect is real and the")
    print("                                        repair fixes it")
    print("  everything near 0.5                   the scene is a coin flip and")
    print("                                        the tag is an artifact")
    print("  controls also mid-range               the scene is unstable for")
    print("                                        reasons unrelated to the tag;")
    print("                                        the tagged rows cannot be read")
    print("                                        as evidence about the repair")
    print("\nIntervals are Wilson. At these N a gap that does not separate the")
    print("intervals is not a result -- this project has published four of those.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
