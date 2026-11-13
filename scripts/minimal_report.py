"""Iterated vs minimised repair distance, with verification status for both.

    python3 scripts/minimal_report.py [minimal_verify_results.json]

The iterated repairs reached ceiling but drifted from the model's own code --
renamed variables, deleted merely-unused lines, rewrote blocks where one edit
would do. Distance is only meaningful as the MINIMUM edit that works, so each
was re-minimised against the initial code, the repair, and the reference.

A minimised version is adopted ONLY if it independently reaches ceiling. Where
it does not, the iterated version stands and the minimisation is recorded as
too aggressive -- the smaller number would be a claim we cannot support.
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
from opencap.repair import normalised_edit_distance  # noqa: E402

CEIL = {"cube_lifting": 30, "cube_stack": 30, "spill_wipe": 30}


def load(pattern: str) -> dict:
    out: dict = {}
    for f in sorted(glob.glob(pattern)):
        out.update(json.loads(Path(f).read_text()))
    return out


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    fails = {x["pid"]: x for x in
             json.loads((root / "results/failures.json").read_text())}
    iterated = load(str(root / "results/repairs/*.json"))
    minimal = load(str(root / "results/minimal/*.json"))

    res_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/minimal_verify.json"
    ver = {}
    if res_path.exists():
        ver = {r["pid"]: r for r in json.loads(res_path.read_text())}

    print(f"{'task':<13}{'model':<19}{'iterated':>9}{'minimal':>9}"
          f"{'saved':>8}{'verified':>10}")
    print("-" * 70)
    adopted, rejected, unrun = [], [], []
    for pid, m in sorted(minimal.items(),
                         key=lambda kv: (fails[kv[0]]["task"], kv[0])):
        f = fails[pid]
        orig = f["code"]
        di = normalised_edit_distance(orig, iterated[pid]["code"])
        dm = normalised_edit_distance(orig, m["code"])
        v = ver.get(pid)
        ceil = CEIL[f["task"]]
        if not v or not v.get("ok"):
            status, bucket = "—", unrun
        elif v["successes"] >= ceil:
            status, bucket = f"{v['successes']}/{v['n']} ✓", adopted
        else:
            status, bucket = f"{v['successes']}/{v['n']} ✗", rejected
        bucket.append((pid, di, dm))
        print(f"{f['task']:<13}{f['models'][0]:<19}{di:>9.3f}{dm:>9.3f}"
              f"{dm - di:>+8.3f}{status:>10}")

    print("-" * 70)
    if adopted:
        gi = sum(d for _, d, _ in adopted) / len(adopted)
        gm = sum(d for _, _, d in adopted) / len(adopted)
        print(f"adopted  {len(adopted):>2}  mean {gi:.3f} -> {gm:.3f} "
              f"({(gm - gi) / gi * 100:+.0f}%)")
    if rejected:
        print(f"rejected {len(rejected):>2}  minimisation too aggressive; the "
              f"iterated distance stands for these")
        for pid, di, dm in rejected:
            print(f"     {fails[pid]['models'][0]:<19}{fails[pid]['task']:<13}"
                  f"keeps {di:.3f}, not {dm:.3f}")
    if unrun:
        print(f"unrun    {len(unrun):>2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
