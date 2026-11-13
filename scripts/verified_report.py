"""The deliverable: verified distance-to-solution per program.

    python3 scripts/verified_report.py [verify_results.json]

Merges three things:
  results/failures.json        the distinct failed programs
  results/repairs/*.json       the minimal repair for each
  verify results (arg)         what the environment said when the repair ran

A distance is reported as MEASURED only where the repair reached the task
ceiling. Anything short is listed separately as still-iterating, with its score,
because a repair that recovers 25 of 30 has not established the distance to a
working program -- it has established a lower bound on it.
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
from opencap.repair import edit_ops, normalised_edit_distance  # noqa: E402

#: spill_wipe was set to 29 from coder-7b's single observation; a repaired
#: qwen3-14b then scored 30/30, so the task is fully solvable and 29 was one
#: noisy draw, not a ceiling.
TASK_CEILING = {"cube_lifting": 30, "cube_stack": 30, "spill_wipe": 30}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    res_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results/verify_results.json"
    fails = {x["pid"]: x for x in json.loads((root / "results/failures.json").read_text())}
    reps: dict = {}
    for f in sorted(glob.glob(str(root / "results/repairs/*.json"))):
        reps.update(json.loads(Path(f).read_text()))
    ver = {}
    if res_path.exists():
        ver = {r["pid"]: r for r in json.loads(res_path.read_text())}

    passed, short, unrun, scene = [], [], [], []
    for pid, rep in reps.items():
        f = fails[pid]
        m = f["models"][0]
        task = f["task"]
        ceil = TASK_CEILING.get(task, 30)
        if rep["code"] is None:
            scene.append((task, m, f["n_episodes"]))
            continue
        d = normalised_edit_distance(f["code"], rep["code"])
        o = edit_ops(f["code"], rep["code"])
        v = ver.get(pid)
        if not v or not v.get("ok"):
            unrun.append((task, d, m, o))
        elif v["successes"] >= ceil:
            passed.append((task, d, m, o, v["successes"], v["n"]))
        else:
            short.append((task, d, m, o, v["successes"], v["n"], ceil))

    def line(t, d, m, o, extra=""):
        sz = f"{BY_KEY[m].params_b:g}B" if m in BY_KEY else "?"
        return (f"{t:<13}{m:<19}{sz:>6}{d:>7.3f}"
                f"{o['added']:>4}{o['removed']:>4}{o['changed']:>4}   {extra}")

    print("=" * 78)
    print("MEASURED distance to a working program (repair verified at ceiling)")
    print("=" * 78)
    print(f"{'task':<13}{'model':<19}{'sz':>6}{'DIST':>7}{'+':>4}{'-':>4}{'~':>4}   verified")
    for t, d, m, o, s, n in sorted(passed):
        print(line(t, d, m, o, f"{s}/{n}"))
    if not passed:
        print("  (none yet)")

    if short:
        print("\n" + "-" * 78)
        print("STILL ITERATING -- repair recovered some scenes but not the ceiling.")
        print("These distances are LOWER BOUNDS, not measurements.")
        for t, d, m, o, s, n, c in sorted(short):
            print(line(t, d, m, o, f"{s}/{n}  (need {c})"))

    if unrun:
        print("\n" + "-" * 78)
        print("NOT YET VERIFIED -- judgement only, no environment result.")
        for t, d, m, o in sorted(unrun):
            print(line(t, d, m, o))

    if scene:
        print("\n" + "-" * 78)
        print("NO DEFECT FOUND -- cell majority passed; tail is scene-limited.")
        for t, m, n in sorted(scene):
            print(f"{t:<13}{m:<19}{n:>4} failing episodes of 30")

    tot = len(passed) + len(short) + len(unrun)
    print(f"\n{len(passed)}/{tot} repairs verified at ceiling; "
          f"{len(short)} iterating; {len(unrun)} unrun; {len(scene)} no-defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
