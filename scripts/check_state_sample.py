"""Is our frozen 30-state sample representative, or did we freeze an easy draw?

Freezing states rescued the paired design, but it also locks the whole study
onto ONE sample of scenes. If those 30 seeds happen to be unusually easy or
hard, every cell inherits that bias and no amount of statistics recovers it.

Cheap check: run the deterministic human oracle on the frozen seeds (1..30)
and on a much larger set (1..N), and compare. The oracle is a fixed program,
so any difference in success rate is a property of the SCENES, not the policy.

    python3 scripts/check_state_sample.py /content/cap-x --task cube_stack --large 200
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402
from opencap.capx_bridge import CORE_TASKS, oracle_config_path  # noqa: E402
from opencap.config import STATES_PER_TASK  # noqa: E402
from opencap.results import cross_check, parse_trials, resolved_output_dir  # noqa: E402

ORACLE_MODEL = "google/gemini-3.1-pro-preview"


def workers() -> int:
    return max(1, min(12, os.cpu_count() or 2))


def run(capx: Path, task: str, trials: int, tag: str):
    out = Path(f"/content/outputs/sample_{tag}_{task}")
    cmd = ["uv", "run", "capx/envs/launch.py",
           "--config-path", oracle_config_path(task, privileged=True),
           "--total-trials", str(trials), "--num-workers", str(workers()),
           "--record-video", "False", "--output-dir", str(out)]
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    p = subprocess.run(cmd, cwd=capx, env=env, capture_output=True,
                       text=True, timeout=7200)
    if p.returncode != 0:
        print(f"  launch failed: {p.stderr[-200:]}")
        return None
    d = resolved_output_dir(out, ORACLE_MODEL)
    ok, msg = cross_check(d)
    if not ok:
        print(f"  cross-check failed: {msg}")
        return None
    return parse_trials(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capx", nargs="?", default="/content/cap-x")
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--large", type=int, default=200)
    args = ap.parse_args()
    if args.task not in CORE_TASKS:
        print(f"unknown task {args.task}")
        return 2

    capx = Path(args.capx)
    print(f"task={args.task} frozen={STATES_PER_TASK} large={args.large} "
          f"workers={workers()}")

    big = run(capx, args.task, args.large, "large")
    if big is None:
        return 1

    # Trials are seeded by index, so the frozen set is exactly the first N of
    # the large run -- no second execution needed, and no chance of drift
    # between the two samples.
    frozen = [t for t in big if t.trial <= STATES_PER_TASK]
    rest = [t for t in big if t.trial > STATES_PER_TASK]

    f_ok, f_n = sum(t.success for t in frozen), len(frozen)
    r_ok, r_n = sum(t.success for t in rest), len(rest)
    fi, ri = wilson(f_ok, f_n), wilson(r_ok, r_n)

    print(f"\n  frozen  seeds 1..{f_n:<4} {f_ok}/{f_n} = {fi}")
    print(f"  rest    seeds {STATES_PER_TASK+1}..{args.large:<3} {r_ok}/{r_n} = {ri}")

    overlap = not (fi.hi < ri.lo or ri.hi < fi.lo)
    diff = abs(fi.point - ri.point)
    print(f"\n  difference {diff:+.3f}, intervals {'overlap' if overlap else 'DO NOT overlap'}")
    if overlap:
        print("  [OK] frozen sample looks representative of the scene distribution")
        return 0
    print("  [WARN] the frozen 30 differ from the wider draw -- every cell in "
          "the study inherits this bias. Consider widening STATES_PER_TASK or "
          "reselecting seeds.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
