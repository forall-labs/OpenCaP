"""Measure how reproducible CaP-Bench actually is.

CaP-X evaluates single-seed and reports no variance. We found why that matters:
`seed=` never reached either the object placement sampler or the robot
initialisation noise, so trial i was a different scene on every run. Two runs
of the identical deterministic human-oracle program scored 27/30 and 26/30 with
7 trials flipping outcome.

This quantifies that properly, across all 7 tasks, using ONLY the oracle -- no
LLM inference at all, so it costs simulator time and nothing else. Two
quantities, and the difference between them is the point:

  scene variance      -- spread across repeats WITHOUT the seeding patch.
                         This is the uncertainty carried by published numbers.
  execution variance  -- spread across repeats WITH the patch, i.e. identical
                         scenes. Irreducible: contact physics, the JAX IK
                         server, worker scheduling. No amount of seeding
                         removes it, and every model-vs-model claim must clear
                         it to mean anything.

    python3 scripts/measure_reliability.py /content/cap-x --repeats 3 --trials 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencap.capx_bridge import CORE_TASKS, oracle_config_path  # noqa: E402
from opencap.results import cross_check, parse_trials, resolved_output_dir  # noqa: E402

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


ORACLE_MODEL = "google/gemini-3.1-pro-preview"  # the config's default label


def default_workers() -> int:
    """Match workers to vCPUs, never exceed them.

    MuJoCo is CPU-bound, and Colab ties vCPU count to GPU tier: a T4 instance
    has 2 vCPUs where an A100 has 12. Running 12 workers on 2 vCPUs thrashes --
    measured: 30 oracle trials finished in ~40s on a 12-vCPU machine and did
    not finish within an HOUR on a 2-vCPU one. The "cheap" GPU is a false
    economy for simulation-bound work.
    """
    import os
    return max(1, min(12, (os.cpu_count() or 2)))


def run_oracle(capx: Path, task: str, trials: int, out: Path,
               workers: int | None = None, timeout: int = 3600) -> list | None:
    workers = workers or default_workers()
    cmd = ["uv", "run", "capx/envs/launch.py",
           "--config-path", oracle_config_path(task, privileged=True),
           "--total-trials", str(trials), "--num-workers", str(workers),
           "--record-video", "False", "--output-dir", str(out)]
    env = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    import os
    p = subprocess.run(cmd, cwd=capx, env=dict(os.environ, **env),
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        print(f"    launch failed rc={p.returncode}: {p.stderr[-300:]}")
        return None
    run_dir = resolved_output_dir(out, ORACLE_MODEL)
    ok, msg = cross_check(run_dir)
    if not ok:
        print(f"    cross-check failed: {msg}")
        return None
    return parse_trials(run_dir)


def _save(args, results: dict) -> None:
    doc = {"label": args.label, "repeats": args.repeats,
           "trials": args.trials, "tasks": results}
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capx", nargs="?", default="/content/cap-x")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--label", default="patched",
                    help="'patched' or 'unpatched' -- recorded in the output")
    ap.add_argument("--out", default="reliability.json")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: min(12, vCPUs) -- oversubscribing thrashes")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-oracle-run seconds; two-arm tasks on few vCPUs "
                         "exceed the 3600 default")
    args = ap.parse_args()

    capx = Path(args.capx)
    print(f"workers={args.workers or default_workers()} (vCPUs={__import__('os').cpu_count()})")
    tasks = args.tasks.split(",") if args.tasks else list(CORE_TASKS)
    results: dict[str, dict] = {}

    for task in tasks:
        print(f"\n=== {task} ===")
        runs = []
        for r in range(args.repeats):
            t0 = time.time()
            out = Path(f"/content/outputs/rel_{args.label}_{task}_{r}")
            try:
                trials = run_oracle(capx, task, args.trials, out,
                                    args.workers, args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  repeat {r}: TIMEOUT after {args.timeout}s -- skipping")
                trials = None
            if trials is None:
                continue
            succ = sum(t.success for t in trials)
            runs.append({"successes": succ, "n": len(trials),
                         "per_trial": {t.trial: t.success for t in trials}})
            print(f"  repeat {r}: {succ}/{len(trials)} in {time.time()-t0:.0f}s")

        if len(runs) < 2:
            print("  too few successful repeats to compare")
            continue

        rates = [r["successes"] / r["n"] for r in runs]
        # Pairwise disagreement on the SAME trial index -- the quantity that
        # says whether trial i means the same thing twice.
        flips = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i]["per_trial"], runs[j]["per_trial"]
                shared = set(a) & set(b)
                flips.append(sum(1 for k in shared if a[k] != b[k]) / max(len(shared), 1))

        results[task] = {
            "rates": rates,
            "rate_mean": statistics.fmean(rates),
            "rate_range": max(rates) - min(rates),
            "rate_stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "flip_rate_mean": statistics.fmean(flips),
            "flip_rate_max": max(flips),
        }
        print(f"  rate {statistics.fmean(rates):.3f} "
              f"(range {max(rates)-min(rates):.3f}), "
              f"flip {statistics.fmean(flips):.1%}")
        # Persist after EVERY task. A timeout in task 6 previously discarded
        # five completed measurements -- the same lesson as cell-atomic commits.
        _save(args, results)

    _save(args, results)
    if results:
        print(f"\nOVERALL ({args.label}): "
              f"mean flip {statistics.fmean(r['flip_rate_mean'] for r in results.values()):.1%}, "
              f"mean rate range {statistics.fmean(r['rate_range'] for r in results.values()):.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
