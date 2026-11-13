"""Gate 0 for the REDUCED API: can a hand-written program do the task at all?

    python3 scripts/reduced_api_gate0.py cube_lifting --code progs/x.py
    python3 scripts/reduced_api_gate0.py cube_stack --code progs/y.py --tier S3

WHY THIS BLOCKS THE WHOLE TAGGING PASS
--------------------------------------
`scripts/run_gate0_oracle.sh` asks the question for S1 and its header says why:
a task whose hand-written reference cannot reach the ceiling is a task where a
model's 0/30 means nothing -- it could be the model, the config, the reward or
a missing dependency, and the score cannot tell you which.

Nobody has ever asked that question of the reduced API, and the tier grid now
rests on the answer. `results/references.json` carries ground truth for all
four tasks and every entry is `tier: S1`: all four call `get_object_pose` and
`sample_grasp_pose`, neither of which S3/S4 binds. So:

  * 169 failing S3/S4 programs have NO oracle to be repaired against, exactly
    as nut_assembly's missing oracle blocked 21 of the 70 S1 repairs; and
  * the headline -- cube_stack is the EASIEST task at S1 (49.0%) and 0/630 at
    BOTH reduced tiers -- is uninterpretable until someone shows the reduced
    API can express cube_stack at all. If it cannot, that 0/630 is a property
    of the API surface and not of the models, and the rank inversion is
    partly an artifact of what S3 can say.

This project has retracted a tier for precisely this failure once already (S4,
whose prompt promised camera observations the environment could not provide),
and has retracted three explanations of the inversion. Reporting the absence
of a capability as a property of the thing measured is the house error.

RUN THE POSITIVE CONTROL FIRST. cube_lifting has a model program that scored
27/30 at S3, so injecting it here must reproduce ~27/30. If it does not, the
instrument is broken and a zero on cube_stack means nothing either.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.capx_bridge import config_path  # noqa: E402
from opencap.results import is_complete, parse_trials  # noqa: E402

#: Every reduced-API program needs these. molmo (:8122) is in the list because
#: it is NOT in any task config -- it is bound unconditionally by
#: control_reduced.py and its absence refused 25 cells before it had a
#: launcher. launch.py warns and runs the cell anyway against a dead service,
#: scoring an environment failure as a program defect, so this refuses instead.
REQUIRED_PORTS = {8114: "sam3", 8115: "graspnet", 8116: "pyroki", 8122: "molmo"}


def assert_gpu_is_consistent() -> str | None:
    """Refuse when OPENCAP_GPU and CUDA_VISIBLE_DEVICES disagree.

    env_server.sh derives CUDA_VISIBLE_DEVICES from OPENCAP_GPU AT SOURCE TIME
    and defaults OPENCAP_GPU to 2. So this, which looks careful, silently runs
    on GPU 2:

        source scripts/env_server.sh     # OPENCAP_GPU=2, CUDA_VISIBLE_DEVICES=2
        export OPENCAP_GPU=5               # CUDA_VISIBLE_DEVICES is STILL 2

    It put a verification run on a neighbour's card on 2026-10-02. NVML reads
    (nvidia-smi -i $OPENCAP_GPU) then report the card you meant while the work
    lands on the card you did not, so nothing in the logs looks wrong.

    The fix at the call site is to set OPENCAP_GPU BEFORE sourcing. This is the
    backstop for when someone forgets, because the failure is invisible.
    """
    want = os.environ.get("OPENCAP_GPU")
    have = os.environ.get("CUDA_VISIBLE_DEVICES")
    if want is None or have is None or want == have:
        return None
    return (f"OPENCAP_GPU={want} but CUDA_VISIBLE_DEVICES={have}. The work would "
            f"run on GPU {have}, not {want}.\n"
            f"Set OPENCAP_GPU BEFORE sourcing env_server.sh:\n"
            f"  OPENCAP_GPU={want} source scripts/env_server.sh")


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(2.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def build_config(capx: Path, task: str, tier: str, code: str, out_dir: str,
                 trials: int, workers: int, dest: Path, video: bool = False) -> None:
    import yaml
    base = yaml.safe_load((capx / config_path(task, tier)).read_text())
    base.setdefault("env", {}).setdefault("cfg", {})["oracle_code"] = code
    base["output_dir"] = out_dir
    base["trials"] = trials
    base["num_workers"] = workers
    base["record_video"] = video
    dest.write_text(yaml.safe_dump(base, sort_keys=False, width=10**6))


def run(capx: Path, task: str, tier: str, label: str, code: str, trials: int,
        workers: int, scratch: Path, video: bool = False) -> dict:
    out_dir = scratch / label
    cfg = scratch / f"{label}.yaml"
    for stale in (out_dir, out_dir.parent / "oracle" / out_dir.name):
        if stale.exists():
            shutil.rmtree(stale)
    build_config(capx, task, tier, code, str(out_dir), trials, workers, cfg, video)
    cmd = ["uv", "run", "--no-sync", "capx/envs/launch.py",
           "--config-path", str(cfg), "--use-oracle-code", "True",
           "--total-trials", str(trials), "--num-workers", str(workers),
           "--record-video", str(video)]
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               OPENCAP_PREIMPORT_NUMPY="1")
    env.pop("OPENCAP_CALL_ENTRYPOINT", None)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=capx, env=env, capture_output=True, text=True)
    wall = round(time.time() - t0, 1)
    run_dir = out_dir.parent / "oracle" / out_dir.name
    if not run_dir.exists():
        run_dir = out_dir
    if not is_complete(run_dir):
        return {"label": label, "ok": False, "error": "run did not complete",
                "tail": (proc.stderr or proc.stdout or "")[-600:], "wall": wall}
    parsed = parse_trials(run_dir)
    # TrialResult is a dataclass, not a dict -- `.get` silently does not exist.
    succ = sum(1 for t in parsed if t.success)
    return {"label": label, "ok": True, "successes": succ, "trials": len(parsed),
            "wall": wall, "run_dir": str(run_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--code", required=True, help="file holding the candidate program")
    ap.add_argument("--tier", default="S3")
    ap.add_argument("--trials", type=int, default=30)
    # 16, not 2. The 2 came from the collection phase, where a vLLM engine
    # shared the card; verification loads no LLM. Measured on the spill_wipe
    # reference (392 waypoints, 30 trials), all scoring 30/30:
    #     w=2  527s | w=16 106s | w=32 99s | w=48 110s
    # so it is a 5x speedup that saturates around 16-32 -- the perception
    # services are the limit, not the 240 vCPUs. Going past 32 is slower.
    ap.add_argument("--workers", type=int, default=int(os.environ.get("OPENCAP_WORKERS", "16")))
    ap.add_argument("--label", default=None)
    ap.add_argument("--video", action="store_true",
                    help="record a video per trial (to see WHY a scene fails)")
    ap.add_argument("--capx", default=os.environ.get("CAPX_DIR", "/data/cap-x"))
    ap.add_argument("--scratch", default=os.environ.get(
        "OPENCAP_SCRATCH", "/data/capx-out") + "/reduced_gate0")
    a = ap.parse_args()

    bad = assert_gpu_is_consistent()
    if bad:
        print(f"REFUSING: {bad}")
        return 1

    down = [f":{p} ({n})" for p, n in REQUIRED_PORTS.items() if not _port_open(p)]
    if down:
        print(f"REFUSING: perception services down: {', '.join(down)}")
        print("launch.py would run the program anyway and score the dead service")
        print("as a program defect -- 30 plausible zeros with no error.")
        return 1

    code = Path(a.code).read_text()
    label = a.label or f"{a.task}__{a.tier}__{Path(a.code).stem}"
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[gate0r] {a.task} {a.tier} <- {a.code}  ({a.trials} trials, {a.workers} workers)")
    r = run(Path(a.capx), a.task, a.tier, label, code, a.trials, a.workers, scratch, a.video)
    if not r["ok"]:
        print(f"[gate0r] FAILED TO RUN: {r['error']}\n{r.get('tail','')}")
        return 1
    print(f"[gate0r] {r['successes']}/{r['trials']} in {r['wall']}s -> {r['run_dir']}")
    rec = scratch / "results.json"
    hist = json.loads(rec.read_text()) if rec.exists() else []
    hist.append({**r, "task": a.task, "tier": a.tier, "code_file": a.code})
    rec.write_text(json.dumps(hist, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
