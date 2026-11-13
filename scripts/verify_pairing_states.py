"""Test the property pairing actually needs: does seed=trial fix the INITIAL STATE?

An earlier version of this check compared end-to-end trial *outcomes* across two
identical oracle runs and reported failure when 7/30 flipped. That was the wrong
property. Pairing requires only that model A and model B face the same initial
conditions at index i -- not that one policy run twice yields the same result.
CaP-X already resets with `seed=trial`, so the question is whether that seeding
actually determines the scene.

The 7/30 flip rate is still meaningful, but as something else: an execution-noise
floor measured on a deterministic policy. Any model-vs-model difference has to
clear it, and no amount of pairing removes it.

    python3 scripts/verify_pairing_states.py /content/cap-x [N]
"""
import subprocess
import sys
from pathlib import Path

PROBE = r'''
import json, numpy as np, sys
sys.path.insert(0, ".")
# CaP-X uses its own OmegaConf-based loader/instantiator, not hydra, despite
# the _target_ syntax. Use the same path runner.py does so the probe builds
# exactly the env the real trials build.
from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


cfg = DictLoader.load(sys.argv[2])
def flatten(obj, prefix="", out=None):
    """Recursively flatten an observation into {path: [floats]}.

    Observations nest arbitrarily (dicts of dicts of arrays), so a one-level
    walk is not enough. Large arrays are truncated -- we are testing whether
    the scene is identical, not archiving it.
    """
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k in sorted(obj):
            flatten(obj[k], f"{prefix}{k}.", out)
        return out
    try:
        a = np.asarray(obj, dtype=float).ravel()
    except (TypeError, ValueError):
        out[prefix.rstrip(".")] = repr(obj)[:120]
        return out
    if a.size and a.size <= 4096:
        out[prefix.rstrip(".")] = [round(float(x), 9) for x in a[:12]]
    return out


def snapshot(seed):
    env = instantiate(cfg["env"])
    obs, _ = env.reset(options={"trial": seed}, seed=seed)
    out = flatten(obs)
    try:
        env.close()
    except Exception:
        pass
    return out


N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
res = {}
for s in range(1, N + 1):
    res[s] = [snapshot(s), snapshot(s)]
print("JSON_START")
print(json.dumps({str(k): v for k, v in res.items()}))
'''


#: Every task with an S1 privileged oracle. ARTIFACT #13: the first version of
#: this check probed cube_stack only and passed, while cube_lifting, spill_wipe
#: and nut_assembly reset through simulator files the seeding patch never
#: touched. A precondition tested on one task and asserted for four is not a
#: precondition. Each task binds its own simulator, so each is probed.
TASKS = ("cube_lifting", "cube_stack", "spill_wipe", "nut_assembly")


def probe_task(capx: Path, task: str, n: str) -> tuple[int, int, str]:
    """-> (n_bad_fields, n_obs_fields, error)."""
    import json
    cfg = (f"env_configs/human_oracle_code/robosuite/"
           f"franka_robosuite_{task}_privileged_oracle.yaml")
    probe = capx / f"_pairing_probe_{task}.py"
    probe.write_text(PROBE)
    try:
        p = subprocess.run(["uv", "run", "--no-sync", "python", probe.name, n, cfg],
                           cwd=capx, capture_output=True, text=True, timeout=900)
    finally:
        probe.unlink(missing_ok=True)
    if "JSON_START" not in p.stdout:
        return -1, 0, (p.stderr or p.stdout)[-600:]
    data = json.loads(p.stdout.split("JSON_START", 1)[1].strip().splitlines()[0])
    bad = sum(1 for _, (a, b) in data.items()
              for k in set(a) | set(b) if a.get(k) != b.get(k))
    return bad, len(next(iter(data.values()))[0]), ""


def main() -> int:
    capx = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")
    n = sys.argv[2] if len(sys.argv) > 2 else "5"
    tasks = sys.argv[3].split(",") if len(sys.argv) > 3 else TASKS
    failed = []
    for task in tasks:
        bad, nf, err = probe_task(capx, task, n)
        if bad < 0:
            print(f"  {task:<13} PROBE FAILED: {err}")
            failed.append(task)
        elif bad:
            print(f"  {task:<13} FAIL  {bad} field(s) differ across same-seed "
                  f"resets ({nf} obs fields, {n} seeds x 2)")
            failed.append(task)
        else:
            print(f"  {task:<13} ok    seed pins the scene ({nf} obs fields, "
                  f"{n} seeds x 2)")
    if failed:
        print(f"\n[FAIL] seed=trial does NOT determine the scene for: "
              f"{', '.join(failed)}. Pairing on state_id is invalid there.")
        return 1
    print("\n[OK] identical seed => identical initial state on every task. "
          "Pairing on state_id is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
