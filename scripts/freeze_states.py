"""Fingerprint the frozen initial states, so drift is detectable.

Now that seed=trial genuinely determines the scene, the "frozen states" are
just trial indices 1..N -- there is no separate state file to store. What DOES
need storing is a fingerprint, because the determinism depends on a patch
(scripts/patch_seeding.py) applied to a third-party checkout. A fresh clone
that skips bootstrap, or a robosuite version bump, would silently restore the
old random-scene behaviour and every later cell would be paired against
different worlds than the earlier ones.

This writes a manifest of per-(task, trial) observation hashes. Re-running it
later and diffing is a cheap guard against exactly that.

    python3 scripts/freeze_states.py /content/cap-x [N] [--upload REPO]
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencap.capx_bridge import CORE_TASKS, oracle_config_path  # noqa: E402
from opencap.config import STATES_PER_TASK  # noqa: E402

PROBE = r'''
import hashlib, json, sys
sys.path.insert(0, ".")
from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader
import numpy as np

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


cfg_path, n = sys.argv[1], int(sys.argv[2])
cfg = DictLoader.load(cfg_path)

def flatten(obj, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k in sorted(obj):
            flatten(obj[k], f"{prefix}{k}.", out)
        return out
    try:
        a = np.asarray(obj, dtype=float).ravel()
    except (TypeError, ValueError):
        return out
    if a.size and a.size <= 4096:
        out[prefix.rstrip(".")] = [round(float(x), 9) for x in a[:12]]
    return out

env = instantiate(cfg["env"])
res = {}
for s in range(1, n + 1):
    obs, _ = env.reset(options={"trial": s}, seed=s)
    blob = json.dumps(flatten(obs), sort_keys=True).encode()
    res[s] = hashlib.sha256(blob).hexdigest()[:16]
print("JSON_START")
print(json.dumps(res))
'''


def main() -> int:
    capx = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")
    n = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") \
        else STATES_PER_TASK
    upload = sys.argv[sys.argv.index("--upload") + 1] if "--upload" in sys.argv else None

    probe = capx / "_freeze_probe.py"
    probe.write_text(PROBE)
    manifest: dict[str, dict[str, str]] = {}
    try:
        for task in CORE_TASKS:
            cfg = oracle_config_path(task, privileged=True)
            p = subprocess.run(["uv", "run", "python", probe.name, cfg, str(n)],
                               cwd=capx, capture_output=True, text=True, timeout=1200)
            if "JSON_START" not in p.stdout:
                print(f"  FAIL {task}: {p.stderr.strip().splitlines()[-1:] or p.stdout[-200:]}")
                continue
            manifest[task] = json.loads(
                p.stdout.split("JSON_START", 1)[1].strip().splitlines()[0])
            print(f"  {task:<18} {len(manifest[task])} states hashed")
    finally:
        probe.unlink(missing_ok=True)

    if not manifest:
        print("no states hashed")
        return 1

    doc = {"states_per_task": n, "tasks": manifest,
           "digest": hashlib.sha256(
               json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]}
    out = Path("state_manifest.json")
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"\nwrote {out} | digest {doc['digest']} | {len(manifest)} tasks")

    if upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(out), path_in_repo="state_manifest.json",
                            repo_id=upload, repo_type="dataset",
                            commit_message=f"frozen states digest {doc['digest']}")
        print(f"uploaded to {upload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
