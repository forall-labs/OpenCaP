"""Check every (task, tier) in our table against a real cap-x checkout.

Run on a pod:  python3 scripts/verify_configs.py /content/cap-x
The table is hand-written from a directory listing, so it must be proven, not
trusted -- a wrong path would silently run the wrong tier.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencap.capx_bridge import (CORE_TASKS, MISSING, config_path,
                               oracle_config_path)
from opencap.cells import ALL_TIERS

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")
missing, ok = [], 0
for task in CORE_TASKS:
    for tier in ALL_TIERS:
        if (task, tier) in MISSING:
            continue
        p = root / config_path(task, tier)
        if p.exists():
            ok += 1
        else:
            missing.append(f"{task:<18} {tier:<3} -> {config_path(task, tier)}")

# also confirm the ones we declared absent really are absent
wrong = []
for task, tier in MISSING:
    for cand in (root / "env_configs" / task).glob("*.yaml"):
        if cand.name.endswith("_multiturn_vdm_reduced_api.yaml"):
            wrong.append(f"{task} {tier} declared MISSING but {cand.name} exists")

for task in CORE_TASKS:
    for priv in (True, False):
        p2 = root / oracle_config_path(task, priv)
        if p2.exists():
            ok += 1
        else:
            missing.append(f"{task:<18} oracle(priv={priv}) -> {oracle_config_path(task, priv)}")

print(f"resolved {ok}/{ok + len(missing)} configs (tiers + oracles)")
for m in missing:
    print("  MISSING FILE:", m)
for w in wrong:
    print("  BAD EXCLUSION:", w)
sys.exit(1 if (missing or wrong) else 0)
