"""Every config we intend to run must bind a simulator the seeding patch covers.

    python3 scripts/verify_tier_simulators.py [CAPX_DIR] [--tiers S1,S3,S4] [--strict]

WHY THIS EXISTS
---------------
ARTIFACT #13, ONE TIER OVER. The seeding patch reached `robosuite_cubes.py` and
two siblings; cube_lifting, spill_wipe and nut_assembly bound their own
simulator files, none of them patched, so those three tasks drew a RANDOM SCENE
on every run for months. It hid because `verify_pairing_states.py` probed the
one task where the patch worked.

The guards added afterwards pin exactly two things, and both say **S1**: the
patch target list must name every simulator an S1 config binds, and the pairing
probe must cover every task. Neither says anything about S3 or S4.

If a reduced-API config binds a different simulator class -- or the same class
through a different module -- then the tier grid repeats artifact #13 with a
fresh set of tasks, and it would look exactly as plausible as the first time.

WHAT IT DOES
------------
Reads the `low_level` binding out of each config's YAML, resolves it against
`capx/envs/simulators/__init__.py`, and checks the resulting module is in
`patch_seeding.TARGETS`. A structural read of the checkout, not a list of names
we already expect -- so a simulator nobody here has heard of still shows up.

Advisory by default; --strict exits non-zero on any unpatched binding. Run it
in stage 0 BEFORE collecting a tier, not after.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.capx_bridge import config_path, has_config  # noqa: E402

TASKS = ("cube_lifting", "cube_stack", "spill_wipe", "nut_assembly")


def _patch_seeding():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "patch_seeding", ROOT / "scripts" / "patch_seeding.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["patch_seeding"] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_targets() -> set[str]:
    return {Path(t).name for t in _patch_seeding().TARGETS}


#: Read from patch_seeding rather than duplicated here: two copies of a marker
#: drift, and the one that drifts is the one nobody re-reads.
MARKER = _patch_seeding().MARKER


def binding_of(text: str) -> str | None:
    """The `low_level` value in a config, however it is spelled."""
    m = re.search(r"low_level\s*:\s*['\"]?([A-Za-z0-9_.]+)", text)
    if m:
        return m.group(1)
    # some configs name it under a _target_
    m = re.search(r"_target_\s*:\s*['\"]?([A-Za-z0-9_.]*low_level[A-Za-z0-9_.]*)", text)
    return m.group(1) if m else None


def module_for(binding: str, root: Path) -> str | None:
    """Resolve a low_level binding to its simulator module filename.

    THE BINDING IS NOT THE IMPORTED NAME. `capx/envs/simulators/__init__.py`
    maps a snake_case registry key to a CamelCase class, and imports that
    class from its module:

        from .robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
        register_env("franka_robosuite_cube_lift_low_level",
                     FrankaRobosuiteCubeLiftLowLevel)

    so resolution is two hops, binding -> class -> module. The first version
    of this function looked for the BINDING in the import list and found it
    nowhere, so every row -- including the four S1 tasks we have been
    collecting correctly for weeks -- came back UNRESOLVED. A checker that
    cannot resolve anything reports the same thing as a checker whose subject
    is broken, which is the failure mode this file exists to prevent.
    """
    init = root / "capx" / "envs" / "simulators" / "__init__.py"
    if not init.exists():
        return None
    text = init.read_text(errors="ignore")

    # Hop 1: registry key -> class name.
    cls = None
    m = re.search(rf"register_env\(\s*['\"]{re.escape(binding)}['\"]\s*,\s*(\w+)",
                  text)
    if m:
        cls = m.group(1)

    # Hop 2: class name (or the binding itself, for checkouts that export it
    # directly) -> the module it is imported from.
    for want in (cls, binding):
        if not want:
            continue
        for m in re.finditer(r"from\s+\.(\w+)\s+import\s+([^\n]+)", text):
            names = [n.strip() for n in
                     m.group(2).replace("(", "").replace(")", "").split(",")]
            if want in names:
                return f"{m.group(1)}.py"

    # `<name> = <module>.<attr>` style registries.
    for m in re.finditer(rf"{re.escape(binding)}\s*[:=]\s*(\w+)\.", text):
        return f"{m.group(1)}.py"
    return None


def patch_applied(module: str, root: Path) -> bool:
    """Is the seeding patch actually IN the file, or only listed as a target?

    patch_seeding.TARGETS is a statement of intent. A fresh checkout, a failed
    patch run, or an upstream merge that reverted the hunk all leave a module
    that is listed and unpatched -- which is artifact #13's exact shape, since
    the scene silently goes random and every number still looks plausible.
    """
    p = root / "capx" / "envs" / "simulators" / module
    if not p.exists():
        return False
    return MARKER in p.read_text(errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capx", nargs="?", default="/data/cap-x")
    ap.add_argument("--tiers", default="S1,S3,S4")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    root = Path(a.capx)
    if not root.exists():
        print(f"[tiersim] no checkout at {root} -- skipping (pass CAPX_DIR)")
        return 0
    targets = patch_targets()
    print(f"[tiersim] seeding patch covers {len(targets)}: {', '.join(sorted(targets))}")

    bad, unknown = [], []
    print(f"\n{'tier':6}{'task':15}{'low_level binding':45}{'simulator':32}ok")
    print("-" * 104)
    for tier in a.tiers.split(","):
        for task in a.tasks.split(","):
            if not has_config(task, tier):
                print(f"{tier:6}{task:15}{'-- not in our grid --':45}{'':32}skip")
                continue
            cfg = root / config_path(task, tier)
            if not cfg.exists():
                print(f"{tier:6}{task:15}{'CONFIG MISSING: ' + str(cfg.name):45}{'':32}FAIL")
                bad.append((tier, task, "config missing"))
                continue
            binding = binding_of(cfg.read_text(errors="ignore"))
            if not binding:
                print(f"{tier:6}{task:15}{'no low_level key found':45}{'':32}?")
                unknown.append((tier, task))
                continue
            module = module_for(binding, root)
            if not module:
                mark = "UNRESOLVED"
            elif module not in targets:
                mark = "NOT A TARGET"
            elif not patch_applied(module, root):
                # Listed but not actually patched: a fresh checkout, a failed
                # patch run, or an upstream merge that reverted the hunk.
                mark = "NOT APPLIED"
            else:
                mark = "ok"
            print(f"{tier:6}{task:15}{binding:45}{str(module):32}{mark}")
            if mark in ("NOT A TARGET", "NOT APPLIED"):
                bad.append((tier, task, f"{module} [{mark}]"))
            elif mark == "UNRESOLVED":
                unknown.append((tier, task))

    print()
    if bad:
        print(f"[tiersim] {len(bad)} config(s) bind a simulator the seeding patch")
        print(f"[tiersim] does NOT touch. Those tasks would draw a random scene on")
        print(f"[tiersim] every run -- artifact #13, exactly. Add the module to")
        print(f"[tiersim] patch_seeding.TARGETS and re-apply BEFORE collecting.")
        for tier, task, mod in bad:
            print(f"[tiersim]   {tier} {task} -> {mod}")
    if unknown:
        print(f"[tiersim] {len(unknown)} binding(s) could not be resolved. That is")
        print(f"[tiersim] NOT a pass -- it means this script could not check them,")
        print(f"[tiersim] and an unchecked binding is how #13 survived. Resolve by")
        print(f"[tiersim] hand before collecting: {unknown}")
    if not bad and not unknown:
        print("[tiersim] OK -- every config in the grid binds a patched simulator.")
    return 1 if (a.strict and (bad or unknown)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
