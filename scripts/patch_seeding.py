"""Make CaP-X's `seed=` argument actually determine the scene.

THE BUG
-------
`RobosuiteCubes.reset(seed=...)` and its siblings do:

    if seed is not None:
        self._rng = np.random.default_rng(seed)   # CaP-X's own generator
    self.robosuite_env.reset()                    # samples object placements

Object placement happens inside `robosuite_env.reset()`, using robosuite's
placement sampler, which draws from `sampler.rng` -- a generator created at
construction time that never sees this seed. So the seed is stored and has no
effect: two resets with the same seed produce scenes ~5cm apart.

Measured on cube_stack: 100 observation fields differ across two same-seed
resets, and two runs of the identical deterministic oracle scored 27/30 and
26/30 with 7 trials flipping outcome.

Note `np.random.seed()` would NOT fix this -- the samplers use their own
Generator instances, not the legacy global.

WHY IT MATTERS BEYOND US
------------------------
`--total-trials N` implies N reproducible scenes. It actually means N fresh
random scenes per run, so trial index i is not comparable across runs, models
or tiers -- and every published per-task number carries unreported scene
variance.

THE FIX
-------
There are TWO independent random sources, and fixing only the first leaves the
scene non-reproducible (measured: 100 differing fields -> 60, with the remainder
all robot joint positions):

  1. Object placement -- the placement initializer, possibly a
     SequentialCompositeSampler holding an OrderedDict of sub-samplers, each
     with its own `rng`.
  2. Robot initialization noise -- `robot.reset(rng=self.rng)` draws from the
     robosuite env's `self.rng`, which is created ONCE at construction. CaP-X
     builds the env per worker and resets many times, so that generator simply
     advances and is never reseeded.

We reseed both before `robosuite_env.reset()`. Note we reseed rather than
disable the noise: setting initialization_noise to zero would also make the
task easier, changing the distribution rather than just making it reproducible.

Idempotent; safe to re-run. Called from bootstrap_colab.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


MARKER = "# opencap: seed the placement samplers"
HELPER_VERSION = "v2"  # bump to force the helper to be rewritten

HELPER = '''

def _opencap_seed_placement(robosuite_env, seed):  # opencap: seed the placement samplers
    """Reseed every RNG the placement initializer owns.

    robosuite samplers hold their own numpy Generator, so seeding the global
    RNG has no effect; and a SequentialCompositeSampler nests further samplers
    in `.samplers`, each with its own.
    """
    import numpy as _np

    # (2) robot initialization noise: robot.reset(rng=env.rng), and env.rng is
    # only ever seeded at construction.
    if hasattr(robosuite_env, "rng"):
        robosuite_env.rng = _np.random.default_rng(seed)

    # (1) object placement
    init = getattr(robosuite_env, "placement_initializer", None)
    if init is None:
        return
    stack, n = [init], 1
    while stack:
        s = stack.pop()
        if hasattr(s, "rng"):
            s.rng = _np.random.default_rng(seed + n)
            n += 1
        sub = getattr(s, "samplers", None)
        if isinstance(sub, dict):
            stack.extend(sub.values())
        elif isinstance(sub, (list, tuple)):
            stack.extend(sub)
'''

# ARTIFACT #13 (2026-09-11). The first version of this list held three files
# and every S1 task was assumed to route through one of them. It does not:
# each task binds its own simulator class, and cube_lifting, spill_wipe and
# nut_assembly reset() through robosuite_cube_lift.py, robosuite_spill_wipe.py
# and robosuite_nut_assembly.py -- none of which were here. Only cube_stack
# was ever seeded. verify_pairing_states.py did not catch it because it
# probed the cube_stack oracle config alone, and passed. Every "paired on
# state_id" statistic on the other three tasks compared different scenes.
#
# The list below is checked by a test against the low_level binding of every
# S1 config, so a new simulator file cannot be left out silently again.
TARGETS = (
    "capx/envs/simulators/robosuite_cubes.py",
    "capx/envs/simulators/robosuite_cubes_restack.py",
    "capx/envs/simulators/robosuite_handover.py",
    "capx/envs/simulators/robosuite_cube_lift.py",
    "capx/envs/simulators/robosuite_spill_wipe.py",
    "capx/envs/simulators/robosuite_nut_assembly.py",
)

# Two shapes of the same bug. Most simulators call reset() bare;
# robosuite_nut_assembly.py keeps its return value.
PATTERNS = (
    ("""        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()""",
     """        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Without this the seed never reaches the object placement sampler
            # and the scene is random every run. See scripts/patch_seeding.py.
            _opencap_seed_placement(self.robosuite_env, seed)

        self.robosuite_env.reset()"""),
    ("""        if seed is not None:
            self._rng = np.random.default_rng(seed)

        first_obs = self.robosuite_env.reset()""",
     """        if seed is not None:
            self._rng = np.random.default_rng(seed)
            # Without this the seed never reaches the object placement sampler
            # and the scene is random every run. See scripts/patch_seeding.py.
            _opencap_seed_placement(self.robosuite_env, seed)

        first_obs = self.robosuite_env.reset()"""),
)
OLD, NEW = PATTERNS[0]   # kept for callers that import them


def patch_file(path: Path) -> str:
    text = path.read_text()
    if MARKER in text:
        # Rewrite the helper in place rather than skipping, so bumping
        # HELPER_VERSION actually takes effect on an already-patched checkout.
        head = text[: text.index(MARKER)]
        head = head[: head.rindex("\ndef _opencap_seed_placement")]
        path.write_text(head.rstrip("\n") + "\n" + HELPER)
        return f"helper updated ({HELPER_VERSION})"
    for old_pat, new_pat in PATTERNS:
        if old_pat in text:
            text = text.replace(old_pat, new_pat, 1)
            break
    else:
        return "PATTERN NOT FOUND -- inspect manually"
    text = text.rstrip("\n") + "\n" + HELPER
    path.write_text(text)
    return "patched"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")
    bad = 0
    for rel in TARGETS:
        p = root / rel
        if not p.exists():
            print(f"  MISSING {rel}")
            bad += 1
            continue
        status = patch_file(p)
        print(f"  {status:<34} {rel}")
        if "NOT FOUND" in status:
            bad += 1
    print("patch incomplete" if bad else "seeding patch applied")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
