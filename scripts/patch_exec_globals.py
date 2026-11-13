"""Optionally pre-import numpy into the code-execution namespace.

WHY THIS EXISTS
---------------
qwen3-4b wrote a *correct* cube-lifting program -- sample grasp, approach,
close gripper, lift -- and scored 0/30 because of one missing line:

    NameError: name 'np' is not defined     (reward 0.525, cube already grasped)

The task prompt does say "if you want to use numpy, you need to import it
explicitly", so this is a genuine instruction-following failure. But it is NOT
a robotics failure, and our headline numbers cannot currently tell the two
apart: an unknown fraction of every "model cannot control a robot" cell may be
"model forgot an import".

This patch makes that measurable rather than assumed. Gated on an environment
variable so the SAME build runs both conditions:

    OPENCAP_PREIMPORT_NUMPY=1   -> numpy available as np in user code
    unset                     -> stock CaP-X behaviour

The difference between the two conditions is the quantity of interest: how much
of the reported floor is prompt compliance versus capability. Reporting only
the patched condition would overstate models; reporting only the unpatched one
conflates two very different failures.

Idempotent. Applied from bootstrap_colab.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

MARKER = "# opencap: optional numpy preimport"

ANCHOR = "    def _init_exec_globals(self)"

INJECT = '''
        # opencap: optional numpy preimport -- see scripts/patch_exec_globals.py.
        # Off by default so stock behaviour is preserved; the A/B difference
        # measures how much of the floor is instruction-following rather than
        # robot control.
        import os as _os
        if _os.environ.get("OPENCAP_PREIMPORT_NUMPY"):
            try:
                import numpy as _np
                self._exec_globals["np"] = _np
                self._exec_globals["numpy"] = _np
            except Exception:
                pass
'''


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")
    target = root / "capx/envs/tasks/base.py"
    if not target.exists():
        print(f"  MISSING {target}")
        return 1
    text = target.read_text()
    if MARKER in text:
        print("  already patched   capx/envs/tasks/base.py")
        return 0
    if ANCHOR not in text:
        print("  ANCHOR NOT FOUND -- inspect _init_exec_globals manually")
        return 1

    # Insert at the end of _init_exec_globals: find the method, then the next
    # method definition at the same indent, and splice in between.
    idx = text.index(ANCHOR)
    nxt = text.find("\n    def ", idx + 1)
    if nxt == -1:
        nxt = len(text)
    target.write_text(text[:nxt].rstrip("\n") + "\n" + INJECT + text[nxt:])
    print("  patched           capx/envs/tasks/base.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
