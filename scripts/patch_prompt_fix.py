"""Optionally append a defect-aware addendum to the task prompt.

WHY THIS EXISTS
---------------
Repairing 28 failing programs produced a catalogue of defects, each verified by
re-executing a minimal fix against the same 30 seeds. Six of those defect
classes need no code change and no different model -- only an instruction the
task prompt never gives.

    60% of all failed episodes carry at least one of them.
    Among models <=3B it is 74%.

This patch tests whether the catalogue is ACTIONABLE or merely descriptive.
Gated so the same build runs both arms:

    OPENCAP_PROMPT_FIX=1   -> addendum appended to the task prompt
    unset                -> stock CaP-X prompt

Every line below is traceable to a defect that a verified repair had to fix.
Nothing is included on the grounds that it sounds like good advice.

    api_rebound / mock_shadowing  -> "do not define your own versions"
    never_invoked                 -> "if you define a function, call it"
    spurious_move                 -> "move to a sampled grasp pose, not the object's pose"
    approach_param                -> z_approach on the grasp, not the lift
    stale_read                    -> re-read a pose after motion that may move it
    missing_action                -> end with the action the task asks for

Idempotent. Applied from bootstrap_colab.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

MARKER = "# opencap: optional defect-aware prompt addendum"

ANCHOR = ("        self._task_prompt = cfg.prompt if cfg.prompt is not None "
          "else self.prompt")

ADDENDUM = '''

Additional requirements:
- The functions listed above are already provided. Do not define your own versions of them, and do not assign anything to their names.
- If you define a function, call it.
- To reach an object, move to a pose returned by sample_grasp_pose. Do not move to the object's own pose from get_object_pose.
- Pass z_approach when approaching an object in order to grasp it. Do not pass z_approach when lifting the object or moving away from it.
- Call get_object_pose again for any object whose pose you need after a motion that may have moved it.
- End with the action the task asks for. A lift must finish with an upward move while still holding the object.'''

INJECT = '''
        # opencap: optional defect-aware prompt addendum -- see
        # scripts/patch_prompt_fix.py. Off by default so stock behaviour is
        # preserved; the A/B difference measures whether a verified defect
        # catalogue can be turned into a prompt-level fix.
        import os as _os
        if _os.environ.get("OPENCAP_PROMPT_FIX"):
            self._task_prompt = self._task_prompt + %r
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
        print("  ANCHOR NOT FOUND -- inspect _task_prompt assignment manually")
        return 1
    i = text.index(ANCHOR) + len(ANCHOR)
    target.write_text(text[:i] + "\n" + (INJECT % ADDENDUM).rstrip("\n") + text[i:])
    print("  patched           capx/envs/tasks/base.py (prompt addendum)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
