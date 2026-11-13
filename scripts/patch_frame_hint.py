"""Optionally tell the model the coordinate conventions the reduced API uses.

WHY THIS EXISTS
---------------
FRAME is the signature defect of the reduced API: 4% of failing programs at S1
carry one, 47% at S3/S4, and 19 of 21 models make at least one. The three
forms, from the hand tags:

    camera-frame values used as world coordinates (the transform is never
      applied, or is applied to the wrong thing)
    a wxyz quaternion handed to a library that takes xyzw, or the reverse
    a 4x4 homogeneous transform applied to a 3-vector

Every one of those is a CONVENTION, not a judgement about the task. Nothing
about the scene tells you whether `cam["points"]` is in the camera's frame or
the world's; it is a fact about the API that the prompt does not state.

That makes this the first intervention in the project with a mechanism behind
it rather than a hope. The two that came before -- the entry-point call and
the defect-aware prompt addendum -- were both negative, and the reason is
known: they targeted failures the model could already see. This targets a
fact the model cannot derive.

THE CLAIM UNDER TEST is the one the paper implicitly makes: that a defect
taxonomy is ACTIONABLE. If naming the conventions moves FRAME at S3/S4, the
taxonomy earns that word. If it does not, the word has to go, and "we
catalogued what fails and could not act on it" is the honest version.

    OPENCAP_FRAME_HINT=1   -> the conventions are appended to the task prompt
    unset                -> stock prompt

WHAT IS DELIBERATELY NOT IN THE ADDENDUM. No task advice, no mention of
grasping, lifting, approach offsets or any defect family other than FRAME. It
states facts about the API's coordinate conventions and nothing else, so a
movement in FRAME is attributable and a movement anywhere else is a surprise
worth reporting rather than a confound we built in.

Idempotent. Same anchor as scripts/patch_prompt_fix.py, which has applied
cleanly since Phase 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

MARKER = "# opencap: optional frame-convention note"

ANCHOR = ("        self._task_prompt = cfg.prompt if cfg.prompt is not None "
          "else self.prompt")

#: Every line traces to a defect form recorded in results/tags_reduced.json.
#: Nothing is here because it sounds like good advice.
ADDENDUM = '''

Coordinate conventions used by this API:
- Points and poses returned by the camera are in the CAMERA frame. To use them as world coordinates you must transform them with the camera pose; they are not world coordinates as returned.
- Quaternions returned by this API are ordered (w, x, y, z). Libraries such as scipy expect (x, y, z, w). Convert in both directions when you pass them across.
- A 4x4 homogeneous transform cannot be applied to a 3-vector directly: append 1, multiply, then drop the last component.'''

INJECT = '''
        # opencap: optional frame-convention note -- see
        # scripts/patch_frame_hint.py. Off by default so stock behaviour is
        # preserved; the A/B difference measures whether naming a convention
        # the prompt never states moves the defect class that depends on it.
        import os as _os
        if _os.environ.get("OPENCAP_FRAME_HINT"):
            self._task_prompt = self._task_prompt + %r
'''


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/cap-x")
    target = root / "capx/envs/tasks/base.py"
    if not target.exists():
        print(f"MISSING {target}")
        return 1
    text = target.read_text()
    if MARKER in text:
        print(f"already patched  {target}")
        return 0
    if ANCHOR not in text:
        # Loudly, not silently: a patch that quietly fails to apply is how
        # artifact #13 survived for months.
        print(f"PATTERN NOT FOUND in {target} -- inspect manually.")
        print(f"anchor: {ANCHOR!r}")
        return 1
    text = text.replace(ANCHOR, ANCHOR + "\n" + (INJECT % ADDENDUM), 1)
    target.write_text(text)
    print(f"patched  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
