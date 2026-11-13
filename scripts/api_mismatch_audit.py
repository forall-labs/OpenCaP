"""Which S3/S4 trials died on a primitive the tier never bound?

    python3 scripts/api_mismatch_audit.py                       # all tiers found
    python3 scripts/api_mismatch_audit.py --tier S3
    python3 scripts/api_mismatch_audit.py --scratch /data/capx-out

CANDIDATE MEASUREMENT ARTIFACT #16
----------------------------------
The reduced-API tiers hand the model a prompt that names functions the reduced
API does not bind. A model that follows the prompt gets a NameError before it
actuates anything, and the cell records a plausible zero.

TWO SOURCES, BOTH UPSTREAM, BOTH CONFIRMED BY READING THE CHECKOUT:

  1. THE DOCSTRING'S OWN WORKED EXAMPLE IS WRONG. The docstring for
     `segment_sam3_text_prompt` -- which IS bound -- ends with

         >>> masks = segment_sam3(rgb, text_prompt="red mug")

     and `segment_sam3` is not bound at any tier. The example is the part of a
     docstring a small model copies most faithfully, so this converts
     prompt-following into a hard failure. Artifact #14 in a new place: the
     documentation and the binding disagree, and the documentation is what the
     model can see.

  2. THE TASK'S "KEY RULES" WERE WRITTEN FOR S1 AND ARE NOT TIER-CONDITIONED.
     They live in a module-level constant in
     capx/envs/tasks/franka/franka_pick_place.py -- the file contains no
     reference to `privileged`, `reduced` or `apis` -- and the reduced-API
     configs inherit them verbatim ("prompt is inherited from
     FrankaPickPlaceCodeEnv class default"). At S3 the model is instructed to
     use `get_object_pose(..., return_bbox_extent=True)`, to reuse the
     quaternion from `sample_grasp_pose`, and to pass `z_approach=0.1`. None of
     the three exists at S3.

WHAT THIS DOES NOT CLAIM
------------------------
It does not claim the S3 floor is an artifact. Most S3 trials fail for reasons
that have nothing to do with this -- syntax errors, bad indexing, wrong frames
-- and a reduced API is genuinely harder. The claim is narrower and it is about
the TAG MIX, which is the deliverable: a trial that died calling a function the
prompt told it to call must not be tagged as a model defect. This script finds
those trials so the tagging pass can hold them out, and prints them as a share
of trials so the size of the hold-out is on the record rather than assumed.

Run it per tier BEFORE tagging, and quote the tier with the number.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402

#: Names the S3/S4 prompt puts in front of the model that the reduced API does
#: not bind. Sourced from the two defects above, not from a guess about what
#: models hallucinate -- a name only belongs here if the PROMPT names it.
PROMPTED_PHANTOMS = {
    "segment_sam3": "docstring example for segment_sam3_text_prompt",
    "sam3_seg_fn": "docstring example for plan_grasp",
    "get_object_pose": "S1 key rule, inherited unconditioned",
    "sample_grasp_pose": "S1 key rule, inherited unconditioned",
}

#: The ground truth we check the phantoms against, read from the checkout so a
#: binding added upstream stops being reported as missing without anyone here
#: remembering to edit this file.
REDUCED_API = Path("capx/integrations/franka/control_reduced.py")

NAME_ERR = re.compile(r"NameError: name '([^']+)' is not defined")
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M)


def bound_names(capx: Path) -> set[str]:
    p = capx / REDUCED_API
    if not p.exists():
        return set()
    return set(DEF_RE.findall(p.read_text(errors="ignore")))


def cell_dirs(scratch: Path, tier: str):
    """tier_<T>/<HF_model_dir>/<cell>/ -- two levels, not one."""
    root = scratch / f"tier_{tier}"
    if not root.exists():
        return
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for cell in sorted(model_dir.iterdir()):
            if cell.is_dir() and (cell / "aaa_done_flag").exists():
                yield cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/data/capx-out")
    ap.add_argument("--capx", default="/data/cap-x")
    ap.add_argument("--tier", action="append", dest="tiers")
    a = ap.parse_args()

    scratch, capx = Path(a.scratch), Path(a.capx)
    tiers = a.tiers or [d.name[5:] for d in sorted(scratch.glob("tier_*"))]

    bound = bound_names(capx)
    if not bound:
        print(f"[apimis] cannot read {capx / REDUCED_API} -- refusing to guess")
        print("[apimis] which names are bound. Pass --capx.")
        return 1

    still_missing = {n: why for n, why in PROMPTED_PHANTOMS.items()
                     if n not in bound}
    fixed = sorted(set(PROMPTED_PHANTOMS) - set(still_missing))
    print(f"[apimis] reduced API binds {len(bound)} names")
    if fixed:
        print(f"[apimis] NOW BOUND UPSTREAM (no longer a mismatch): {', '.join(fixed)}")
    print(f"[apimis] prompted-but-unbound: {', '.join(sorted(still_missing)) or 'none'}")

    for tier in tiers:
        cells = list(cell_dirs(scratch, tier))
        if not cells:
            continue
        print(f"\n===== {tier} -- {len(cells)} completed cell(s) =====")

        trials = affected = steered = 0
        by_name: dict[str, int] = defaultdict(int)
        hit_cells: dict[str, set] = defaultdict(set)
        steered_cells: dict[str, set] = defaultdict(set)
        other_nameerr: dict[str, int] = defaultdict(int)

        # A phantom is only reported as the CAUSE when it raised. But a program
        # that calls one and dies earlier of something else was still steered by
        # the same bad prompt, and its defect tag is just as contaminated -- the
        # 60 get_object_pose trials all died on an IndexError several lines
        # above the phantom call. Count that exposure separately rather than
        # folding it into the causal number or dropping it.
        calls = {n: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(n)}\s*\(")
                 for n in still_missing}

        for cell in cells:
            for trial in sorted(cell.glob("trial_*")):
                summary = trial / "summary.txt"
                if not summary.exists():
                    continue
                trials += 1
                code = trial / "code.py"
                src = code.read_text(errors="ignore") if code.exists() else ""
                named = [n for n, rx in calls.items() if rx.search(src)]

                m = NAME_ERR.search(summary.read_text(errors="ignore"))
                name = m.group(1) if m else None
                if name in still_missing:
                    affected += 1
                    by_name[name] += 1
                    hit_cells[name].add(cell.name)
                elif named:
                    # called a phantom, died of something else first
                    steered += 1
                    for n in named:
                        steered_cells[n].add(cell.name)
                if name and name not in still_missing:
                    other_nameerr[name] += 1

        if not trials:
            print("  no trial summaries")
            continue

        ci = wilson(affected, trials)
        print(f"  trials with a summary        {trials}")
        print(f"  died on a PROMPTED phantom   {affected}"
              f"  = {100 * ci.point:.1f}%  [{100 * ci.lo:.1f}, {100 * ci.hi:.1f}]")
        for name in sorted(by_name, key=lambda n: -by_name[n]):
            print(f"      {name:<24} {by_name[name]:>4} trial(s) in "
                  f"{len(hit_cells[name])} cell(s)  <- {still_missing[name]}")
            for c in sorted(hit_cells[name]):
                print(f"          {c}")

        sci = wilson(steered, trials)
        print(f"  called one, died elsewhere   {steered}"
              f"  = {100 * sci.point:.1f}%  [{100 * sci.lo:.1f}, {100 * sci.hi:.1f}]"
              f"   (steered, not killed)")
        for name in sorted(steered_cells, key=lambda n: -len(steered_cells[n])):
            print(f"      {name:<24} {len(steered_cells[name])} cell(s): "
                  f"{', '.join(sorted(steered_cells[name]))}")

        if other_nameerr:
            print("  other NameErrors (model invented it; the prompt did NOT "
                  "name it -- these ARE model defects):")
            for name, n in sorted(other_nameerr.items(), key=lambda kv: -kv[1]):
                print(f"      {name:<24} {n:>4}")

    print("\n[apimis] Hold the prompted-phantom trials out of the tag mix. They")
    print("[apimis] measure the prompt, not the model. The other NameErrors stay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
