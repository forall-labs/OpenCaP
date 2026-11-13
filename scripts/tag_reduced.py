"""Hand tags for S3/S4 programs, in the S1 taxonomy, backed by a minimal repair.

    python3 scripts/tag_reduced.py add --tier S3 --pid 863aab6689bb \
        --tag call_signature="plan_grasp given a list, not a mask" \
        --tag frame_convention="grasp pose used in camera frame"
    python3 scripts/tag_reduced.py report

WHY BY HAND
-----------
The S1 tags are computed by `opencap/editclass.py` from the diff between a
program and its verified minimal repair. Its predicates are written against
the S1 API -- `goto_pose`, `get_object_pose`, `z_approach`, a PROVIDED set of
S1 names -- and S3/S4 programs call none of those. Running it on reduced-API
code would misfire exactly the way the static checker did when it was pointed
at S1 code: the cross-tier comparison retracted on 2026-10-04.

So the tags here are assigned by reading the original against its minimal
repair, ONE TAG PER DISTINCT CAUSE THE REPAIR HAD TO FIX -- the rule
classify_edits implements -- and every tag carries an evidence line. The tag
NAMES are the S1 taxonomy's (editclass.FAMILY), so a cross-tier mix compares
like with like.

A defect the S1 taxonomy cannot name goes in NEW_TAGS with a definition. Each
one is a reduced-API-only class, and the report keeps it out of the S1-
comparable mix rather than forcing it into the nearest S1 bucket.

Conventions where the S1 definitions had to be read onto the reduced API:
  * call_signature covers misreading what an API RETURNS as well as what it
    takes -- S1 filed get_object_pose unpack arity under it, and KeyError on a
    molmo/observation dict is the same mistake.
  * frame_convention covers a camera-frame pose used as a world pose and a
    wxyz/xyzw reorder. Both are "arithmetic done in the wrong frame".
  * object_name covers the text prompt handed to molmo/SAM3 -- the reduced
    API's equivalent of the object string.
  * approach_param covers approach and lift offsets fed to solve_ik, the
    reduced API's equivalent of z_approach.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opencap.editclass import FAMILY  # noqa: E402
from opencap.repair import looks_truncated, normalised_edit_distance  # noqa: E402

OUT = ROOT / "results" / "tags_reduced.json"

#: Reduced-API-only defect classes: {tag: (family, definition)}. Kept out of
#: the S1-comparable mix. Add one only when no S1 tag fits, and say why.
NEW_TAGS: dict[str, tuple[str, str]] = {
    # spill_wipe, 2026-10-04: a 1 cm raster over a ~20 cm spill is ~440
    # waypoints per pass; the episode's step horizon ends the run mid-wipe (the
    # trial is re-run and the latest attempt scores a partial wipe). Dropping
    # ONLY the density to 14 points per axis took S4 qwen3vl-32b from 14/30 to
    # 30/30 with its own bounds and row-major comb kept. S1's missing_coverage
    # is the opposite defect (too few waypoints) and spurious_move is a move
    # that should not exist at all, so neither fits.
    "over_dense_path": ("TRAJECTORY",
                        "a waypoint path so dense the episode's step horizon ends it before the motion completes"),
}


def family_of(tag: str) -> str:
    if tag in FAMILY:
        return FAMILY[tag]
    if tag in NEW_TAGS:
        return NEW_TAGS[tag][0]
    raise SystemExit(f"[tag_reduced] unknown tag {tag!r}: not in the S1 taxonomy "
                     f"(editclass.FAMILY) and not defined in NEW_TAGS")


def _degenerate(code: str) -> bool:
    spec = importlib.util.spec_from_file_location(
        "_swb", ROOT / "scripts" / "spill_wipe_batch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.degenerate(code)


def original(tier: str, pid: str, root: Path = ROOT) -> dict:
    for r in json.loads((root / f"results/failures_{tier.lower()}.json").read_text()):
        if r["pid"] == pid:
            return r
    raise SystemExit(f"[tag_reduced] {pid} not in failures_{tier.lower()}.json")


def build_record(tier: str, pid: str, tags: dict[str, str], note: str = "",
                 root: Path = ROOT, substitutions: dict[str, str] | None = None) -> dict:
    f = original(tier, pid, root)
    task = f["task"]
    mpath = root / "results" / "minimal" / f"{task}.json"
    rep = json.loads(mpath.read_text()).get(pid) if mpath.exists() else None
    if rep is not None and rep.get("tier") != tier:
        raise SystemExit(f"[tag_reduced] minimal repair for {pid} is tier "
                         f"{rep.get('tier')}, not {tier}")
    if rep is None and "syntax" not in tags:
        raise SystemExit(f"[tag_reduced] {pid} has no verified minimal repair; only a "
                         f"tag-only `syntax` record may be filed without one")
    families = sorted({family_of(t) for t in tags})
    return {
        "pid": pid, "task": task, "tier": tier, "models": f["models"],
        "n_episodes": f["n_episodes"], "tags": sorted(tags), "families": families,
        "evidence": dict(sorted(tags.items())),
        "distance": (round(normalised_edit_distance(f["code"], rep["code"]), 4)
                     if rep and rep.get("code") else None),
        "no_defect": bool(rep and rep.get("code") is None),
        # L5_* = the repair did NOT pass the bar (nut's shared reference, or a
        # repair rejected on its scene): tags are diagnoses, not verified.
        "level": rep.get("level") if rep else None,
        "truncated": looks_truncated(f["code"]),
        "degenerate": _degenerate(f["code"]),
        "lines": len(f["code"].splitlines()),
        "tagged_by": "hand", "note": note,
        # Changes the repair needed that are NOT the model's defect -- e.g.
        # plan_grasp cannot grasp the nut's handle at all (0/30 with an otherwise
        # correct program), so every nut repair replaces it. Recorded so the
        # distance can be read honestly; never counted in the tag mix.
        "substitutions": dict(substitutions or {}),
    }


def add(tier: str, pid: str, tags: dict[str, str], note: str = "",
        root: Path = ROOT, substitutions: dict[str, str] | None = None) -> dict:
    rec = build_record(tier, pid, tags, note, root, substitutions)
    path = root / "results" / "tags_reduced.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    d[pid] = rec
    path.write_text(json.dumps(d, indent=2, sort_keys=True))
    return rec


def report(root: Path = ROOT) -> None:
    path = root / "results" / "tags_reduced.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    recs = list(d.values())
    print(f"{len(recs)} programs hand-tagged")
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["tier"], r["task"])].append(r)
    for (tier, task), rs in sorted(by.items()):
        print(f"  {tier} {task:14s} {len(rs):3d}")
    for tier in sorted({r["tier"] for r in recs}):
        rs = [r for r in recs if r["tier"] == tier]
        n = len(rs)
        fam = collections.Counter(f for r in rs for f in r["families"])
        tag = collections.Counter(t for r in rs for t in r["tags"])
        print(f"\n{tier}: {n} programs (shares are of PROGRAMS; families counted once each)")
        for k, v in fam.most_common():
            print(f"    {k:12s} {v:3d}  {v/n:6.1%}")
        for k, v in tag.most_common():
            mark = "  [reduced-only]" if k in NEW_TAGS else ""
            print(f"    {k:22s} {v:3d}  {v/n:6.1%}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("add")
    a1.add_argument("--tier", required=True, choices=["S3", "S4"])
    a1.add_argument("--pid", required=True)
    a1.add_argument("--tag", action="append", default=[],
                    help="tag=evidence; repeat per distinct cause")
    a1.add_argument("--note", default="")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "report":
        report()
        return 0
    tags = {}
    for t in a.tag:
        k, _, v = t.partition("=")
        tags[k.strip()] = v.strip()
    rec = add(a.tier, a.pid, tags, a.note)
    print(f"[tag_reduced] {rec['tier']} {rec['task']} {rec['models'][0]}: "
          f"{','.join(rec['tags']) or '(no defect)'}  distance={rec['distance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
