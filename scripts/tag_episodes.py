"""Tag every failed episode with the defects its program actually had.

    python3 scripts/tag_episodes.py [out.json] [--failures PATH]

The failures export was hardcoded to results/failures.json, which is the
pre-seeding corpus. results/tags_seeded.json -- the tagging of the seeded
corpus, and the thing this project calls its deliverable -- was therefore
produced outside the repo and had no generator at all: it could not be
regenerated, reviewed or re-derived after a tagger fix. It has one now.

The tag comes from the diff between the model's program and a MINIMAL repair
that was re-executed and reached the task ceiling -- so it is what the program
demonstrably needed, not what an exception message suggested.

Episodes are tagged through their program: greedy decoding means a cell's 30
episodes share one program, so 926 failed episodes carry 32 distinct diagnoses.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.editclass import classify_edits  # noqa: E402
from opencap.repair import looks_truncated, normalised_edit_distance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def best_repairs() -> dict:
    """Minimal repair where one verified; otherwise the iterated one."""
    out: dict = {}
    for f in sorted(glob.glob(str(ROOT / "results/repairs/*.json"))):
        out.update(json.loads(Path(f).read_text()))
    for f in sorted(glob.glob(str(ROOT / "results/minimal/*.json"))):
        out.update(json.loads(Path(f).read_text()))
    return out


def main() -> int:
    argv = sys.argv[1:]
    fails_path = ROOT / "results/failures.json"
    if "--failures" in argv:
        i = argv.index("--failures")
        fails_path = Path(argv[i + 1])
        del argv[i:i + 2]
    out_path = Path(argv[0]) if argv else ROOT / "results/episode_tags.json"
    fails = {x["pid"]: x for x in json.loads(fails_path.read_text())}
    reps = best_repairs()
    print(f"failures: {fails_path}  ({len(fails)} programs)")

    recs, by_tag, by_family, by_model = [], Counter(), Counter(), {}
    n_eps = 0
    for pid, f in fails.items():
        rep = reps.get(pid)
        if rep is None or rep["code"] is None:
            defects = []          # no defect found; the cell's majority passed
        else:
            defects = classify_edits(f["code"], rep["code"])
        rec = {
            "pid": pid, "task": f["task"], "models": f["models"],
            "n_episodes": f["n_episodes"],
            "tags": [d.tag for d in defects],
            "families": sorted({d.family for d in defects}),
            # The size of the verified minimal edit. Carried here so the
            # taxonomy and the distance metric cannot disagree about which
            # repair a program was measured against.
            "distance": (round(normalised_edit_distance(f["code"], rep["code"]), 4)
                         if rep and rep.get("code") else None),
            # Artifact #15: a program cut off at the token cap carries tags
            # that describe the cut, not the model. Flagged rather than
            # dropped -- the cell's score is still a real measurement of what
            # the harness produced, and deleting data to protect a trend is
            # how this project would earn its ninth retraction.
            "truncated": looks_truncated(f["code"]),
            "evidence": {d.tag: d.evidence for d in defects},
        }
        recs.append(rec)
        n_eps += f["n_episodes"]
        for d in defects:
            by_tag[d.tag] += f["n_episodes"]
        # Count each episode ONCE per family: a program with three semantic
        # defects is still one semantically-broken program, and counting per
        # tag pushed the shares past 100%.
        for fam in {d.family for d in defects}:
            by_family[fam] += f["n_episodes"]
        for m in f["models"]:
            by_model.setdefault(m, Counter()).update(
                {d.tag: f["n_episodes"] for d in defects})

    out_path.write_text(json.dumps(recs, indent=2))
    print(f"{len(recs)} programs / {n_eps} failed episodes tagged -> {out_path}\n")

    print("BY FAMILY (episodes counted once per family)")
    for fam, n in by_family.most_common():
        print(f"  {fam:<12}{n:>5}  {n/n_eps:>6.1%}")
    print("\nBY DEFECT")
    for tag, n in by_tag.most_common():
        print(f"  {tag:<22}{n:>5}  {n/n_eps:>6.1%}")
    untagged = sum(r["n_episodes"] for r in recs if not r["tags"])
    print(f"\nuntagged: {untagged} episodes "
          f"({untagged/n_eps:.1%}) -- programs with no defect found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
