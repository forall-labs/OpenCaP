"""Which defects belong to which models -- the "how to improve small models" view.

    python3 scripts/defect_by_scale.py [--tags results/tags_seeded.json]

Reads results/episode_tags.json (defects derived from verified minimal repairs)
and reports them per checkpoint, ordered by size.

A caveat the table cannot show: each model contributes at most three programs
here (one per task), so a defect either appears for a model or it does not.
These are counts of programs, not rates over a sample, and nothing about their
distribution across scale is estimated from more than three observations per
model.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.editclass import FAMILY  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


BANDS = (("<=2B", 0, 2), ("3-4B", 2, 4), ("7-9B", 4, 9), (">=12B", 9, 1e9))


def band_of(model: str) -> str:
    p = BY_KEY[model].params_b if model in BY_KEY else 0
    for name, lo, hi in BANDS:
        if lo < p <= hi:
            return name
    return BANDS[0][0]


def main() -> int:
    argv = sys.argv[1:]
    tags_path = ROOT / "results/episode_tags.json"
    if "--tags" in argv:
        tags_path = Path(argv[argv.index("--tags") + 1])
    recs = json.loads(tags_path.read_text())
    print(f"tags: {tags_path}  ({len(recs)} programs)\n")
    per_model: dict[str, Counter] = defaultdict(Counter)
    fam_model: dict[str, Counter] = defaultdict(Counter)
    progs: Counter = Counter()
    for r in recs:
        for m in r["models"]:
            progs[m] += 1
            for t in r["tags"]:
                per_model[m][t] += 1
            for f in r["families"]:
                fam_model[m][f] += 1

    models = sorted(per_model, key=lambda m: BY_KEY[m].params_b if m in BY_KEY else 0)
    # FRAME last, and it is n=4 corpus-wide: only four programs attempt a
    # coordinate transform at all. Read that column as a count, not a rate.
    fams = ["STRUCTURE", "INTERFACE", "SEMANTIC", "TRAJECTORY", "FRAME"]

    print("FAILING PROGRAMS CARRYING EACH FAMILY (of up to 3 per model)")
    print(f"{'model':<19}{'size':>6}{'progs':>7}" + "".join(f"{f[:9]:>11}" for f in fams))
    print("-" * 76)
    for m in models:
        sz = f"{BY_KEY[m].params_b:g}B" if m in BY_KEY else "?"
        row = "".join(f"{fam_model[m][f]:>11}" for f in fams)
        print(f"{m:<19}{sz:>6}{progs[m]:>7}{row}")

    print("\nDEFECTS PER FAILING PROGRAM, by size")
    print(f"{'model':<19}{'size':>6}{'defects/prog':>14}  most common")
    print("-" * 76)
    for m in models:
        n = sum(per_model[m].values())
        top = ", ".join(t for t, _ in per_model[m].most_common(3))
        sz = f"{BY_KEY[m].params_b:g}B" if m in BY_KEY else "?"
        print(f"{m:<19}{sz:>6}{n/max(progs[m],1):>14.1f}  {top}")

    small = [m for m in models if m in BY_KEY and BY_KEY[m].params_b <= 3]
    large = [m for m in models if m in BY_KEY and BY_KEY[m].params_b >= 7]
    print(f"\n<=3B ({len(small)} models) vs >=7B ({len(large)} models), "
          "share of their failing programs carrying each family:")
    for f in fams:
        s = sum(fam_model[m][f] for m in small) / max(sum(progs[m] for m in small), 1)
        l = sum(fam_model[m][f] for m in large) / max(sum(progs[m] for m in large), 1)
        print(f"  {f:<12}{s:>7.0%}{l:>8.0%}   {'<-- small models worse' if s > l + .15 else ('<-- large worse' if l > s + .15 else '')}")

    # The band table. It was published out of an ad-hoc script that lived
    # nowhere, which is how "INTERFACE 69% at >=12B" survived a tagger bug --
    # nobody could regenerate it to find out. It regenerates here now.
    band: dict[str, Counter] = defaultdict(Counter)
    bprogs: Counter = Counter()
    bdef: Counter = Counter()
    bdist: dict[str, list] = defaultdict(list)
    for r in recs:
        b = band_of(r["models"][0])
        bprogs[b] += 1
        bdef[b] += len(r["tags"])
        if r.get("distance") is not None:
            bdist[b].append(r["distance"])
        for f in r["families"]:
            band[b][f] += 1
    print("\nBY BAND (one program per model per task; n is programs, not episodes)")
    print(f"{'band':<8}{'progs':>6}{'def/prog':>10}{'distance':>10}"
          + "".join(f"{f[:4]:>7}" for f in fams))
    print("-" * 76)
    for name, _, _ in BANDS:
        n = bprogs[name]
        if not n:
            continue
        d = sum(bdist[name]) / len(bdist[name]) if bdist[name] else float("nan")
        cells = "".join(f"{band[name][f]/n:>6.0%} " for f in fams)
        print(f"{name:<8}{n:>6}{bdef[name]/n:>10.2f}{d:>10.3f}  {cells}")
    print("FRAME is 4 programs corpus-wide. Read that column as a count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
