"""The cross-tier defect mix: S1 against S3/S4, in ONE taxonomy.

    python3 scripts/cross_tier_report.py            # all filed records
    python3 scripts/cross_tier_report.py --task cube_stack

WHY THIS COMPARISON IS VALID WHEN THE LAST ONE WAS NOT
------------------------------------------------------
The retracted comparison (STATUS 2026-10-04 00:30) ran a reduced-API static
checker over S1 code, whose API surface it was never written for. Here both
sides are tagged against a VERIFIED repair with the same tag names
(opencap/editclass.FAMILY): S1 by classify_edits over minimal repairs
(results/tags_seeded.json), S3/S4 by hand, one tag per cause the repair had to
fix (results/tags_reduced.json, scripts/tag_reduced.py). The unit is the
PROGRAM, as in tags_seeded.json.

What still differs, and is reported rather than hidden:
  * reduced-only tags (tag_reduced.NEW_TAGS) are listed apart and are not in the
    S1-comparable shares;
  * tags whose evidence says "(prompted)" follow the S1-written task prompt the
    reduced tiers inherit -- counted, and also shown with them removed;
  * nut_assembly S3/S4 records are filed against a SHARED reference that does
    not pass the bar (level L5_shared_reference); their count is printed in the
    header, and --by-task separates them;
  * tag-only truncations carry only `syntax`; --no-truncated drops them on
    both sides.

READ PER TASK. The tiers' records are not spread over tasks alike (S1 has
15/15/21/19; the reduced tiers are filed as their repairs are accepted), so a
pooled share is mostly a statement about whichever task is filed most. Use
--by-task before quoting a pooled number.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from opencap.analysis import wilson  # noqa: E402
from opencap.editclass import FAMILY  # noqa: E402

FAMILIES = ["SYNTAX", "STRUCTURE", "INTERFACE", "SEMANTIC", "TRAJECTORY", "FRAME"]


def load(root: Path = ROOT, task: str | None = None, s1_source: str = "hand") -> dict[str, list[dict]]:
    """Records by tier: S1 hand tags (results/tags_s1_hand.json, the default) or
    automated (tags_seeded.json); S3/S4 from tags_reduced.json.

    HAND is the default because S3/S4 are hand-tagged: on the same 70 S1
    programs the automated classify_edits gives 2.06 tags/program and the hand
    procedure 2.77, with low agreement (blind sample Jaccard 0.31). Comparing
    hand with automated measures the taggers, not the tiers."""
    if s1_source == "hand":
        s1 = list(json.loads((root / "results/tags_s1_hand.json").read_text()).values())
    else:
        s1 = json.loads((root / "results/tags_seeded.json").read_text())
    red = json.loads((root / "results/tags_reduced.json").read_text())
    out: dict[str, list[dict]] = {"S1": [], "S3": [], "S4": []}
    for r in s1:
        out["S1"].append({**r, "tier": "S1", "evidence": r.get("evidence", {})})
    for r in red.values():
        out[r["tier"]].append(r)
    if task:
        out = {k: [r for r in v if r["task"] == task] for k, v in out.items()}
    return out


def shares(recs: list[dict], drop_prompted: bool = False,
           comparable_only: bool = True) -> tuple[int, collections.Counter, collections.Counter]:
    """(n programs, family counts, tag counts) -- each program counted once per tag/family."""
    fam, tag = collections.Counter(), collections.Counter()
    for r in recs:
        tags = [t for t in r["tags"]
                if (not comparable_only or t in FAMILY)
                and not (drop_prompted and "(prompted)" in r.get("evidence", {}).get(t, ""))]
        tag.update(set(tags))
        fam.update({FAMILY[t] for t in tags if t in FAMILY})
    return len(recs), fam, tag


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "    -     "
    ci = wilson(k, n)
    return f"{k:3d} {k / n:5.0%} [{ci.lo:4.0%},{ci.hi:4.0%}]"


def report(groups: dict[str, list[dict]], drop_prompted: bool = False) -> str:
    cols = ["S1", "S3", "S4", "S3+S4"]
    data = dict(groups)
    data["S3+S4"] = groups["S3"] + groups["S4"]
    stats = {c: shares(data[c], drop_prompted) for c in cols}
    lines = ["".join([f"{'':22s}"] + [f"{c + ' (n=' + str(stats[c][0]) + ')':>24s}" for c in cols])]
    lines.append("FAMILY (programs with >= 1 tag in the family)")
    for f in FAMILIES:
        lines.append(f"  {f:20s}" + "".join(f"{fmt(stats[c][1][f], stats[c][0]):>24s}" for c in cols))
    lines.append("TAG")
    all_tags = sorted({t for c in cols for t in stats[c][2]},
                      key=lambda t: (FAMILIES.index(FAMILY[t]), t))
    for t in all_tags:
        lines.append(f"  {t:20s}" + "".join(f"{fmt(stats[c][2][t], stats[c][0]):>24s}" for c in cols))
    extra = collections.Counter(t for r in data["S3+S4"] for t in r["tags"] if t not in FAMILY)
    if extra:
        lines.append("REDUCED-ONLY (tag_reduced.NEW_TAGS; not in the shares above)")
        for t, v in extra.most_common():
            lines.append(f"  {t:20s} {v} program(s) at S3/S4")
    return "\n".join(lines)


def by_band(groups: dict[str, list[dict]]) -> str:
    """Mean S1-taxonomy tags per program, per size band (the headline's axis)."""
    import statistics
    from opencap.models import BY_KEY

    def band(m: str) -> str:
        p = BY_KEY[m].params_b
        return "<=2B" if p <= 2 else "3-4B" if p <= 4 else "7-9B" if p < 12 else ">=12B"
    bands = ["<=2B", "3-4B", "7-9B", ">=12B"]
    lines = [f"{'':6s}" + "".join(f"{b:>16s}" for b in bands)]
    for tier in ("S1", "S3", "S4"):
        by: dict[str, list[int]] = {b: [] for b in bands}
        for r in groups[tier]:
            n = len([t for t in r["tags"] if t in FAMILY])
            for m in r["models"]:
                by[band(m)].append(n)
        lines.append(f"{tier:6s}" + "".join(
            f"{(f'{statistics.mean(v):.2f} (n={len(v)})' if v else '-'):>16s}" for v in by.values()))
    return "\n".join(lines)


def by_task(groups: dict[str, list[dict]]) -> str:
    tasks = sorted({r["task"] for v in groups.values() for r in v})
    lines = [f"{'':14s}{'':12s}" + "".join(f"{f[:10]:>12s}" for f in FAMILIES)]
    for t in tasks:
        for col, recs in (("S1", groups["S1"]), ("S3+S4", groups["S3"] + groups["S4"])):
            rs = [r for r in recs if r["task"] == t]
            n, fam, _ = shares(rs)
            cells = "".join(f"{(f'{fam[f]}/{n} {fam[f] / n:4.0%}' if n else '-'):>12s}" for f in FAMILIES)
            lines.append(f"{t if col == 'S1' else '':14s}{col + f' n={n}':12s}{cells}")
    return "\n".join(lines)


def presence(groups: dict) -> str:
    """The cross-tier result stated WITHOUT COUNTING TAGS.

    WHY THIS VIEW EXISTS. The strongest claim in Phase 9 -- that S1's scale
    advantage largely vanishes at S3/S4 -- is measured in defect tags per
    program, and the reliability check flags exactly that quantity as the
    least robust: a blind same-rater re-tag found 2.93 causes per S3/S4
    program against 3.43, a ~15% shift. A reviewer will put those two
    sentences next to each other.

    So state the same result in a form a rater can barely disagree about.
    "Did the repair have to fix three or more distinct things?" survives a
    disagreement about whether it was three or four; "3.46 tags per program"
    does not. Family PRESENCE (>= 1 tag) is the same idea for FRAME and
    INTERFACE, and it is the quantity whose kappa was actually measured
    (0.67 and 0.90).

    If the presence view reproduces the band pattern, the headline does not
    depend on tag counting. If it does not, the count result was carrying it
    and should be demoted.
    """
    from opencap.models import BY_KEY

    # The SAME banding as by_band(), deliberately: two views of one result
    # must not disagree about which band a model is in.
    def band(m: str) -> str:
        p = BY_KEY[m].params_b
        return "<=2B" if p <= 2 else "3-4B" if p <= 4 else "7-9B" if p < 12 else ">=12B"

    out = [presence.__doc__.split("\n")[1].strip(), ""]
    bands = ["<=2B", "3-4B", "7-9B", ">=12B"]
    # Only FAMILY tags are counted, matching by_band: the reduced-only tags
    # (over_dense_path, degenerate_repetition, truncated) are not causes in
    # the S1 taxonomy and counting them here would inflate S3/S4 by fiat.
    rows = {k: [(band(r["models"][0]), set(r.get("families", [])),
                 len([t for t in r.get("tags", []) if t in FAMILY])) for r in v]
            for k, v in groups.items()}
    red = rows["S3"] + rows["S4"]

    def block(title, pick):
        lines = [title, f"    {'band':8}{'S1':>24}{'S3/S4':>24}"]
        for b in bands:
            a = [x for x in rows["S1"] if x[0] == b]
            c = [x for x in red if x[0] == b]
            if not a or not c:
                continue
            ka, kc = sum(1 for x in a if pick(x)), sum(1 for x in c if pick(x))
            ia, ic = wilson(ka, len(a)), wilson(kc, len(c))
            lines.append(
                f"    {b:8}"
                f"{f'{ka}/{len(a)} {ia.point:.2f} [{ia.lo:.2f},{ia.hi:.2f}]':>24}"
                f"{f'{kc}/{len(c)} {ic.point:.2f} [{ic.lo:.2f},{ic.hi:.2f}]':>24}")
        return "\n".join(lines)

    out.append(block("  programs with >= 1 FRAME tag",
                     lambda x: "FRAME" in x[1]))
    out.append("")
    out.append(block("  programs with >= 1 INTERFACE tag",
                     lambda x: "INTERFACE" in x[1]))
    out.append("")
    out.append(block("  programs the repair had to fix >= 3 distinct causes in",
                     lambda x: x[2] >= 3))
    out.append("")
    out.append("  Wilson 95%. The last block is the count-free form of the scale")
    out.append("  result: if its bands separate, the headline does not rest on")
    out.append("  how many causes a rater counted, only on whether there were")
    out.append("  several. Quote it alongside the per-band means, not instead.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--s1", choices=["hand", "auto"], default="hand",
                    help="S1 tag source (default hand, matching S3/S4)")
    ap.add_argument("--by-band", action="store_true",
                    help="mean defect tags per program by model-size band and tier")
    ap.add_argument("--verified-only", action="store_true",
                    help="drop S3/S4 records whose repair did not pass the bar (level L5_*)")
    ap.add_argument("--by-task", action="store_true",
                    help="family shares per task, S1 against S3+S4")
    ap.add_argument("--presence", action="store_true",
                    help="the cross-tier result without counting tags: family "
                         "presence and '>= 3 distinct causes' by band")
    ap.add_argument("--no-truncated", action="store_true",
                    help="drop tag-only truncations (records whose only tag is syntax and that have no repair)")
    a = ap.parse_args()
    g = load(task=a.task, s1_source=a.s1)
    if a.verified_only:
        g = {k: [r for r in v if not str(r.get("level") or "").startswith("L5")] for k, v in g.items()}
    if a.no_truncated:
        g = {k: [r for r in v if not (r.get("truncated") and set(r["tags"]) <= {"syntax"})] for k, v in g.items()}
    if a.by_task:
        print(by_task(g))
        return 0
    if a.by_band:
        print(by_band(g))
        return 0
    if a.presence:
        print(presence(g))
        return 0
    red = g["S3"] + g["S4"]
    shared = sum(1 for r in red if "SHARED REFERENCE" in r.get("note", "") or r.get("task") == "nut_assembly")
    print(f"task={a.task or 'all'}  S1={len(g['S1'])}  S3={len(g['S3'])}  S4={len(g['S4'])}"
          f"  (S3/S4 nut_assembly records against the unaccepted shared reference: {shared})\n")
    print(report(g))
    prompted = sum(1 for r in red for t, e in r.get("evidence", {}).items() if "(prompted)" in e)
    if prompted:
        print(f"\n--- the same with the {prompted} PROMPTED tags removed (they follow the S1-written task prompt) ---\n")
        print(report(g, drop_prompted=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
