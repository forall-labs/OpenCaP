"""Score a second rater's CSV against the filed tags.

    python3 scripts/score_second_rater.py artifacts/second_rater/answers_x.csv \
        [--label "gpt-5 codex, isolated, 2026-10-14"]

WHY THIS EXISTS
---------------
The same-rater reliability figures (pooled kappa 0.89, FRAME 0.67) were
computed inside a session on 2026-10-04 and the computation was not committed.
That is the pattern this project keeps paying for: a published number with no
generator cannot be re-derived, reviewed, or applied to the next rater. The
second-rater packet shipped with "scored with the same computation" and no way
to run it.

This is that computation, and it runs on `retag_session2.json` too, so the new
rater and the same-rater upper bound are produced by one code path.

WHAT IT REPORTS, IN ORDER OF WHAT THE CLAIM RESTS ON
----------------------------------------------------
1. Whether the FAMILY CONTRASTS reproduce in the second rater's own tags. The
   headline is "FRAME and INTERFACE are far more common at S3/S4"; if an
   independent rater sees that in their own labels, the claim is
   rater-independent. This is printed FIRST because it is the result, and
   because a kappa table invites reading agreement as the finding.
2. Agreement: pooled tag-level kappa, per-family kappa, per-tag kappa, and
   mean per-program Jaccard.
3. The count gap, last and labelled as the least robust figure.

Cohen's kappa here is over binary presence cells: for the pooled figure, the
60 x 18 (program, tag) matrix; per family, the 60 x 1 presence column. Computed
directly rather than via a dependency so the definition is visible.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from opencap.editclass import FAMILY  # noqa: E402

PACKET = ROOT / "artifacts" / "second_rater"
KEY = PACKET / "key.json"


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa over paired binary judgements."""
    n = len(a)
    if not n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        # Both raters said the same thing every time; kappa is undefined, and
        # reporting 0.0 would read as total disagreement.
        return None
    return (po - pe) / (1 - pe)


def filed_tags() -> dict[str, dict]:
    out = {}
    for name in ("tags_s1_hand.json", "tags_reduced.json"):
        p = ROOT / "results" / name
        if p.exists():
            out.update(json.loads(p.read_text()))
    return out


def read_rater(path: Path) -> dict[str, list[str]]:
    rows = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rid = (r.get("id") or "").strip()
            if not rid:
                continue
            raw = (r.get("tags") or "").strip()
            tags = [t.strip() for t in raw.split(";") if t.strip()]
            rows[rid] = tags
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--label", default="(unlabelled rater)")
    a = ap.parse_args()

    key = json.loads(KEY.read_text())
    filed = filed_tags()
    rater = read_rater(Path(a.answers))

    ids = [i for i in sorted(key) if i in rater]
    missing = [i for i in sorted(key) if i not in rater]
    unknown = sorted(set(rater) - set(key))
    bad = sorted({t for ts in rater.values() for t in ts if t not in FAMILY})

    print("=" * 76)
    print(f"SECOND RATER: {a.label}".center(76))
    print("=" * 76)
    print(f"\n{len(ids)} of {len(key)} packet items scored")
    if missing:
        print(f"  MISSING {len(missing)}: {', '.join(missing[:12])}"
              f"{' ...' if len(missing) > 12 else ''}")
        print("  A partial return is scorable but say so; do not quote it as 60.")
    if unknown:
        print(f"  ids not in the packet (ignored): {', '.join(unknown[:8])}")
    if bad:
        print(f"  TAGS NOT IN THE TAXONOMY: {', '.join(bad)}")
        print("  These are dropped from agreement. If there are many, the brief")
        print("  was unclear and that is a finding about the brief.")
    if not ids:
        print("\nnothing to score")
        return 1

    tiers = {i: key[i]["tier"] for i in ids}
    pid_of = {i: key[i]["pid"] for i in ids}
    f_tags = {i: set(t for t in filed.get(pid_of[i], {}).get("tags", [])
                     if t in FAMILY) for i in ids}
    r_tags = {i: set(t for t in rater[i] if t in FAMILY) for i in ids}
    fam_of = lambda ts: {FAMILY[t] for t in ts}
    s1 = [i for i in ids if tiers[i] == "S1"]
    red = [i for i in ids if tiers[i] != "S1"]

    # --- 1. does the result reproduce in the rater's OWN tags? --------------
    print("\n" + "-" * 76)
    print("1. DO THE FAMILY CONTRASTS REPRODUCE IN THIS RATER'S OWN TAGS?")
    print("-" * 76)
    print("   This is the claim. Agreement with the filed tags comes after.\n")
    print(f"   {'family':<12}{'':<6}{'S1':>22}{'S3/S4':>22}")
    for fam in ("FRAME", "INTERFACE", "TRAJECTORY", "SEMANTIC", "STRUCTURE"):
        for who, label in ((f_tags, "filed"), (r_tags, "rater")):
            ka = sum(1 for i in s1 if fam in fam_of(who[i]))
            kb = sum(1 for i in red if fam in fam_of(who[i]))
            ia = wilson(ka, len(s1)) if s1 else None
            ib = wilson(kb, len(red)) if red else None
            fa = f"{ka}/{len(s1)} {ia.point:.2f} [{ia.lo:.2f},{ia.hi:.2f}]  " if ia else "--"
            fb = f"{kb}/{len(red)} {ib.point:.2f} [{ib.lo:.2f},{ib.hi:.2f}]" if ib else "--"
            name = fam if label == "filed" else ""
            print(f"   {name:<12}{label:<6}{fa:>22}{fb:>22}")
        print()

    # --- 2. agreement ------------------------------------------------------
    print("-" * 76)
    print("2. AGREEMENT WITH THE FILED TAGS")
    print("-" * 76)
    tag_names = sorted(FAMILY)
    A = [t in f_tags[i] for i in ids for t in tag_names]
    B = [t in r_tags[i] for i in ids for t in tag_names]
    k = kappa(A, B)
    jac = [len(f_tags[i] & r_tags[i]) / max(len(f_tags[i] | r_tags[i]), 1)
           for i in ids]
    print(f"\n   pooled tag-level kappa  {k if k is None else f'{k:.2f}'}"
          f"   (same-rater upper bound: 0.89)")
    print(f"   mean per-program Jaccard {sum(jac)/len(jac):.2f}"
          f"   (same-rater: 0.89)")
    print(f"\n   {'family':<14}{'kappa':>8}   (same-rater)")
    prior = {"SEMANTIC": "1.00", "SYNTAX": "1.00", "STRUCTURE": "0.96",
             "TRAJECTORY": "0.96", "INTERFACE": "0.90", "FRAME": "0.67"}
    for fam in sorted(set(FAMILY.values())):
        fa = [fam in fam_of(f_tags[i]) for i in ids]
        fb = [fam in fam_of(r_tags[i]) for i in ids]
        kf = kappa(fa, fb)
        print(f"   {fam:<14}{'n/a' if kf is None else f'{kf:>8.2f}'}"
              f"       {prior.get(fam, '-')}")
    print(f"\n   {'tag':<22}{'kappa':>8}{'filed':>7}{'rater':>7}")
    for t in tag_names:
        fa = [t in f_tags[i] for i in ids]
        fb = [t in r_tags[i] for i in ids]
        kt = kappa(fa, fb)
        if sum(fa) == 0 and sum(fb) == 0:
            continue
        print(f"   {t:<22}{'n/a' if kt is None else f'{kt:>8.2f}'}"
              f"{sum(fa):>7}{sum(fb):>7}")

    # --- 3. the count gap, last ---------------------------------------------
    print("\n" + "-" * 76)
    print("3. THE COUNT GAP -- the least robust figure, reported last")
    print("-" * 76)
    for who, label in ((f_tags, "filed"), (r_tags, "rater")):
        ma = sum(len(who[i]) for i in s1) / max(len(s1), 1)
        mb = sum(len(who[i]) for i in red) / max(len(red), 1)
        print(f"   {label:<6} mean causes per program:  S1 {ma:.2f}   "
              f"S3/S4 {mb:.2f}   gap {mb-ma:+.2f}")
    print("\n   The same-rater re-tag already moved this (3.43 -> 2.93 on")
    print("   S3/S4). If it moves again, say so and lead with the family")
    print("   contrasts, which is what the write-up already does.")
    print("\n   Record the model, the date, and whether the rater ran isolated")
    print("   or inside the repo. An unisolated run is not a blind rating.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
