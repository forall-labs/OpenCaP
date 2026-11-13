"""Which failing programs have a repair, which are verified, which are missing.

    python3 scripts/repair_status.py [failures.json] [--verify FILE] [--strict]

WHY THIS EXISTS
---------------
The pipeline had a stage that EXPORTS failing programs and a stage that
RE-VERIFIES existing repairs, and nothing in between reported the difference.
When nut_assembly joined the panel, its 21 programs were exported and then
skipped in silence: stage 8 re-verified the 119 repairs that already existed
and never mentioned that a fifth of the corpus had none. The gap was only found
by hand-counting 123 repairs against 119 verification records.

Repairing is judgement and cannot be scripted. Reporting that repairs are
MISSING can be, and that is the whole job here. `--strict` exits non-zero when
anything is unrepaired, so a pipeline can refuse to call itself finished.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dir(name: str) -> dict:
    out = {}
    for f in glob.glob(str(ROOT / "results" / name / "*.json")):
        out.update(json.loads(Path(f).read_text()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("failures", nargs="?",
                    default=str(ROOT / "results" / "failures_seeded.json"))
    ap.add_argument("--verify", default=str(ROOT / "results" / "verify_seeded.json"))
    ap.add_argument("--minimal-verify",
                    default=str(ROOT / "results" / "minimal_verify_seeded.json"))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    fp = Path(a.failures)
    if not fp.exists():
        print(f"[repair] {fp} not found -- run stage 4 (export) first")
        return 1
    fails = {x["pid"]: x for x in json.loads(fp.read_text())}
    rep, mini = load_dir("repairs"), load_dir("minimal")

    def verified(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        return {r["pid"]: r for r in json.loads(p.read_text())}
    vr, vm = verified(a.verify), verified(a.minimal_verify)

    by_task = collections.defaultdict(lambda: collections.Counter())
    missing = collections.defaultdict(list)
    # Every way a program can be short of "repaired, minimised, both verified
    # at ceiling". The first version tracked only the first and the fourth,
    # and closed with "every failing program has a verified repair at ceiling"
    # over 21 repairs nobody had verified and 2 minimals that did not exist.
    no_minimal, unverified, min_unverified, short, min_short = [], [], [], [], []
    for pid, x in fails.items():
        t = x["task"]
        c = by_task[t]
        c["programs"] += 1
        if pid in rep:
            c["repaired"] += 1
        else:
            missing[t].append((pid, x["models"][0]))
            continue
        if pid in mini:
            c["minimal"] += 1
        else:
            no_minimal.append((pid, t))
        if pid in vr:
            c["verified"] += 1
            if vr[pid].get("successes", 0) < vr[pid].get("n", 30):
                short.append((pid, t, vr[pid]["successes"], vr[pid]["n"]))
        else:
            unverified.append((pid, t))
        if pid in vm:
            c["min_verified"] += 1
            if vm[pid].get("successes", 0) < vm[pid].get("n", 30):
                min_short.append((pid, t, vm[pid]["successes"], vm[pid]["n"]))
        elif pid in mini:
            min_unverified.append((pid, t))

    w = 14
    print(f"{'task':<{w}}{'programs':>9}{'repaired':>10}{'minimal':>9}"
          f"{'verified':>10}{'min.ver':>9}")
    tot = collections.Counter()
    for t in sorted(by_task):
        c = by_task[t]
        for k, v in c.items():
            tot[k] += v
        print(f"{t:<{w}}{c['programs']:>9}{c['repaired']:>10}{c['minimal']:>9}"
              f"{c['verified']:>10}{c['min_verified']:>9}")
    print(f"{'TOTAL':<{w}}{tot['programs']:>9}{tot['repaired']:>10}"
          f"{tot['minimal']:>9}{tot['verified']:>10}{tot['min_verified']:>9}")

    gap = tot["programs"] - tot["repaired"]
    if missing:
        print(f"\n{gap} program(s) have NO repair:")
        for t, items in sorted(missing.items()):
            print(f"  {t} ({len(items)}):")
            for pid, model in sorted(items, key=lambda x: x[1]):
                print(f"     {pid[:12]}  {model}")
    def listing(title, items):
        print(f"\n{len(items)} {title}:")
        for it in items:
            pid, t = it[0], it[1]
            score = f"  {it[2]}/{it[3]}" if len(it) > 2 else ""
            print(f"  {pid[:12]}  {t}{score}")
    if no_minimal:
        listing("repair(s) with NO minimal version -- minimise", no_minimal)
    if unverified:
        listing("repair(s) never verified -- run stage 8", unverified)
    if short:
        listing("repair(s) below ceiling -- iterate", short)
    if min_unverified:
        listing("minimal version(s) never verified -- run stage 8", min_unverified)
    if min_short:
        listing("minimal version(s) below ceiling -- iterate", min_short)
    gaps = bool(missing or no_minimal or unverified or short or min_unverified or min_short)
    if not gaps:
        print("\nevery failing program has a repair and a minimal version, both verified at ceiling.")
    return 1 if (a.strict and gaps) else 0


if __name__ == "__main__":
    raise SystemExit(main())
