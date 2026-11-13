"""Fold verification batches into results/verify_h200.json and
results/minimal_verify_h200.json.

The rebuild verified in several serial batches on GPU 4 (stage 8, the fetch
re-runs, the 55 arm repairs, second draws for the short records, the iterated
repairs, the in-place minimals). Each batch wrote its own records file; this
folds them into the two committed files, chronologically, so the committed
record for a pid is the LAST draw of the LAST code version, and every earlier
draw is kept under `history` rather than discarded -- a 29/30 followed by a
30/30 on the same code is scene noise, and the reader should see both.

Minimal versions that are the repair verbatim ("adopted") share the repair's
run: the record is copied with `adopted_from_repair: true` rather than spent
twice on the GPU.

    python3 scripts/merge_verify_records.py            # rebuild both -h200 files
    python3 scripts/merge_verify_records.py --seeded   # the -seeded pair (Phase 5)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "/data/capx-out"
# The stage-8 records as the clean run wrote them (results/verify_h200.json at
# commit 589506d), copied aside before the first merge: this script overwrites
# the repository files, so it must not read them back as its base.
STAGE8 = "/data/capx-logs"

# (batch label, records file, code iteration the batch measured)
REPAIR_BATCHES = [
    ("stage8", f"{STAGE8}/verify_h200.stage8.json", 1),
    ("new55", f"{OUT}/p4new/results.json", 1),
    ("draw2", f"{OUT}/p4draw2/repairs.json", 1),
    ("draw2_new", f"{OUT}/p4draw2/new_repairs.json", 1),
    ("iter2", f"{OUT}/p4iter/repairs.json", 2),
]
MINIMAL_BATCHES = [
    ("stage8", f"{STAGE8}/minimal_verify_h200.stage8.json", 1),
    ("draw2", f"{OUT}/p4draw2/minimal.json", 1),
    ("inplace55", f"{OUT}/p4new/minimal.json", 1),
    ("inplace55_draw2", f"{OUT}/p4new/minimal_draw2.json", 1),
    ("iter2", f"{OUT}/p4iter/minimal.json", 2),
    ("iter3", f"{OUT}/p4iter/minimal_iter3.json", 3),
    ("iter3_draw2", f"{OUT}/p4iter/minimal_iter3_draw2.json", 3),
    ("iter3b", f"{OUT}/p4iter/minimal_iter3b.json", 3),
    ("iter4", f"{OUT}/p4iter/minimal_iter4.json", 4),
    ("iter4b", f"{OUT}/p4iter/minimal_iter4b.json", 4),
]

# Phase 5: stage 8 of the seeded run wrote results/verify_seeded.json and
# results/minimal_verify_seeded.json (copied aside before the first fold, as
# above); the records at 29/30 got a second draw of the same code on GPU 6.
SEEDED_REPAIR_BATCHES = [
    ("stage8", f"{STAGE8}/verify_seeded.stage8.json", 1),
    ("draw2", f"{OUT}/p5draw2/repairs.json", 1),
]
SEEDED_MINIMAL_BATCHES = [
    ("stage8", f"{STAGE8}/minimal_verify_seeded.stage8.json", 1),
    ("draw2", f"{OUT}/p5draw2/minimal.json", 1),
    ("draw3", f"{OUT}/p5draw2/minimal_draw3.json", 1),   # 016f6392e871
    ("iter2", f"{OUT}/p5draw2/minimal_iter2.json", 2),   # c8a218af1442, 016f6392e871
]

# Draws that never reached a records file. verify_repairs.py keeps one record
# per pid in its --out file, so a re-run into the same file replaces the
# earlier draw; these are taken from the console log the batch printed.
LOG_ONLY = {
    ("repairs", "6a1443eea0cc"): [dict(
        batch="iter2", code_iteration=2, ok=False, successes=0, n=30, wall=38.9,
        error="NameError: wiping_quaternion -- the iteration script's regex "
              "swallowed the definition; fixed, re-verified in batch 3",
        source="p4_verify2.log")],
}
# (batch label, pid) pairs to leave out of a set: the record exists but the
# code it measured is not the committed one.
SKIP = {
    # the minimal was at 30/30 on its stage-8 draw; the iteration script edited
    # it alongside its repair anyway, and the committed minimal is the stage-8
    # code again.
    ("minimal", "iter2", "b21323ef80f7"),
}


def load(path) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def fold(name, batches) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for label, path, code_iter in batches:
        for r in load(path):
            r = dict(r)
            r.pop("note", None)
            pid = r["pid"]
            if (name, label, pid) in SKIP:
                continue
            prev = merged.get(pid)
            hist = list(prev.get("history", [])) if prev else []
            if prev:
                hist.append({k: prev.get(k) for k in
                             ("batch", "code_iteration", "ok", "successes",
                              "n", "wall", "run_dir", "error")
                             if prev.get(k) is not None})
            for extra in LOG_ONLY.get((name, pid), []):
                if extra["batch"] == label and extra not in hist:
                    hist.append(dict(extra))
            r.update(batch=label, code_iteration=code_iter, history=hist)
            merged[pid] = r
    return merged


def main() -> int:
    seeded = "--seeded" in sys.argv[1:]
    rep_batches = SEEDED_REPAIR_BATCHES if seeded else REPAIR_BATCHES
    min_batches = SEEDED_MINIMAL_BATCHES if seeded else MINIMAL_BATCHES
    suffix = "seeded" if seeded else "h200"
    reps: dict = {}
    mins: dict = {}
    for f in sorted(glob.glob(str(ROOT / "results/repairs/*.json"))):
        reps.update(json.loads(Path(f).read_text()))
    for f in sorted(glob.glob(str(ROOT / "results/minimal/*.json"))):
        mins.update(json.loads(Path(f).read_text()))

    rep_rec = fold("repairs", rep_batches)
    min_rec = fold("minimal", min_batches)
    adopted = 0
    for pid, m in mins.items():
        if pid in min_rec or pid not in rep_rec:
            continue
        if m["code"] == reps.get(pid, {}).get("code"):
            r = dict(rep_rec[pid])
            r["adopted_from_repair"] = True
            min_rec[pid] = r
            adopted += 1

    def order(d):
        return sorted(d.values(), key=lambda r: (r["task"], r["model"], r["pid"]))

    (ROOT / f"results/verify_{suffix}.json").write_text(
        json.dumps(order(rep_rec), indent=2))
    (ROOT / f"results/minimal_verify_{suffix}.json").write_text(
        json.dumps(order(min_rec), indent=2))
    for name, d in (("repairs", rep_rec), ("minimal", min_rec)):
        ok = [r for r in d.values() if r.get("ok")]
        full = [r for r in ok if r["successes"] >= 30]
        print(f"[merge] {name}: {len(d)} records, {len(ok)} ok, "
              f"{len(full)} at 30/30, "
              f"{len([r for r in d.values() if r.get('history')])} with history"
              + (f", {adopted} adopted" if name == "minimal" else ""))
        for r in d.values():
            if not r.get("ok") or r["successes"] < 30:
                print(f"    short/fail: {r['pid']} {r['model']} {r['task']} "
                      f"{r.get('successes')}/{r.get('n')} {r.get('error', '')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
