"""Do the repairs hold on scenes they were never written against?

    python3 scripts/run_heldout_scenes.py              # references + a sample
    python3 scripts/run_heldout_scenes.py --n 40

THE GAP THIS CLOSES
-------------------
Every repair in this corpus was accepted by re-running on trials 1..30, and
every distance was measured against that acceptance. `seed = trial index`
(scripts/freeze_states.py), so trials 31..60 are a scene set nothing in this
project has ever touched.

If the repairs were fitted to those thirty scenes -- and a repair loop that
iterates until a scene stops failing is exactly the shape of a process that
would fit them -- then the distance metric is an in-sample number and the
central artifact of the paper is overfit. It is the first thing a careful
reviewer asks and it has never been checked.

THE DESIGN
----------
Run at `--trials 60` and split the outcome: trials in the lower half are the
scenes the repair was accepted on, the upper half are held out. One run gives
both, so in-sample and held-out are measured under identical conditions --
same engine, same host, same card occupancy, same day. Running them as two
jobs would reintroduce every between-run difference this project has spent
months removing.

THE REFERENCES ARE THE CONTROL AND ARE NOT OPTIONAL. A repair scoring 26/30 on
held-out scenes means nothing until you know what the human reference scores
there. The references were themselves written against trials 1..30 -- nut v8
fails scene 14 of that set and nobody knows what it does on the next thirty --
so their held-out score IS the ceiling for this comparison, not 30/30.

    repairs hold, references hold          the corpus generalises; distance is
                                           a property of the program
    repairs drop, references hold          the repairs are fitted to the scene
                                           set and the distance metric is
                                           in-sample
    both drop                              the second thirty scenes are harder;
                                           read repairs RELATIVE to the
                                           reference, not against 30/30

LLM-free. Resumable on pid. Same execution path as the verifier.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.models import BY_KEY  # noqa: E402
from opencap.results import parse_trials  # noqa: E402
from verify_repairs import run_one  # noqa: E402

TASKS = ("cube_lifting", "cube_stack", "spill_wipe", "nut_assembly")


def band_of(model: str) -> str:
    p = BY_KEY[model].params_b if model in BY_KEY else 0
    return "<=2B" if p <= 2 else "3-4B" if p <= 4 else "7-9B" if p < 12 else ">=12B"


def load_repairs() -> dict:
    out: dict = {}
    for sub in ("repairs", "minimal"):
        for f in sorted(glob.glob(str(ROOT / "results" / sub / "*.json"))):
            out.update(json.loads(Path(f).read_text()))
    return {k: v for k, v in out.items() if v.get("code")}


def load_failures() -> dict:
    out: dict = {}
    for name in ("failures.json", "failures_seeded.json"):
        f = ROOT / "results" / name
        if f.exists():
            out.update({x["pid"]: x for x in json.loads(f.read_text())})
    return out


def sample(reps: dict, fails: dict, n: int, seed: int) -> list[str]:
    """Stratified by task x band, so the sample cannot be all easy repairs.

    Deterministic: a sample that changes between runs is not a sample, and a
    second pass has to be able to extend the first rather than replace it.
    """
    buckets: dict = {}
    for pid in reps:
        f = fails.get(pid)
        if not f or f["task"] not in TASKS:
            continue
        buckets.setdefault((f["task"], band_of(f["models"][0])), []).append(pid)
    rng = random.Random(seed)
    for v in buckets.values():
        v.sort()
        rng.shuffle(v)
    picked, i = [], 0
    keys = sorted(buckets)
    while len(picked) < n:
        added = False
        for k in keys:
            if i < len(buckets[k]) and len(picked) < n:
                picked.append(buckets[k][i])
                added = True
        if not added:
            break
        i += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="repairs to sample")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--trials", type=int, default=60,
                    help="lower half = in-sample, upper half = held out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--capx", default=os.environ.get("CAPX_DIR", "/data/cap-x"))
    ap.add_argument("--scratch-root",
                    default=os.path.join(
                        os.environ.get("OPENCAP_SCRATCH_ROOT", "/data/capx-out"),
                        "heldout"))
    ap.add_argument("--out", default=str(ROOT / "results" / "heldout_scenes.json"))
    ap.add_argument("--no-references", action="store_true",
                    help="skip the control; the result is then uninterpretable")
    a = ap.parse_args()

    reps, fails = load_repairs(), load_failures()
    targets: list[tuple[str, str, str, str, str]] = []   # pid, task, model, kind, code

    if not a.no_references:
        refs = json.loads((ROOT / "results" / "references.json").read_text())
        for task, r in sorted(refs.items()):
            if task in TASKS and r.get("code"):
                targets.append((f"reference_{task}", task, "oracle",
                                "reference", r["code"]))
    else:
        print("[heldout] WARNING: no references. A repair's held-out score "
              "cannot be read\n[heldout] without the ceiling on those scenes.")

    for pid in sample(reps, fails, a.n, a.seed):
        f = fails[pid]
        targets.append((pid, f["task"], f["models"][0], "repair", reps[pid]["code"]))

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else []
    seen = {r["pid"] for r in done}
    todo = [t for t in targets if t[0] not in seen]

    print(f"[heldout] {len(targets)} programs "
          f"({sum(1 for t in targets if t[3] == 'reference')} references), "
          f"{len(todo)} to run, {len(seen)} already recorded")
    print(f"[heldout] {a.trials} trials each; lower half in-sample, upper half held out")

    for i, (pid, task, model, kind, code) in enumerate(todo, 1):
        scratch = Path(a.scratch_root) / kind
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"[run ] {i}/{len(todo)} {pid} {model} {task} ({kind})")
        res = run_one(Path(a.capx), task, pid, code, a.trials, a.workers, scratch)
        rec = {"pid": pid, "task": task, "model": model, "kind": kind,
               "band": band_of(model) if model in BY_KEY else "-",
               "trials": a.trials, "workers": a.workers,
               "ok": res.get("ok"), "error": res.get("error"),
               "wall": res.get("wall")}
        if res.get("run_dir"):
            ts = parse_trials(res["run_dir"])
            idx = sorted(t.trial for t in ts)
            # Split on the observed indices rather than on a constant: the
            # corpus is 1..30 in one place and 0..29 in another, and guessing
            # wrong would silently swap in-sample for held-out.
            cut = idx[len(idx) // 2] if idx else 0
            rec["cut"] = cut
            rec["in_sample"] = sum(1 for t in ts if t.trial < cut and t.success)
            rec["in_sample_n"] = sum(1 for t in ts if t.trial < cut)
            rec["heldout"] = sum(1 for t in ts if t.trial >= cut and t.success)
            rec["heldout_n"] = sum(1 for t in ts if t.trial >= cut)
            rec["failed_trials"] = sorted(t.trial for t in ts if not t.success)
        done.append(rec)
        out_path.write_text(json.dumps(done, indent=2))
        print(f"[done] {pid}  in-sample {rec.get('in_sample')}/{rec.get('in_sample_n')}"
              f"  held-out {rec.get('heldout')}/{rec.get('heldout_n')}")

    print(f"[heldout] {len(done)} records -> {out_path}")
    print("[heldout] now: python3 scripts/heldout_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
