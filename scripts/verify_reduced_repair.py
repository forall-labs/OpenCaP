"""Verify a hand-written MINIMAL repair of an S3/S4 program on the measured bar.

    python3 scripts/verify_reduced_repair.py cube_lifting --tier S3 \
        --pid 863aab6689bb --code fix.py [--save] [--note "..."]

A minimal repair keeps the model's program and fixes only what is broken, so
the diff between the two is the defect set -- the same evidence the S1 tags
come from (`opencap/editclass.py`). The template repairs `spill_wipe_batch.py`
saves are NOT that: they are byte-identical across models, so accepting one
says nothing about the program it is filed under.

THE BAR is `accepted()` in spill_wipe_batch.py, unchanged: no scene fails in
all k draws. What this script chooses is k, which RESUME.md leaves open ("raise
k to 5 or 8 for a program sitting mid-range"). The choice is MEASURED, from the
references' own per-scene failure rates, as the chance that the reference
itself would be rejected:

    cube_lifting  20 draws, 86.8%   k=3 0.24 | k=5 0.03 | k=8 0.00
    cube_stack     6 draws, 64.4%   k=3 0.98 | k=5 0.81 | k=8 0.43
    cube_stack    15 draws, 66.9%   k=8 0.24 | k=10 0.11 | k=12 0.05 | k=14 0.03
      (v7pg: the plan_grasp program cube_stack repairs are held to; 2026-10-04)

    cube_stack    15 draws, 60.2%   k=12 0.13 | k=14 0.08 | k=16 0.04
      (v7pg_text: v7pg with SAM3 TEXT-prompt masks instead of molmo point ->
      SAM3 point; scene 28 fails 73% of draws against v7pg's 20%; 2026-10-04)

A cube_stack program is held to the reference of ITS OWN perception route
(`k_for`): which route to take is the model's choice, not a defect, and the
text route is measurably harder on scene 28 (cubes close to the base). Text
route -> 16, molmo route -> 12.

THE RULE: the smallest k at which the reference is rejected <= 5% of the time
(`p_reference_rejected`; `python3 scripts/verify_reduced_repair.py kref
<label_prefix>` prints it from the reference's draws).

cube_stack repairs keep the model's plan_grasp (Contact-GraspNet), so they are
judged against v7pg -- plan_grasp plus the v7 placement -- not against the
analytic-grasp v7 at 90/90. At k=8, v7pg is rejected 24% of the time: its
worst scenes (16, 6, 8, 24) fail 60-73% of draws, and one of them failing all
8 is common. Hence 12.

k=3 suits the homogeneous-scene table in RESUME.md, but the scenes are not
homogeneous: cube_lifting scene 29 fails in 7 of 13 template draws, so k=3
rejects byte-identical correct code a quarter of the time. Hence 5. cube_stack
at 8 still rejects its own reference 43% of the time -- that is a statement
about the reference, and it has to be raised before cube_stack repairs can be
judged at all.

Draws stop EARLY only on acceptance: once every scene has passed somewhere,
more draws cannot make one fail in all of them, so the verdict is fixed.

THERE IS NO EARLY REJECTION ON RATE. The first version stopped after 3 draws
when the pooled rate sat below the reference's (Fisher p<0.001) and called that
verdict-preserving. It was not: the bar is about scenes, not rate, and it
rejected qwen3vl-8b's repair at 64/90 while qwen3-32b's original passed the bar
at 60/90. The rate comparison is still computed and RECORDED
(`below_reference`), because a repair the bar accepts at ~70% against an 87%
reference may still carry a uniform defect -- a missing approach move, say --
that no single scene exposes. That is a flag for the write-up, not a verdict.

Draws already completed for the SAME code (same sha, same label) are reused
rather than re-run, so an interrupted or re-queued verification resumes.

Every attempt is appended to results/reduced_verify.json, so iterations are
recorded rather than overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

ORACLE = Path("/data/capx-out/reduced_gate0/oracle")
LOG = ROOT / "results" / "reduced_verify.json"

#: Draws per task. spill_wipe never grasps and is deterministic (one draw is a
#: measurement); see the module docstring for the others.
K_MAX = {"spill_wipe": 1, "cube_lifting": 5, "cube_stack": 12, "nut_assembly": 8}
#: cube_stack programs that segment by SAM3 text prompt (v7pg_text reference).
K_MAX_CUBE_STACK_TEXT = 16

#: Pooled reference rates, (successes, trials), for the early-rejection test.
#: cube_lifting: the qwen35-27b control (4 draws) plus 16 draws of the LIFT
#: template, which is the same program condensed. cube_stack: v1 + v2, three
#: draws each (p=0.436 between them). spill_wipe: the REFERENCE, 90/90.
#: nut_assembly has no reference -- Gate 0 has not cleared -- so no early stop.
REFERENCE_RATE = {"cube_lifting": (521, 600), "cube_stack": (116, 180),
                  "spill_wipe": (90, 90)}


def _batch_module():
    spec = importlib.util.spec_from_file_location(
        "_swb", ROOT / "scripts" / "spill_wipe_batch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def clearly_below_reference(task: str, succ: int, trials: int) -> bool:
    """One-sided Fisher: is this pooled rate below the reference's at p<0.001?"""
    ref = REFERENCE_RATE.get(task)
    if ref is None or trials == 0:
        return False
    from scipy.stats import fisher_exact
    rs, rt = ref
    _, p = fisher_exact([[succ, trials - succ], [rs, rt - rs]], alternative="less")
    return bool(p < 1e-3)        # a numpy bool_ is not JSON-serialisable


def p_reference_rejected(fail_rates: dict[int, float], k: int) -> float:
    """Chance the bar rejects the REFERENCE itself at k draws: some scene fails all k.

    fail_rates are the reference's per-scene failure rates over its own draws;
    the scenes are treated as independent. This is how K_MAX is chosen.
    """
    keep = 1.0
    for f in fail_rates.values():
        keep *= 1.0 - f ** k
    return 1.0 - keep


def reference_fail_rates(label_prefix: str, trials: int = 30,
                         oracle: Path | None = None) -> tuple[int, dict[int, float]]:
    """(n draws, per-scene failure rate) over completed runs <prefix>_d<i>."""
    root = oracle or ORACLE
    fails: dict[int, int] = {s: 0 for s in range(1, trials + 1)}
    n = 0
    for rd in sorted(root.glob(f"{label_prefix}_d*")):
        if not re.fullmatch(re.escape(label_prefix) + r"_d\d+", rd.name):
            continue
        d = reuse_draw(rd.name, trials, root)
        if d is None:
            continue
        n += 1
        for s in d["fails"]:
            fails[s] += 1
    return n, ({s: c / n for s, c in fails.items()} if n else {})


def k_for(task: str, code: str) -> int:
    """k_max for a program: cube_stack depends on the program's perception route."""
    if task == "cube_stack":
        own = code.split("# ---- end repair kit ----")[-1]   # the model's code, not the kit's
        # The route is where the MASKS come from. A program may call molmo only to
        # check the cube exists (b9a6c83ddbc4) and still segment by text.
        if "segment_sam3_text_prompt" in own and "segment_sam3_point_prompt" not in own:
            return K_MAX_CUBE_STACK_TEXT
    return K_MAX[task]


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:10]


def run_draw(task: str, tier: str, code_file: Path, label: str, trials: int) -> dict:
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reduced_api_gate0.py"), task,
         "--tier", tier, "--code", str(code_file), "--label", label,
         "--trials", str(trials)],
        capture_output=True, text=True)
    m = re.search(r"\[gate0r\] (\d+)/(\d+)", p.stdout or "")
    rd = ORACLE / label
    out = {"label": label, "score": int(m.group(1)) if m else None}
    if m is None or not rd.exists():
        out["error"] = ((p.stdout or "") + (p.stderr or ""))[-800:]
        return out
    outcomes, crashed = {}, []
    for d in rd.glob("trial_*"):
        mm = re.match(r"trial_(\d+)_sandboxrc_(-?\d+)_reward_[\d.]+_taskcompleted_(\d)", d.name)
        if mm:
            outcomes[int(mm.group(1))] = mm.group(3) == "1"
            if mm.group(2) != "0":
                crashed.append(int(mm.group(1)))
    out["outcomes"] = outcomes
    out["fails"] = sorted(s for s, ok in outcomes.items() if not ok)
    out["crashed"] = sorted(crashed)
    return out


def _parse_run_dir(rd: Path) -> tuple[dict[int, bool], list[int]]:
    outcomes, crashed = {}, []
    for d in rd.glob("trial_*"):
        mm = re.match(r"trial_(\d+)_sandboxrc_(-?\d+)_reward_[\d.]+_taskcompleted_(\d)", d.name)
        if mm:
            outcomes[int(mm.group(1))] = mm.group(3) == "1"
            if mm.group(2) != "0":
                crashed.append(int(mm.group(1)))
    return outcomes, sorted(crashed)


def reuse_draw(label: str, trials: int, oracle: Path | None = None) -> dict | None:
    """A completed draw already on disk for this exact label, or None.

    The label carries the code's sha, so a hit is the same program."""
    rd = (oracle or ORACLE) / label
    if not (rd / "aaa_done_flag").exists():
        return None
    outcomes, crashed = _parse_run_dir(rd)
    if len(outcomes) != trials:
        return None
    return {"label": label, "score": sum(outcomes.values()), "outcomes": outcomes,
            "fails": sorted(s for s, ok in outcomes.items() if not ok),
            "crashed": crashed, "reused": True}


def verify(task: str, tier: str, pid: str, code_file: Path, trials: int,
           k_max: int) -> dict:
    swb = _batch_module()
    code = code_file.read_text()
    sha = code_sha(code)
    draws, scores, rec_draws = [], [], []
    stopped = "k_max"
    for di in range(1, k_max + 1):
        label = f"min_{task}_{tier}_{pid}_{sha}_d{di}"
        d = reuse_draw(label, trials) or run_draw(task, tier, code_file, label, trials)
        if d["score"] is None:
            print(f"   draw {di}: FAILED TO RUN\n{d.get('error','')}")
            return {"ok": False, "error": d.get("error", ""), "sha": sha}
        draws.append(d["outcomes"])
        scores.append(d["score"])
        rec_draws.append({k: d[k] for k in ("label", "score", "fails", "crashed")})
        print(f"   draw {di}: {d['score']}/{trials}  fails={d['fails']}"
              + ("  (reused)" if d.get("reused") else "")
              + (f"  CRASHED={d['crashed']}" if d["crashed"] else ""))
        ok, always = swb.accepted(draws, trials)
        if ok:
            stopped = "accepted"
            break
    ok, always = swb.accepted(draws, trials)
    succ, tot = sum(scores), trials * len(scores)
    return {"ok": True, "sha": sha, "accepted": ok, "always_fail": always,
            "scores": scores, "k_used": len(scores), "k_max": k_max,
            "stopped": stopped, "draws": rec_draws,
            "pooled": [succ, tot],
            "below_reference": clearly_below_reference(task, succ, tot)}


def log_attempt(pid: str, entry: dict, path: Path = LOG) -> None:
    hist = json.loads(path.read_text()) if path.exists() else {}
    hist.setdefault(pid, []).append(entry)
    path.write_text(json.dumps(hist, indent=2, sort_keys=True))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "kref":
        # python3 scripts/verify_reduced_repair.py kref <label_prefix> [k ...]
        n, fr = reference_fail_rates(sys.argv[2])
        ks = [int(x) for x in sys.argv[3:]] or [3, 5, 8, 10, 12, 16]
        worst = sorted(fr.items(), key=lambda kv: -kv[1])[:6]
        print(f"[kref] {sys.argv[2]}: {n} draws, rate {1 - sum(fr.values()) / max(len(fr), 1):.3f}, "
              f"worst scenes {[(s, round(f, 2)) for s, f in worst]}")
        print("[kref] P(reference rejected): " + " | ".join(
            f"k={k} {p_reference_rejected(fr, k):.2f}" for k in ks))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--tier", required=True, choices=["S3", "S4"])
    ap.add_argument("--pid", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--k-max", type=int, default=0)
    ap.add_argument("--save", action="store_true",
                    help="on acceptance, save to results/minimal/<task>.json")
    ap.add_argument("--no-defect", action="store_true",
                    help="the code IS the original: record 'no defect found'")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    code_file = Path(a.code)
    k_max = a.k_max or k_for(a.task, code_file.read_text())
    print(f"[vrr] {a.task} {a.tier} {a.pid} <- {code_file.name}  (k_max={k_max})")
    r = verify(a.task, a.tier, a.pid, code_file, a.trials, k_max)
    if not r["ok"]:
        return 1
    entry = {"task": a.task, "tier": a.tier, "code_file": str(code_file),
             "when": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
             "no_defect": a.no_defect, "note": a.note,
             **{k: r[k] for k in ("sha", "accepted", "always_fail", "scores",
                                  "k_used", "k_max", "stopped", "draws",
                                  "pooled", "below_reference")}}
    log_attempt(a.pid, entry)
    print(f"[vrr] scores={r['scores']} accepted={r['accepted']} "
          f"stopped={r['stopped']} below_reference={r['below_reference']}"
          + (f" always_fail={r['always_fail']}" if r["always_fail"] else ""))
    if r["accepted"] and a.save:
        res = save_accepted(a.task, a.tier, a.pid, code_file.read_text(), r,
                            a.no_defect, a.note)
        print(f"[vrr] saved -> {res['path']} ({res['entries']} entries)")
    return 0


def save_accepted(task: str, tier: str, pid: str, code: str, r: dict,
                  no_defect: bool, note: str, root: Path = ROOT) -> dict:
    """File an accepted repair in results/minimal/<task>.json, tier recorded.

    A program that passes the bar AS WRITTEN is stored the way S1 stored its
    no-defect programs: code None, level L4_scene_limited. tag_episodes.py reads
    code None as "no defect found" rather than diffing a program to itself.
    """
    spec = importlib.util.spec_from_file_location(
        "_save_repair", root / "scripts" / "save_repair.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    verdict = (f"Accepted on the measured bar: no scene fails in all "
               f"{r['k_used']} draw(s) (k_max {r['k_max']}); scores {r['scores']}; "
               f"sha {r['sha']}.")
    if no_defect:
        return m.save(task, pid, None, minimal=True, root=root, tier=tier,
                      level="L4_scene_limited",
                      note=f"NO DEFECT FOUND -- the original passes as written. {verdict} {note}".strip())
    return m.save(task, pid, code, minimal=True, root=root, tier=tier,
                  note=f"MINIMAL hand repair (CLI session). {verdict} {note}".strip())


if __name__ == "__main__":
    raise SystemExit(main())
