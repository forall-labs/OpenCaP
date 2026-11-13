"""Unattended spill_wipe repair pass: tag the original, template the repair, verify, save.

    python3 scripts/spill_wipe_batch.py --tier S3            # or S4, or both
    python3 scripts/spill_wipe_batch.py --dry-run            # tags only, no GPU

WHAT IT CAN AND CANNOT DO
-------------------------
It cannot replace judgement. What it CAN do is the part that turned out to be
formulaic: every spill_wipe repair so far (3 of 3) needed the same small set of
fixes, and three of them are detectable statically.

So for each program it (1) TAGS the original against the checklist -- that is
the deliverable, and it is pure static analysis -- then (2) builds a repair by
keeping the model's own perception preamble and replacing the downstream with
the canonical wipe, (3) verifies it at 30 trials, (4) saves it only at 30/30.

HONEST LIMITS, stated in every record it writes:
  * a template repair is NOT a minimum edit. Distance-to-solution must not be
    computed from these without a minimisation pass. The `tier` and
    `template: true` fields mark them.
  * the TAGS come from the original program and are the trustworthy output.
  * spill_wipe only. It never grasps, so it is deterministic (2/30 scenes
    flip, the S1 baseline) and a single 30-trial run is a real measurement.
    cube_lifting/cube_stack/nut_assembly all call plan_grasp and need pooled
    draws, so they are deliberately out of scope here.
"""
from __future__ import annotations
import argparse, ast, collections, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(line_buffering=True)
except AttributeError: pass

#: The checklist, built from the three repairs done by hand. Each entry is a
#: tag name and a predicate over the ORIGINAL source.
def _has_deprojection(c: str) -> bool:
    # a world transform means multiplying by pose_mat / extrinsics
    return bool(re.search(r"(pose_mat|extrinsic\w*)\s*@|@\s*(pose_mat|extrinsic)", c))

def _has_raster(c: str) -> bool:
    return len(re.findall(r"move_to_joints", c)) >= 1 and bool(
        re.search(r"for\b.*\b(linspace|arange|range)", c, re.S))

def _downward_quat(c: str) -> bool:
    return bool(re.search(r"\[\s*0\.?0?\s*,\s*0\.?0?\s*,\s*1\.?0?\s*,\s*0\.?0?\s*\]", c))

CHECKS = {
    "frame_no_world_transform": lambda c: not _has_deprojection(c),
    "missing_coverage":         lambda c: not _has_raster(c),
    "wrong_orientation":        lambda c: not _downward_quat(c),
    "unbound_name":             lambda c: bool(re.search(
        r"\b(segment_sam3|get_object_pose|sample_grasp_pose|goto_pose|"
        r"get_3d_points_from_mask|goto_home_joint_position)\s*\(", c)),
    "never_invoked":            lambda c: '__main__' in c or bool(
        re.search(r"^def \w+\(", c, re.M) and not re.search(r"^\w+\(\)", c, re.M)),
    "pixel_as_world":           lambda c: bool(re.search(r"\[\"box\"\]|\['box'\]", c)),
}

WIPE = '''
_m = np.asarray(_mask); _m = _m[:, :, 0] if _m.ndim == 3 else _m
_ys, _xs = np.where(_m > 0)
_z = np.asarray(_depth)[_ys, _xs].astype(np.float64).reshape(-1)
_ok = _z > 0; _ys, _xs, _z = _ys[_ok], _xs[_ok], _z[_ok]
if len(_z) == 0: raise RuntimeError("mask had no valid depth")
_cam = np.stack([(_xs - _K[0, 2]) * _z / _K[0, 0],
                 (_ys - _K[1, 2]) * _z / _K[1, 1], _z], axis=-1)
_pts = (_E @ np.hstack([_cam, np.ones((len(_cam), 1))]).T).T[:, :3]
_x0, _x1 = np.percentile(_pts[:, 0], [2, 98]); _y0, _y1 = np.percentile(_pts[:, 1], [2, 98])
_P = 0.018; _x0, _x1, _y0, _y1 = _x0-_P, _x1+_P, _y0-_P, _y1+_P
_zt = float(np.median(_pts[:, 2])); _down = np.array([0.0, 0.0, 1.0, 0.0])
_gx = np.linspace(_x0, _x1, 14); _gy = np.linspace(_y0, _y1, 14)
_path = []
for _i, _y in enumerate(_gy):
    for _x in (_gx if _i % 2 == 0 else _gx[::-1]): _path.append(np.array([_x, _y, _zt]))
for _i, _x in enumerate(_gx):
    for _y in (_gy if _i % 2 == 0 else _gy[::-1]): _path.append(np.array([_x, _y, _zt]))
for _p in _path:
    try: _j = solve_ik(_p, _down)
    except Exception: continue
    move_to_joints(_j)
'''

LIFT = '''import numpy as np
from scipy.spatial.transform import Rotation
obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
pr = point_prompt_molmo(rgb, "red cube")
pc = pr.get("red cube", (None, None))
if pc is None or tuple(pc) == (None, None):
    raise RuntimeError("molmo could not find the red cube")
res = segment_sam3_point_prompt(rgb, pc)
if not res:
    raise RuntimeError("sam3 returned no mask")
mask = max(res, key=lambda d: d["score"])["mask"]
poses, scores = plan_grasp(depth, K, mask)
if len(poses) == 0:
    raise RuntimeError("no grasp candidates")
T = E @ poses[int(np.argmax(scores))]
gp = T[:3, 3]
gq = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)   # xyzw -> wxyz
move_to_joints(solve_ik(gp + np.array([0.0, 0.0, 0.10]), gq))
open_gripper()
move_to_joints(solve_ik(gp, gq))
close_gripper()
move_to_joints(solve_ik(gp + np.array([0.0, 0.0, 0.15]), gq))
'''

PREAMBLE = '''import numpy as np
_obs = get_observation()
_rgb = _obs["robot0_robotview"]["images"]["rgb"]
_depth = _obs["robot0_robotview"]["images"]["depth"]
_K = _obs["robot0_robotview"]["intrinsics"]
_E = _obs["robot0_robotview"]["pose_mat"]
_res = segment_sam3_text_prompt(_rgb, text_prompt=%(prompt)r)
if not _res:
    _pr = point_prompt_molmo(_rgb, %(prompt)r)
    _pc = _pr.get(%(prompt)r, (None, None))
    if _pc is None or tuple(_pc) == (None, None):
        raise RuntimeError("could not locate the spill")
    _res = segment_sam3_point_prompt(_rgb, _pc)
if not _res: raise RuntimeError("no mask for the spill")
_mask = max(_res, key=lambda d: d["score"])["mask"]
'''

def tags_for(code: str) -> list[str]:
    return sorted(k for k, f in CHECKS.items() if f(code))

def prompt_of(code: str) -> str:
    m = re.search(r'["\']((?:[a-z]+ )?brown spill|spill)["\']', code)
    return m.group(1) if m else "brown spill"

def build_repair(code: str) -> str:
    return PREAMBLE % {"prompt": prompt_of(code)} + WIPE

#: Owner decision 2026-10-04: oversized programs are KEPT in the corpus with a
#: note, not excluded. S1 excluded its 8; we do not. The two sub-cases differ:
#:   * degenerate repetition / merely long -> tag AND repair normally
#:   * unparseable (ends mid-token)        -> TAG ONLY. Completing a program the
#:     model never finished is authoring, not repairing, and a "repair" of it
#:     would not measure the model's defect. These carry a `truncated` tag.
def degenerate(code: str) -> bool:
    ls = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    return bool(ls) and collections.Counter(ls).most_common(1)[0][1] >= 5

#: spill_wipe never grasps, so it is deterministic and one draw is a
#: measurement. The other three call plan_grasp, which resamples: the
#: qwen35-27b cube_lifting program scores 27/26/28 with ZERO systematically
#: failing scenes, so `score == trials` is unreachable by CORRECT code and
#: would send it back for rework forever.
#:
#: The measured bar instead: a repair is accepted when NO SCENE FAILS IN ALL
#: k DRAWS -- the line between "a defect that always breaks scene X" and "the
#: sampler sometimes misses". k from the false-positive rate over 30 scenes:
#:     p>=0.80 -> k=3 (0.24 expected)   p~0.65 -> k=5 (0.16)   p~0.5 -> k=8 (0.12)
DETERMINISTIC = {"spill_wipe"}
DEFAULT_DRAWS = {"spill_wipe": 1}


def scene_outcomes(run_dir: Path) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for d in run_dir.glob("trial_*"):
        m = re.match(r"trial_(\d+)_sandboxrc_\d+_reward_[\d.]+_taskcompleted_(\d)", d.name)
        if m:
            out[int(m.group(1))] = m.group(2) == "1"
    return out


def accepted(draws: list[dict[int, bool]], trials: int) -> tuple[bool, list[int]]:
    """Accept when every scene passed at least once across the draws."""
    if not draws:
        return False, []
    scenes = sorted(set().union(*(set(d) for d in draws)))
    always_fail = [s for s in scenes if not any(d.get(s) for d in draws)]
    enough = len(scenes) >= trials
    return (not always_fail and enough), always_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="S3")
    ap.add_argument("--task", default="spill_wipe")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "spill_wipe_batch.json"))
    # A2's failing programs are a DIFFERENT population at the same tier, and
    # they must not overwrite the baseline export. `results/failures_s3.json`
    # is the baseline's; the frame-hint arm exports to its own file and names
    # it here. Defaulting to the old path keeps every existing invocation.
    ap.add_argument("--failures", default="",
                    help="failures JSON to repair (default: "
                         "results/failures_<tier>.json, i.e. the baseline)")
    ap.add_argument("--draws", type=int, default=0,
                    help="verification draws. 0 = pick per task: 1 for "
                         "spill_wipe (deterministic), 3 for grasping tasks.")
    a = ap.parse_args()

    fpath = Path(a.failures) if a.failures else (
        ROOT / f"results/failures_{a.tier.lower()}.json")
    if not fpath.exists():
        print(f"[batch] no such failures file: {fpath}")
        print(f"[batch] export it first, e.g.")
        print(f"[batch]   python3 scripts/export_failures.py {fpath} "
              f"<store> --tier {a.tier}")
        return 1
    recs = [r for r in json.loads(fpath.read_text()) if r["task"] == a.task]
    saved = json.loads((ROOT / f"results/repairs/{a.task}.json").read_text())
    out_path = Path(a.out)
    log = json.loads(out_path.read_text()) if out_path.exists() else {}

    print(f"[batch] {len(recs)} {a.task} programs at {a.tier} from {fpath.name}")
    for i, r in enumerate(recs, 1):
        pid, model, code = r["pid"], r["models"][0], r["code"]
        if pid in log and log[pid].get("score") == a.trials:
            print(f"[{i}/{len(recs)}] {model}: already at ceiling, skip"); continue
        t = tags_for(code)
        patho = degenerate(code) or len(code.splitlines()) >= 100
        try: ast.parse(code); parses = True
        except Exception: parses = False
        if not parses:
            t = sorted(set(t) | {"truncated"})
        elif patho:
            t = sorted(set(t) | {"degenerate_repetition"})
        # Only the UNPARSEABLE are skipped now, and they are still tagged.
        entry = {"model": model, "tier": a.tier, "tags": t,
                 "tag_only": not parses, "oversized": bool(patho),
                 "parses": parses, "lines": len(code.splitlines())}
        print(f"[{i}/{len(recs)}] {model:18s} tags={','.join(t) or '-'}"
              f"{'  TAG-ONLY (truncated)' if entry['tag_only'] else ''}"
              f"{'  oversized' if entry['oversized'] and parses else ''}")
        if not a.dry_run and not entry["tag_only"]:
            rep = ROOT / "results" / "reduced_gate0" / f"_batch_{a.task}_{model}.py"
            rep.write_text(build_repair(code) if a.task == "spill_wipe" else LIFT)
            k = a.draws or DEFAULT_DRAWS.get(a.task, 3)
            scores, draws = [], []
            for di in range(1, k + 1):
                label = f"batch_{a.task}_{a.tier}_{model}" + (f"_d{di}" if k > 1 else "")
                p = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/reduced_api_gate0.py"), a.task,
                     "--tier", a.tier, "--code", str(rep), "--label", label,
                     "--trials", str(a.trials)],
                    capture_output=True, text=True)
                m = re.search(r"\[gate0r\] (\d+)/(\d+)", p.stdout or "")
                scores.append(int(m.group(1)) if m else None)
                rd = Path(f"/data/capx-out/reduced_gate0/oracle/{label}")
                if rd.exists():
                    draws.append(scene_outcomes(rd))
                # a draw that already passes every scene cannot be improved on
                if scores[-1] == a.trials:
                    break
            entry["scores"] = scores
            entry["score"] = max([x for x in scores if x is not None], default=None)
            ok, always_fail = accepted(draws, a.trials)
            entry["accepted"] = ok
            entry["always_failing_scenes"] = always_fail
            print(f"      -> {scores}  accepted={ok}"
                  + (f"  always-fail {always_fail}" if always_fail else ""))
            if ok and pid not in saved:
                subprocess.run([sys.executable, str(ROOT / "scripts/save_repair.py"),
                    "--task", a.task, "--tier", a.tier, "--pid", pid,
                    "--code-file", str(rep),
                    "--note", f"TEMPLATE repair, NOT minimised -- do not compute distance "
                              f"from this without a minimisation pass. Original tags: "
                              f"{','.join(t)}. Accepted on the measured bar (no scene "
                              f"fails in all {len(draws)} draw(s)); scores {scores}."],
                    capture_output=True, text=True)
                entry["saved"] = True
        log[pid] = entry
        out_path.write_text(json.dumps(log, indent=2, sort_keys=True))
    mix = collections.Counter(t for e in log.values() for t in e["tags"])
    n = len(log)
    print(f"\n[batch] TAG MIX over {n} programs (static, on the ORIGINALS):")
    for k, v in mix.most_common():
        print(f"    {k:26s} {v:3d}  {v/n*100:4.0f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
