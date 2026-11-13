"""Error bars for a corpus that has none, and FRAME at more than n=4.

    python3 scripts/sampled_report.py [1 2 3 4 5]
    python3 scripts/sampled_report.py --task cube_stack --tier S3    # A4

Reads hf:savoji/capx-nut-sampled[-<task>-<tier>]-k<K> for each draw and answers three things
the greedy corpus cannot:

  1. HOW FAR IS THE GREEDY POINT ESTIMATE FROM THE MIDDLE? Greedy gives one
     program per cell. If the sampled draws of the same cell spread widely,
     then every cross-model comparison in this project is one draw from each of
     two wide distributions -- which is, verbatim, how four of the eight
     inverted findings were produced.

  2. HOW MANY DISTINCT PROGRAMS DOES A MODEL HAVE? At temperature 0 a cell
     emits exactly 1.00 unique program. At temperature > 0 the count is a
     measure of how narrow the model's behaviour is, which is a property worth
     reporting in its own right for a policy meant to be deployed.

  3. DOES THE FRAME CLASS SURVIVE A BIGGER SAMPLE? Four programs corpus-wide
     attempt a coordinate transform, all >=8B, all wrong. That is a hypothesis,
     not a rate. Five draws give up to five times the programs, and the
     attempt test needs no repair -- it reads the program text.

Note what this does NOT do: it does not re-run the repair loop. FRAME's
"attempted" half is detectable from the program alone; its "and got it wrong"
half is, on nut_assembly, given by the score -- the panel scores 3 of 630 there,
so an attempt that succeeded would be visible as a success.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402
from opencap.editclass import _frame_ops  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.repair import program_id, program_of  # noqa: E402
from opencap.store import (StoreUnreachable, open_store,  # noqa: E402
                         store_exists)

GREEDY = "hf:savoji/capx-ab-on-seeded"

#: Default arm: the original nut_assembly/S1 draws. A4 ran cube_stack/S3.
TASK = "nut_assembly"
TIER = "S1"


def store_for(task: str, tier: str, k: int) -> str:
    """Draw k's store, scoped EXACTLY as run_sampled_arm.sh scopes it.

    The first arm (nut_assembly/S1) predates scoping and owns the unsuffixed
    names, so that pair must keep them or its five collected stores become
    invisible. Any other (task, tier) gets a suffix. If this rule and the one
    in run_sampled_arm.sh ever disagree, the report silently reads the wrong
    arm -- or an empty one -- so there is a test pinning them together.
    """
    scope = "" if (task, tier) == ("nut_assembly", "S1") else f"-{task}-{tier.lower()}"
    return f"hf:savoji/capx-nut-sampled{scope}-k{k}"


def band_of(model: str) -> str:
    p = BY_KEY[model].params_b if model in BY_KEY else 0
    return "<=2B" if p <= 2 else "3-4B" if p <= 4 else "7-9B" if p <= 9 else ">=12B"


def read(spec: str, task: str = TASK, tier: str = TIER) -> dict:
    """{model: [episodes]} for (task, tier), or {} if absent/empty."""
    # store_exists first: opening an HF store CREATES it, so a read-only
    # report that names a store nobody has collected brings an empty repo
    # into being, and an empty repo reads like a collected zero.
    if not store_exists(spec):
        return {}
    try:
        s = open_store(spec)
        out = {}
        for c in s.completed():
            if c.tier == tier and c.task == task:
                out[c.model] = list(s.read_cell(c))
        return out
    except Exception as e:
        print(f"  [warn] {spec}: {type(e).__name__}: {e}")
        return {}


def main() -> int:
    argv = list(sys.argv[1:])
    task, tier = TASK, TIER
    for flag, setter in (("--task", "task"), ("--tier", "tier")):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            argv[i:i + 2] = []
            if setter == "task":
                task = val
            else:
                tier = val
    draws = [int(x) for x in argv] or [1, 2, 3, 4, 5]
    print(f"arm: {task} / {tier}   draws {draws}")
    greedy = read(GREEDY, task, tier)
    sampled = {k: read(store_for(task, tier, k), task, tier) for k in draws}
    have = [k for k, v in sampled.items() if v]
    if not have:
        print("no sampled stores readable -- run scripts/run_sampled_arm.sh")
        return 1
    print(f"draws present: {have}")

    print(f"\n{'model':<20}{'band':<8}{'greedy':>8}{'sampled k/30 per draw':>28}"
          f"{'mean':>7}{'sd':>6}{'progs':>7}")
    print("-" * 84)
    rows = []
    for model in sorted(greedy, key=lambda m: BY_KEY[m].params_b if m in BY_KEY else 0):
        g = sum(1 for e in greedy[model] if e.get("success"))
        per = []
        progs = set()
        for k in have:
            eps = sampled[k].get(model)
            if not eps:
                continue
            per.append(sum(1 for e in eps if e.get("success")))
            for e in eps:
                code = program_of(e)
                if code:
                    progs.add(program_id(code))
        if not per:
            continue
        m = mean(per)
        sd = pstdev(per) if len(per) > 1 else 0.0
        rows.append((model, g, per, m, sd, progs))
        print(f"{model:<20}{band_of(model):<8}{g:>8}"
              f"{str(per):>28}{m:>7.1f}{sd:>6.1f}{len(progs):>7}")

    # 1. is the greedy draw representative?
    print("\nGREEDY VS THE SAMPLED MIDDLE")
    off = [abs(g - m) for _, g, _, m, _, _ in rows]
    outside = sum(1 for _, g, per, _, _, _ in rows if per and (g < min(per) or g > max(per)))
    if off:
        print(f"  mean |greedy - sampled mean| = {mean(off):.2f} scenes of 30")
        print(f"  {outside} of {len(rows)} cells have their greedy score OUTSIDE "
              f"the sampled range")
        print("  A greedy point estimate that lands outside its own sampled range is")
        print("  not a bad measurement -- greedy is a different condition -- but it is")
        print("  a measurement of one program, and rankings built from it inherit that.")

    # 2. program diversity
    print("\nPROGRAM DIVERSITY (greedy emits exactly 1.00 per cell)")
    div = Counter(len(p) for _, _, _, _, _, p in rows)
    for n, c in sorted(div.items()):
        print(f"  {c:>3} cells emitted {n} distinct program(s) across {len(have)} draws")

    # 3. FRAME at a bigger n
    print("\nFRAME: PROGRAMS THAT ATTEMPT A COORDINATE TRANSFORM")
    att: dict = defaultdict(lambda: [0, 0])          # band -> [attempts, programs]
    succeeded = 0
    for k in have:
        for model, eps in sampled[k].items():
            # Success is a property of the PROGRAM over all its episodes, not
            # of whichever episode happened to carry it first. Every episode
            # in a sampled cell shares one program, so testing e["success"]
            # on the first one scored a program off scene 1 alone: the
            # qwen3vl-32b draw that succeeded on scenes 2, 3, 14 and 26 was
            # counted as "attempted and failed", printing "0 attempts
            # succeeded" -- a false confirmation of the very hypothesis this
            # line exists to test.
            wins: dict = defaultdict(int)
            codes: dict = {}
            for e in eps:
                code = program_of(e)
                if not code:
                    continue
                pid = program_id(code)
                codes.setdefault(pid, code)
                wins[pid] += bool(e.get("success"))
            b = band_of(model)
            for pid, code in codes.items():
                att[b][1] += 1
                _, rot = _frame_ops(code)
                if rot:
                    att[b][0] += 1
                    if wins[pid]:
                        succeeded += 1
    print(f"  {'band':<8}{'attempts':>10}{'programs':>10}{'rate':>18}")
    for b in ("<=2B", "3-4B", "7-9B", ">=12B"):
        a, n = att.get(b, [0, 0])
        if not n:
            continue
        iv = wilson(a, n)
        print(f"  {b:<8}{a:>10}{n:>10}{f'{iv.point:.2f} [{iv.lo:.2f},{iv.hi:.2f}]':>18}")
    print(f"  attempts that SUCCEEDED on {task}: {succeeded} "
          f"(a program counts as succeeding if ANY of its 30 scenes did)")
    print("\n  The greedy corpus had 4 attempts, all >=8B, all wrong, and nothing")
    print("  below 8B attempting one. If the bands above separate at these n, the")
    print("  class is real; if the intervals overlap, it was four programs.")
    return 0


def _guarded(fn):
    """Run main(), but never let an unreachable hub print as a null result.

    `store_exists` raises StoreUnreachable when the hub could not answer --
    a dead token makes every private store 401, i.e. indistinguishable from
    never created. Printing "not collected" then would be a measurement
    artifact dressed as a finding, so this exits 2 and says what to fix.
    """
    def run() -> int:
        try:
            return fn()
        except StoreUnreachable as e:
            print(f"CANNOT READ THE STORES -- this says NOTHING about what was\n"
                  f"collected. Nothing below would have been a result.\n\n{e}")
            return 2
    return run

if __name__ == "__main__":
    raise SystemExit(_guarded(main)())
