"""Read results/heldout_scenes.json: is the repair corpus fitted to its scenes?

    python3 scripts/heldout_report.py [results/heldout_scenes.json]

READ THE REFERENCES FIRST. They were written against the same thirty scenes as
the repairs, so their held-out score is the ceiling for this comparison -- not
30/30. A repair that drops exactly as much as the reference has not been shown
to overfit; the second thirty scenes are simply harder.

The quantity is the DROP, in-sample minus held-out, per program, and whether
the repairs' drop exceeds the references'.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/heldout_scenes.json"
    if not path.exists():
        print(f"{path} missing -- run scripts/run_heldout_scenes.py first")
        return 1
    recs = json.loads(path.read_text())
    good = [r for r in recs if r.get("ok") and r.get("heldout_n")]
    bad = [r for r in recs if not r.get("ok")]

    print("=" * 78)
    print("HELD-OUT SCENES".center(78))
    print("=" * 78)
    print(f"\n{len(recs)} programs, {len(bad)} that did not complete")
    for r in bad:
        print(f"  DID NOT RUN {r['pid']}: {r.get('error')}")
    if not good:
        print("nothing scorable")
        return 1

    refs = [r for r in good if r["kind"] == "reference"]
    reps = [r for r in good if r["kind"] == "repair"]
    if not refs:
        print("\n  NO REFERENCES IN THIS FILE. The repairs' held-out scores are")
        print("  not interpretable without the ceiling on the same scenes.")

    def drop(r):
        return (r["in_sample"] / max(r["in_sample_n"], 1)
                - r["heldout"] / max(r["heldout_n"], 1))

    print("\nREFERENCES -- the ceiling on each half\n")
    print(f"  {'task':<15}{'in-sample':>12}{'held-out':>12}{'drop':>8}")
    for r in sorted(refs, key=lambda x: x["task"]):
        print(f"  {r['task']:<15}{f'{r[chr(39)+chr(39)] if False else r['in_sample']}/{r['in_sample_n']}':>12}"
              f"{f'{r['heldout']}/{r['heldout_n']}':>12}{drop(r):>+8.0%}")

    print("\nREPAIRS\n")
    print(f"  {'model':<20}{'task':<14}{'band':<7}{'in-sample':>11}{'held-out':>11}{'drop':>8}")
    for r in sorted(reps, key=lambda x: (x["task"], -drop(x))):
        print(f"  {r['model']:<20}{r['task']:<14}{r['band']:<7}"
              f"{f'{r['in_sample']}/{r['in_sample_n']}':>11}"
              f"{f'{r['heldout']}/{r['heldout_n']}':>11}{drop(r):>+8.0%}")

    print("\nSUMMARY\n")
    for label, group in (("references", refs), ("repairs", reps)):
        if not group:
            continue
        ins = sum(r["in_sample"] for r in group)
        inn = sum(r["in_sample_n"] for r in group)
        hos = sum(r["heldout"] for r in group)
        hon = sum(r["heldout_n"] for r in group)
        ii, ih = wilson(ins, inn), wilson(hos, hon)
        print(f"  {label:<12} in-sample {ins}/{inn} {ii.point:.2f} "
              f"[{ii.lo:.2f},{ii.hi:.2f}]   held-out {hos}/{hon} {ih.point:.2f} "
              f"[{ih.lo:.2f},{ih.hi:.2f}]   mean drop {mean(drop(r) for r in group):+.1%}")

    if refs and reps:
        dr, dp = mean(drop(r) for r in refs), mean(drop(r) for r in reps)
        print()
        if dp <= dr + 0.02:
            print("  READS AS: NO OVERFIT. The repairs drop no more than the human")
            print("  references do on the same unseen scenes, so the drop is a")
            print("  property of the second thirty scenes and not of the repairs.")
            print("  The distance metric is a property of the program.")
        elif dp > dr + 0.15:
            print("  READS AS: FITTED TO THE SCENE SET. The repairs drop "
                  f"{dp:.0%} against the references' {dr:.0%}.")
            print("  Every distance in the corpus is then an IN-SAMPLE number and")
            print("  must be reported as one. This does not invalidate the defect")
            print("  taxonomy -- the tags come from what the repair changed, not")
            print("  from how well it scored -- but it does bound the distance claim.")
        else:
            print(f"  READS AS: PARTIAL. Repairs drop {dp:.0%}, references {dr:.0%}.")
            print("  A gap this size needs more programs before it is called either")
            print("  way; extend the sample with --n before interpreting it.")
    print("\n  Per-program drops are noisy at n=30 scenes a side. The pooled")
    print("  comparison against the references is the result; the per-program")
    print("  table is for finding which repairs to look at, not for ranking.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
