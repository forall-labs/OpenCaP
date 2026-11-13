"""Root cause by model scale, in both scaffold conditions.

The question success rate cannot answer: as models grow, WHICH failure mode
disappears -- and how much of the small-model gap is the scaffold rather than
the model?

Reading the two arms side by side separates them. `off` is the scaffold with a
known defect (no numpy preimport); `on` is the same scaffold repaired. A cause
that shrinks from off to on at fixed scale is harness. A cause that shrinks
with scale at fixed arm is capability.

    python3 scripts/scale_report.py hf:savoji/capx-ab-off hf:savoji/capx-ab-on
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.failures import classify_root_cause, family  # noqa: E402
from opencap.models import BY_KEY  # noqa: E402
from opencap.store import open_store  # noqa: E402

MIN_STDERR_COVERAGE = 0.50


def load(spec):
    """Per-model root-cause counts over cells shared by BOTH arms."""
    store = open_store(spec)
    out, cells = {}, list(store.completed())
    for c in cells:
        out.setdefault(c.model, {})[(c.tier, c.task)] = list(store.read_cell(c))
    return out


def main() -> int:
    off, on = load(sys.argv[1]), load(sys.argv[2])
    models = sorted(set(off) & set(on),
                    key=lambda m: BY_KEY[m].params_b if m in BY_KEY else 0)
    if not models:
        print("no model appears in both arms yet")
        return 1
    # Balanced across models AND arms: anything else compares different work.
    panel = set.intersection(*[set(off[m]) for m in models],
                             *[set(on[m]) for m in models])
    if not panel:
        print("no (tier, task) cell is present for every model in both arms")
        return 1

    n_eps = n_err = 0
    for src in (off, on):
        for m in models:
            for k in panel:
                for e in src[m][k]:
                    n_eps += 1
                    if any((t.get("summary_tail") or "")
                           for t in (e.get("turns") or [])):
                        n_err += 1
    if n_eps and n_err / n_eps < MIN_STDERR_COVERAGE:
        print(f"REFUSING: stderr coverage {n_err}/{n_eps} = {n_err/n_eps:.1%}; "
              "failures would be unlabelled, which is missing data.")
        return 1

    print(f"panel: {len(panel)} cells x {len(models)} models x 2 arms "
          f"= {n_eps} episodes; stderr coverage {n_err/n_eps:.1%}")
    print("cells: " + ", ".join(f"{t}/{k}" for t, k in sorted(panel)))

    print(f"\n{'':<12}" + "".join(f"{m.replace('qwen3-',''):>18}" for m in models))
    print(f"{'':<12}" + "".join(f"{'off':>9}{'on':>9}" for _ in models))
    print("-" * (12 + 18 * len(models)))
    rows = defaultdict(dict)
    for m in models:
        for arm, src in (("off", off), ("on", on)):
            fam = Counter()
            for k in panel:
                for e in src[m][k]:
                    fam[family(classify_root_cause(e))] += 1
            tot = sum(fam.values()) or 1
            for f in ("success", "execution", "semantic"):
                rows[f][(m, arm)] = fam[f] / tot
    for f in ("success", "execution", "semantic"):
        line = "".join(f"{rows[f][(m,a)]:>8.1%} " for m in models
                       for a in ("off", "on"))
        print(f"{f:<12}{line}")

    print("\nwhat the two arms separate, per scale:")
    for m in models:
        d_exec = rows["execution"][(m, "off")] - rows["execution"][(m, "on")]
        d_succ = rows["success"][(m, "on")] - rows["success"][(m, "off")]
        resid = rows["execution"][(m, "on")]
        print(f"  {m:<12} scaffold-removable execution: {d_exec:>6.1%}   "
              f"success gained: {d_succ:>+6.1%}   execution left after repair: "
              f"{resid:>5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
