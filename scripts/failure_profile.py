"""Failure root-cause profile by model scale.

Answers the question success rate cannot: as models grow, WHICH failure mode
disappears? An execution failure and a semantic failure have different
remedies, so a scaling curve over success conflates two trends that should be
read separately.

    python3 scripts/failure_profile.py hf:savoji/capx-ab-off [hf:...on]
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


def profile(spec: str):
    """Root-cause counts per model over a BALANCED panel.

    Comparing models across whatever cells each happens to have is not a
    scaling comparison -- a model carrying extra hard-tier cells looks worse
    for reasons that have nothing to do with its size. Only (tier, task) pairs
    present for every model are counted; the rest are reported as dropped.
    """
    store = open_store(spec)
    cells = list(store.completed())
    models = {c.model for c in cells}
    per_model = {m: {(c.tier, c.task) for c in cells if c.model == m} for m in models}
    shared = set.intersection(*per_model.values()) if per_model else set()
    dropped = sorted({(c.tier, c.task) for c in cells} - shared)

    by_model: dict[str, Counter] = defaultdict(Counter)
    fam: dict[str, Counter] = defaultdict(Counter)
    n_eps = n_stderr = 0
    for cell in cells:
        if (cell.tier, cell.task) not in shared:
            continue
        for e in store.read_cell(cell):
            n_eps += 1
            if any((t.get("summary_tail") or "") for t in (e.get("turns") or [])):
                n_stderr += 1
            c = classify_root_cause(e)
            by_model[cell.model][c] += 1
            fam[cell.model][family(c)] += 1
    return by_model, fam, shared, dropped, n_eps, n_stderr


#: Below this share of episodes carrying stderr, every failure lands in
#: `unknown_execution` and the table looks like a trend while being a data gap.
#: Traces collected before the `turns` fix have 0% coverage -- refuse them.
MIN_STDERR_COVERAGE = 0.50


def show(title: str, by_model, fam, shared, dropped, n_eps, n_stderr):
    print(f"\n===== {title} =====")
    cov = n_stderr / n_eps if n_eps else 0.0
    if cov < MIN_STDERR_COVERAGE:
        print(f"REFUSING to profile: only {n_stderr}/{n_eps} = {cov:.1%} of "
              f"episodes carry stderr.")
        print("Every failure would be labelled `unknown_execution`, which is "
              "missing data, not a root cause.")
        print("Re-collect with a harness that records turns before reading "
              "this corpus for failure modes.")
        return
    print(f"stderr coverage: {n_stderr}/{n_eps} = {cov:.1%}")
    print(f"balanced panel: {len(shared)} (tier, task) cells per model -- "
          + ", ".join(f"{t}/{k}" for t, k in sorted(shared)))
    if dropped:
        print("dropped (not present for every model): "
              + ", ".join(f"{t}/{k}" for t, k in dropped))
    models = sorted(by_model, key=lambda m: BY_KEY[m].params_b if m in BY_KEY else 0)
    causes = sorted({c for v in by_model.values() for c in v})
    w = max((len(c) for c in causes), default=10) + 2
    print(f"{'cause':<{w}}" + "".join(f"{m.replace('qwen3-',''):>10}" for m in models))
    print("-" * (w + 10 * len(models)))
    for c in causes:
        row = "".join(f"{by_model[m].get(c, 0):>10}" for m in models)
        print(f"{c:<{w}}{row}")
    print("-" * (w + 10 * len(models)))
    for f in ("success", "execution", "semantic"):
        row = ""
        for m in models:
            tot = sum(fam[m].values())
            row += f"{(fam[m].get(f, 0) / tot if tot else 0):>9.1%} "
        print(f"{f:<{w}}{row}")


def main() -> int:
    show("arm: " + sys.argv[1], *profile(sys.argv[1]))
    if len(sys.argv) > 2:
        show("arm: " + sys.argv[2], *profile(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
