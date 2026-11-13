"""Operational CLI: plan the sweep, check what is left, resume.

    python -m opencap.cli plan   --wave 1
    python -m opencap.cli status --wave 1 --store hf:USER/capx-scaling-traces
    python -m opencap.cli next   --wave 1 --store local:./runs
"""
from __future__ import annotations

import argparse

from .cells import CORE_TASKS, sweep_order
from .config import STATES_PER_TASK, context_manifest
from .models import BY_KEY, LADDER
from .store import open_store


def _plan(args):
    cells = sweep_order(args.wave)
    per_model: dict[str, int] = {}
    for c in cells:
        per_model[c.model] = per_model.get(c.model, 0) + 1
    print(f"wave {args.wave}: {len(cells)} cells x {STATES_PER_TASK} states "
          f"= {len(cells) * STATES_PER_TASK:,} episodes")
    print(f"{len(per_model)} models, {len(CORE_TASKS)} tasks\n")
    for spine_name in ("A", "B", "C", "D", "anchor"):
        members = [m for m in LADDER if m.spine == spine_name and m.key in per_model]
        if not members:
            continue
        total = sum(per_model[m.key] for m in members)
        print(f"  spine {spine_name:<6} {len(members):>2} models  {total:>4} cells"
              f"   {', '.join(m.key for m in members)}")
    print("\ncontext policy:", context_manifest())


def _status(args):
    planned = sweep_order(args.wave)
    store = open_store(args.store)
    remaining = store.remaining(planned)
    done = len(planned) - len(remaining)
    pct = 100.0 * done / len(planned) if planned else 0.0
    print(f"wave {args.wave}: {done}/{len(planned)} cells complete ({pct:.1f}%)")
    print(f"episodes done ~ {done * STATES_PER_TASK:,}")
    if remaining:
        nxt = remaining[0]
        print(f"next: {nxt.model} {nxt.tier} {nxt.task}")
        print(f"      serve with: scripts/serve_vllm.sh {nxt.model}")


def _next(args):
    store = open_store(args.store)
    remaining = store.remaining(sweep_order(args.wave))
    for c in remaining[: args.n]:
        print(f"{c.model}\t{c.tier}\t{c.task}\t{c.path()}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="opencap")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="print the sweep for a wave")
    pl.add_argument("--wave", type=int, default=1)
    pl.set_defaults(fn=_plan)

    st = sub.add_parser("status", help="how much of the wave is done")
    st.add_argument("--wave", type=int, default=1)
    st.add_argument("--store", required=True)
    st.set_defaults(fn=_status)

    nx = sub.add_parser("next", help="list the next cells to run")
    nx.add_argument("--wave", type=int, default=1)
    nx.add_argument("--store", required=True)
    nx.add_argument("-n", type=int, default=10)
    nx.set_defaults(fn=_next)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
