"""Collapse failed episodes into distinct programs and write them out for repair.

    python3 scripts/export_failures.py out.json hf:savoji/capx-ab-on [more...]
    python3 scripts/export_failures.py s3.json hf:savoji/capx-ab-on-seeded --tier S3

Emits one record per DISTINCT program, not per episode: greedy decoding means a
cell's 30 episodes share one program, so 926 failures collapse to ~32 records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.repair import group_failures, strip_markers  # noqa: E402
from opencap.store import open_store  # noqa: E402


def main() -> int:
    # --tier is not cosmetic. group_failures buckets by program_id ALONE, so
    # the same code appearing at two tiers collapses into one record whose
    # task/tier is whichever cell was read last. S1 was exported before any
    # other tier existed in the store; now that it holds three, an unfiltered
    # export silently mixes them. Measured on the 2026-10-02 store: 168 S3/S4
    # cells yield 169 programs with NO id spanning two (tier,task) pairs, so
    # filtering is sufficient here -- but the filter is what makes it so.
    argv = list(sys.argv[1:])
    tiers: set[str] | None = None
    if "--tier" in argv:
        i = argv.index("--tier")
        tiers = {t.strip() for t in argv[i + 1].split(",") if t.strip()}
        del argv[i:i + 2]

    out_path = Path(argv[0])
    progs: dict = {}
    for spec in argv[1:]:
        s = open_store(spec)
        cells = s.completed()
        if tiers is not None:
            cells = [c for c in cells if c.tier in tiers]
            print(f"{spec}: {len(cells)} cells at tier(s) {sorted(tiers)}")
        progs.update(group_failures(cells, s.read_cell))
    recs = []
    for p in sorted(progs.values(), key=lambda x: (x.task, -x.n_episodes)):
        recs.append({
            "pid": p.pid, "task": p.task, "tier": p.tier,
            "models": list(p.models), "n_episodes": p.n_episodes,
            "root_causes": list(p.root_causes),
            "median_reward": round(p.median_reward, 4),
            "error": p.error_excerpt,
            "code": strip_markers(p.code),
        })
    out_path.write_text(json.dumps(recs, indent=2))
    print(f"{len(recs)} distinct programs -> {out_path}")
    print(f"covering {sum(r['n_episodes'] for r in recs)} failed episodes")
    by_task: dict[str, int] = {}
    for r in recs:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    print("per task:", by_task)
    by_tier: dict[str, int] = {}
    for r in recs:
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print("per tier:", by_tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
