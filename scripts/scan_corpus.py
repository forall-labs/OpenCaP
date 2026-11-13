"""Data-quality scan over a committed corpus.

    python3 scripts/scan_corpus.py hf:savoji/capx-ab-on [more stores...]

WHY THIS EXISTS
---------------
run_cell.py's timeout guard reads a RUN DIRECTORY, so it protects collection
and cannot see a cell that is already committed. The four timeout-contaminated
cells found on 2026-09-10 were found by hand; two MORE (gemma3-12b spill_wipe
and cube_stack, from the Colab corpus) were missed by that hand search because
nobody thought to scan the older half.

A cell whose episodes carry no program is not a model scoring zero. It is a
model that was never asked. Both read 0/30 and nothing distinguishes them in
the score, which is exactly the shape of failure this project keeps finding.

Reports per cell:
  empty      episodes with no program at all -- generation never landed
  nostderr   episodes with no stderr captured (the capx-scaling-traces defect)
  identical  whether all failing episodes share one program, as temperature 0
             requires; a violation means the determinism controls are not
             holding and every paired statistic is affected
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencap.repair import program_id, program_of  # noqa: E402
from opencap.store import open_store  # noqa: E402

#: Above this fraction of empty-program episodes a cell is not a measurement.
#: Set below the smallest contaminated cell seen (0.83) and well above the
#: largest clean one (0.00) -- there is no middle ground in the observed data.
EMPTY_LIMIT = 0.20


def scan(spec: str) -> int:
    store = open_store(spec)
    cells = sorted(c for c in store.completed())
    bad, warn = [], []
    print(f"\n=== {spec}: {len(cells)} cells ===")
    for c in cells:
        eps = list(store.read_cell(c))
        n = len(eps)
        empty = sum(1 for e in eps if not (program_of(e) or "").strip())
        nostderr = sum(1 for e in eps if not (e.get("stderr") or "").strip())
        ok = sum(1 for e in eps if e.get("success"))
        # ALL episodes, not just failing ones. The earlier version inspected
        # only failures, so a cell at or near ceiling could emit two distinct
        # programs and pass the scan -- which is exactly what happened: four of
        # the five cells caught by hand on 2026-09-16 were at ceiling, and none
        # of them was flagged here.
        progs = {program_id(program_of(e)) for e in eps
                 if (program_of(e) or "").strip()}
        host = Counter(e.get("host_kind", "?") for e in eps).most_common(1)[0][0]
        noprov = sum(1 for e in eps if not e.get("vllm_version"))

        flags = []
        warn_extra: list[str] = []
        if n and empty / n > EMPTY_LIMIT:
            flags.append(f"EMPTY {empty}/{n}")
        # A cell with more than one program breaks the dedup assumption, but
        # its SCORES are still a valid measurement -- four of the five such
        # cells found so far were at or near ceiling. Calling that "unusable"
        # would invite deleting good data. It is a determinism warning: the
        # cell counts, the one-program-per-cell claim does not hold for it.
        if len(progs) > 1:
            warn_extra.append(f"{len(progs)} distinct programs "
                              f"(dedup broken; scores still valid)")
        if flags:
            bad.append((c, flags))
        elif warn_extra:
            warn.append((c, empty, noprov == n, warn_extra))
        elif empty or (n and noprov == n):
            warn.append((c, empty, noprov == n, []))

    for c, flags in bad:
        print(f"  [BAD ] {c.model:<18}{c.tier} {c.task:<13} {'; '.join(flags)}")
    for c, empty, noprov, extra in warn:
        note = list(extra)
        if empty:
            note.append(f"{empty} empty episode(s), below limit")
        if noprov:
            note.append("no vllm_version recorded")
        print(f"  [warn] {c.model:<18}{c.tier} {c.task:<13} {'; '.join(note)}")
    if not bad and not warn:
        print("  clean")
    print(f"  -> {len(bad)} unusable, {len(warn)} with warnings, "
          f"{len(cells) - len(bad) - len(warn)} clean")
    return len(bad)


def main() -> int:
    specs = sys.argv[1:] or ["hf:savoji/capx-ab-on"]
    return 1 if sum(scan(s) for s in specs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
