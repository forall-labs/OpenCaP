"""Migrate reusable cells from a retired store into its single-host successor.

    python3 scripts/copy_clean_cells.py hf:savoji/capx-promptfix hf:savoji/capx-promptfix-h200

Not everything collected before the cutover is worthless. A cell that already
ran on H200 and passes the data-quality scan is a perfectly good cell, and
re-running it would burn GPU to reproduce a result we have.

What is NOT copied, and why:
  * cells from any other host -- artifact #12: the generated program itself
    changes across GPU architectures in half of all cells;
  * cells whose episodes carry no program (the 200s-timeout artifact), because
    a model that was never asked is indistinguishable in the score from a model
    that failed.

Both are re-run instead. The point of the cutover is that every cell in the new
store was collected under known-good conditions on one machine; copying a bad
cell forward would defeat it while making the store look complete.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencap.repair import program_of  # noqa: E402
from opencap.store import open_store  # noqa: E402

EMPTY_LIMIT = 0.20
WANT_HOST = "h200"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = open_store(sys.argv[1]), open_store(sys.argv[2])
    dry = "--dry-run" in sys.argv
    have = set(dst.completed())
    copied = skipped = present = 0
    for c in sorted(src.completed()):
        eps = list(src.read_cell(c))
        host = Counter(e.get("host_kind", "?") for e in eps).most_common(1)[0][0]
        empty = sum(1 for e in eps if not (program_of(e) or "").strip()) / max(len(eps), 1)
        if c in have:
            present += 1
            continue
        if host != WANT_HOST:
            print(f"  re-run  {c.model:<18}{c.task:<13} host={host}")
            skipped += 1
        elif empty > EMPTY_LIMIT:
            print(f"  re-run  {c.model:<18}{c.task:<13} {empty:.0%} empty programs")
            skipped += 1
        else:
            if not dry:
                dst.write_cell(c, eps, verified=True)
            print(f"  copy    {c.model:<18}{c.task:<13} {len(eps)} episodes")
            copied += 1
    print(f"\n{copied} copied, {skipped} left to re-run, {present} already present"
          + (" (dry run, nothing written)" if dry else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
