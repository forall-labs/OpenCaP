"""Delete ONE cell's shard from a store, so a re-run can replace it.

    python3 scripts/delete_cell.py hf:savoji/capx-m1-seeded qwen3-1.7b M1 spill_wipe
    python3 scripts/delete_cell.py ... --yes        # actually do it

WHY THIS EXISTS, AND WHY IT IS DELIBERATELY AWKWARD
---------------------------------------------------
Cells are atomic and resume skips anything already present, which is what
makes an interrupted 84-cell sweep cheap to restart. The cost is that a cell
collected under the WRONG CONDITIONS is sticky: resume will keep it forever,
and nothing downstream can tell it apart from a good one.

That happened on 2026-10-15. A3 ran with the perception stack down because
run_m1_arm.sh called M1 the privileged API. Seven of its eight cells died
'Connection refused' x30 and were correctly refused. The eighth, qwen3-1.7b
M1 spill_wipe, committed 0/30 -- a FALSE ZERO, indistinguishable in the store
from a model that genuinely solved nothing. Every M1 number computed over that
store is wrong by one cell until it is gone.

Deleting from a published corpus is not a thing to do casually, so:

  * it prints what it will delete and exits WITHOUT doing it unless --yes;
  * it refuses frozen stores (the same guard as writing);
  * it refuses a store that does not exist rather than creating one;
  * it reports what the shard CONTAINS first -- episode count, success count,
    and any error strings -- because the whole point is that you cannot see
    from the outside whether a cell is bad, so you should look before you cut.

It does not "repair" anything. After this, re-run the arm; the cell is simply
missing and resume collects it.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.cells import Cell  # noqa: E402
from opencap.store import (StoreUnreachable, _check_not_frozen,  # noqa: E402
                         open_store, store_exists)


def describe(eps: list[dict]) -> None:
    """What is in this shard? Look before cutting."""
    ok = sum(1 for e in eps if e.get("success"))
    print(f"  episodes      {len(eps)}")
    print(f"  successes     {ok}")
    errs = Counter((e.get("err") or "")[:70] for e in eps if e.get("err"))
    if errs:
        print("  error strings (truncated, by count):")
        for msg, n in errs.most_common(5):
            print(f"    x{n:<4} {msg!r}")
    else:
        print("  error strings none -- every episode ran to completion.")
        print("  NOTE: that is exactly how a cell collected against a DEAD")
        print("  perception stack can look, if the program never reached a")
        print("  perception call before failing for its own reasons.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", help="hf:user/repo or local:/path")
    ap.add_argument("model")
    ap.add_argument("tier")
    ap.add_argument("task")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete; without it this is a dry run")
    ap.add_argument("--why", default="",
                    help="recorded in the commit message")
    a = ap.parse_args()

    cell = Cell(a.model, a.tier, a.task)      # validates the triple
    print(f"store {a.store}")
    print(f"cell  {cell.model} / {cell.tier} / {cell.task}")
    print(f"path  {cell.path()}")

    try:
        if not store_exists(a.store):
            print("\nthat store does not exist -- nothing to delete, and this "
                  "will not create it.")
            return 1
    except StoreUnreachable as e:
        print(f"\nCANNOT REACH THE STORE. Refusing to guess.\n\n{e}")
        return 2

    if a.store.startswith("hf:"):
        # Deleting a shard IS a write to the repo, so the frozen-corpus guard
        # applies with more force here than it does to adding one.
        _check_not_frozen(a.store.partition(":")[2])

    s = open_store(a.store)
    if cell not in s.completed():
        print("\nthat cell is NOT in the store. Nothing to delete; a re-run "
              "will collect it.")
        return 1

    print("\nwhat the shard holds now:")
    describe(list(s.read_cell(cell)))

    if not a.yes:
        print("\nDRY RUN. Nothing was deleted. Re-run with --yes to delete, "
              "then re-run the arm to collect the cell again.")
        return 0

    kind, _, rest = a.store.partition(":")
    msg = f"delete cell {cell.model}/{cell.tier}/{cell.task}"
    if a.why:
        msg += f": {a.why}"
    if kind == "local":
        (Path(rest) / cell.path()).unlink()
    else:
        s.api.delete_file(path_in_repo=cell.path(), repo_id=rest,
                          repo_type="dataset", commit_message=msg)
    # LocalStore reads the tree each time; HFStore caches the file list.
    if hasattr(s, "refresh"):
        s.refresh()
    still = cell in s.completed()
    print(f"\ndeleted. {msg}")
    print(f"store still reports the cell: {still}  (expected False)")
    return 1 if still else 0


if __name__ == "__main__":
    raise SystemExit(main())
