"""Cells that cannot be collected while :8122 has no backend, listed so a
resume does not spend three hours reproducing the same refusals.

    python3 scripts/molmo_skiplist.py                 # regenerate the list
    python3 scripts/molmo_skiplist.py --show          # print it, change nothing
    python3 scripts/molmo_skiplist.py --clear         # remove it (after :8122 is fixed)

WHY THIS EXISTS
---------------
`capx/integrations/vision/molmo.py` line 20 hardcodes
`SERVICE_URL = "http://127.0.0.1:8122/v1"`. No config declares that port, and
capx/serving/ ships no molmo launcher -- it has graspnet, curobo, owlvit,
pyroki, sam2 and sam3 and nothing else. So the reduced API advertises
`point_prompt_molmo` and binds it to a service that cannot be started from
this checkout.

At temperature 0 a model writes the same program for the same cell every
time, so a cell whose program calls point_prompt_molmo refuses IDENTICALLY on
every attempt. Measured on 2026-09-30: seventeen S4 cells were re-attempted
after a restart and all seventeen refused again, 17 for 17, at roughly twenty
minutes each. That is three hours per resume spent reproducing a known
result, against a mean time between external process-kills of about ten
hours -- so a third of each run was being spent on cells that cannot succeed,
and twice the run was killed before reaching the cells that can.

WHAT IT DOES NOT DO
-------------------
It does not change the corpus. These cells are ALREADY excluded, by
run_cell.py's refusal -- an environment failure must never be scored as a
model defect. The only thing that changes is whether a resume re-attempts
them. Skipping is a scheduling decision, not a data decision.

It is deliberately reversible and deliberately evidence-based:

  * the list is REGENERATED from trial summaries on disk, never hand-written,
    so a cell only appears here because it actually hit :8122;
  * `--clear` removes it, and `OPENCAP_IGNORE_MOLMO_SKIP=1` overrides it for one
    run without deleting anything;
  * once :8122 is resolved, clear the list and the cells collect normally.

DO NOT let this list outlive the outage it describes. A skip list nobody
revisits is how a permanent hole enters a corpus while every run looks clean.
The header written into the JSON says so, with the date.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

SKIPFILE = ROOT / "results" / "molmo_skip.json"
MARKER = "port=8122"


def scan(scratch: Path) -> list[dict]:
    """Every cell on disk with at least one trial that hit :8122."""
    found: list[dict] = []
    for tier_dir in sorted(scratch.glob("tier_*")):
        tier = tier_dir.name[5:]
        for model_dir in sorted(p for p in tier_dir.iterdir() if p.is_dir()):
            for cell in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                name = cell.name                      # model__TIER__task
                parts = name.split("__")
                if len(parts) != 3:
                    continue
                hits = 0
                for f in cell.glob("trial_*/summary.txt"):
                    try:
                        if MARKER in f.read_text(errors="ignore"):
                            hits += 1
                    except OSError:
                        continue
                if hits:
                    found.append({"model": parts[0], "tier": tier,
                                  "task": parts[2], "trials_hit": hits})
    return found


def load() -> set[tuple[str, str, str]]:
    """The skip set, for run_cell.py. Empty when the file is absent."""
    if not SKIPFILE.exists():
        return set()
    try:
        doc = json.loads(SKIPFILE.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {(c["model"], c["tier"], c["task"]) for c in doc.get("cells", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/data/capx-out")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()

    if a.clear:
        if SKIPFILE.exists():
            SKIPFILE.unlink()
            print(f"[molmo] removed {SKIPFILE.relative_to(ROOT)}")
            print("[molmo] those cells will be attempted again on the next run.")
        else:
            print("[molmo] no skip list to remove")
        return 0

    if a.show:
        cells = sorted(load())
        if not cells:
            print("[molmo] no skip list in place")
            return 0
        print(f"[molmo] {len(cells)} cell(s) skipped while :8122 has no backend:")
        for m, t, k in cells:
            print(f"    {t}  {m:<20} {k}")
        return 0

    cells = scan(Path(a.scratch))
    if not cells:
        print("[molmo] no cells on disk hit :8122 -- nothing to skip")
        return 0

    doc = {
        "why": ("Cells whose generated program calls point_prompt_molmo. "
                "capx/integrations/vision/molmo.py hardcodes 127.0.0.1:8122 "
                "and capx/serving ships no launcher for it, so these refuse "
                "identically on every attempt. They are ALREADY excluded from "
                "the corpus by run_cell.py's refusal; this only stops a resume "
                "re-running them."),
        "remove_when": ("port 8122 is resolved -- then run "
                        "`python3 scripts/molmo_skiplist.py --clear` and the "
                        "cells collect normally. Do NOT let this outlive the "
                        "outage; a stale skip list is a permanent hole that "
                        "looks like a clean run."),
        "override_once": "OPENCAP_IGNORE_MOLMO_SKIP=1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "cells": cells,
    }
    SKIPFILE.parent.mkdir(parents=True, exist_ok=True)
    SKIPFILE.write_text(json.dumps(doc, indent=2) + "\n")

    by_tier: dict[str, int] = {}
    for c in cells:
        by_tier[c["tier"]] = by_tier.get(c["tier"], 0) + 1
    print(f"[molmo] wrote {SKIPFILE.relative_to(ROOT)}: {len(cells)} cell(s) "
          f"({', '.join(f'{t} {n}' for t, n in sorted(by_tier.items()))})")
    print("[molmo] these are skipped on resume, NOT removed from the corpus --")
    print("[molmo] they were already refused. Clear the list once :8122 works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
