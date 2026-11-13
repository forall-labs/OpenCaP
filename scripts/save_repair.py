"""Merge one repair into results/{repairs,minimal}/<task>.json without clobbering.

    python3 scripts/save_repair.py --task cube_lifting --pid 620009e0637c \
            --code-file /tmp/fix.py [--minimal] [--note "..."] [--force]

WHY THIS EXISTS
---------------
Those files are dicts of {program_id: record} holding every repair for a task --
10 to 11 each at the time of writing. Writing one out with json.dump() from a
fresh dict silently deletes the rest, and nothing downstream would notice: the
distance report would simply describe a smaller corpus, and the missing entries
look exactly like programs nobody got to yet.

So: read, merge, write. Refuse to replace an existing program id unless --force
is passed, because a second repair for a pid that already has one is far more
often a mistake than an intention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def save(task: str, pid: str, code: str, minimal: bool = False,
         note: str = "", force: bool = False, root: Path = ROOT,
         level: str = "L3_edit", tier: str = "S1") -> dict:
    # TIER IS NOT OPTIONAL METADATA. These files are keyed by program id and
    # hold every repair for a task, and until 2026-10-03 every record in them
    # was S1 because S1 was the only tier with a repair corpus. The moment an
    # S3 repair lands in the same file, a downstream tag or distance report
    # mixes two tiers and nothing notices -- the same defect export_failures.py
    # had, where group_failures buckets by program_id alone.
    d = root / "results" / ("minimal" if minimal else "repairs")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{task}.json"
    existing: dict = json.loads(path.read_text()) if path.exists() else {}
    if pid in existing and not force:
        raise SystemExit(
            f"[save_repair] {pid} already has an entry in {path}.\n"
            f"[save_repair] Refusing to replace it. Pass --force if you really\n"
            f"[save_repair] mean to, or use a different pid if this is a new program."
        )
    # `level` matches the schema of the hand-built records already in these
    # files (L3_edit = a code edit, the only kind this helper saves), and
    # verify_repairs.py reads it.
    record = {"code": code, "level": level, "tier": tier}
    if note:
        record["note"] = note
    existing[pid] = record
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return {"path": str(path), "pid": pid, "entries": len(existing)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--code-file", required=True,
                    help="file holding the repaired program")
    ap.add_argument("--tier", default="S1",
                    help="which tier this repair is written against. The "
                         "existing corpus is all S1; an S3/S4 repair MUST say "
                         "so or downstream reports mix tiers silently.")
    ap.add_argument("--minimal", action="store_true",
                    help="write to results/minimal/ instead of results/repairs/")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--level", default="L3_edit")
    a = ap.parse_args()
    code = Path(a.code_file).read_text()
    r = save(a.task, a.pid, code, a.minimal, a.note, a.force,
             level=a.level, tier=a.tier)
    print(f"[save_repair] wrote {r['pid']} -> {r['path']} ({r['entries']} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
