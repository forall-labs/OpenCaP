"""Which COMMITTED cells contain trials where a perception service did not answer?

    python3 scripts/service_loss_audit.py                    # every tier in scratch
    python3 scripts/service_loss_audit.py --tier S3
    python3 scripts/service_loss_audit.py --tier S3 --quiet   # one line per cell

WHY THIS EXISTS
---------------
run_cell.py refuses a cell when a perception service did not answer, so in
principle no such cell reaches the corpus. In practice the guard has been one
error string short twice:

  2026-09-28  it fired only when ALL thirty trials failed, so a partial loss
              committed silently. Fixed to refuse on any.
  2026-09-29  it matched only "Connection refused" and "Max retries exceeded".
              A LIVE service returning 500 matches neither. qwen3-1.7b S3
              cube_lifting committed with 4 of 30 trials dead on
              "Failed to communicate with GraspNet service ... 500", whose
              cause the server log gives as CUDA out of memory with a
              neighbour holding 77 GB of the card. The next request returned
              200 and 198 grasps: the service was alive and simply could not
              allocate.

Both times the cells already collected under the old guard stayed in the
corpus, because a guard only protects what comes after it. This script is the
retrospective half: it scans trial summaries on disk and names the cells whose
numbers were produced with a service partly missing, so they can be recollected
or held out rather than quietly believed.

A cell listed here is NOT necessarily wrong. It is unattributable, which for
the failure MIX -- the deliverable of the tier grid -- is the same problem.
An environment failure tagged as a model defect is the artifact this project
has retracted findings over.

SHARED-CARD NOTE. The OOM variant is a function of what else is on the GPU, so
it recurs and it is not reproducible on demand. Quote the audit with the run,
not with the tier.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.analysis import wilson  # noqa: E402

#: Kept in step with run_cell.py's SERVICE_LOST. A string here and not there
#: means the audit finds contamination the guard will let through again.
SERVICE_LOST = (
    "Connection refused",
    "Max retries exceeded",
    "Failed to communicate with",
)

SERVICE_RE = re.compile(r"Failed to communicate with (\w+) service at (\S+?):")
PORT_RE = re.compile(r"port=(\d+)|127\.0\.0\.1:(\d{4})")


def guard_is_in_step() -> list[str]:
    """Strings this audit looks for that run_cell.py does NOT refuse on."""
    src = (ROOT / "scripts" / "run_cell.py").read_text(errors="ignore")
    return [s for s in SERVICE_LOST if s not in src]


def cells(scratch: Path, tier: str):
    root = scratch / f"tier_{tier}"
    if not root.exists():
        return
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cell in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if (cell / "aaa_done_flag").exists():
                yield cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/data/capx-out")
    ap.add_argument("--tier", action="append", dest="tiers")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    scratch = Path(a.scratch)
    tiers = a.tiers or [d.name[5:] for d in sorted(scratch.glob("tier_*"))]

    drift = guard_is_in_step()
    if drift:
        print(f"[svcloss] WARNING: run_cell.py does not refuse on {drift}.")
        print("[svcloss] The guard and this audit have drifted; new cells can")
        print("[svcloss] commit with the contamination this script reports.")

    for tier in tiers:
        found = list(cells(scratch, tier))
        if not found:
            continue
        print(f"\n===== {tier} -- {len(found)} committed cell(s) on disk =====")

        dirty: list[tuple[str, int, int, str]] = []
        trials_all = trials_bad = 0
        by_service: dict[str, int] = defaultdict(int)
        by_port: dict[str, int] = defaultdict(int)

        for cell in found:
            n = bad = 0
            svcs: set[str] = set()
            for trial in sorted(cell.glob("trial_*")):
                f = trial / "summary.txt"
                if not f.exists():
                    continue
                n += 1
                txt = f.read_text(errors="ignore")
                if not any(s in txt for s in SERVICE_LOST):
                    continue
                bad += 1
                for m in SERVICE_RE.finditer(txt):
                    by_service[m.group(1)] += 1
                    svcs.add(m.group(1))
                for m in PORT_RE.finditer(txt):
                    p = m.group(1) or m.group(2)
                    by_port[p] += 1
                    svcs.add(f":{p}")
            trials_all += n
            trials_bad += bad
            if bad:
                dirty.append((cell.name, bad, n, ", ".join(sorted(svcs)) or "?"))

        if not trials_all:
            print("  no trial summaries")
            continue

        ci = wilson(trials_bad, trials_all)
        print(f"  cells with any service loss  {len(dirty)} of {len(found)}")
        print(f"  trials affected              {trials_bad} of {trials_all}"
              f"  = {100 * ci.point:.1f}%  [{100 * ci.lo:.1f}, {100 * ci.hi:.1f}]")

        # Total loss vs partial loss is the distinction that matters. A cell
        # where every trial lost the service is obvious wreckage and was very
        # likely refused. A cell where SOME trials lost it looks like a real
        # result and is the one that gets believed.
        partial = [d for d in dirty if d[1] < d[2]]
        total = [d for d in dirty if d[1] == d[2]]
        print(f"  ... of which TOTAL loss      {len(total)}  (obvious wreckage)")
        print(f"  ... of which PARTIAL loss    {len(partial)}  "
              f"<-- these look like real results")

        if by_service:
            print("  by service: " + ", ".join(
                f"{k} {v}" for k, v in sorted(by_service.items(), key=lambda kv: -kv[1])))
        if by_port:
            print("  by port:    " + ", ".join(
                f":{k} {v}" for k, v in sorted(by_port.items(), key=lambda kv: -kv[1])))

        if not a.quiet and partial:
            print("\n  PARTIAL-LOSS CELLS -- recollect or hold out of the mix:")
            for name, bad, n, svcs in sorted(partial, key=lambda d: -d[1]):
                print(f"    {name:<44} {bad:>3}/{n:<3} {svcs}")
        if not a.quiet and total:
            print("\n  TOTAL-LOSS CELLS (check these were refused, not committed):")
            for name, bad, n, svcs in sorted(total):
                print(f"    {name:<44} {bad:>3}/{n:<3} {svcs}")

    print("\n[svcloss] A partial-loss cell is not necessarily wrong -- it is")
    print("[svcloss] unattributable, which for a failure MIX is the same thing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
