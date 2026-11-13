"""Record a verification result and swap in the next repair attempt.

    python3 scripts/apply_iteration.py <pid> <result> <hypothesis> <outcome> <new_code_file>

Keeps results/iterations/<pid>.json as an append-only log of
    attempt -> code -> what the environment said
so the reported distance is the distance of the version that PASSED, and the
path to it is auditable. Hand-editing each iteration was becoming the most
error-prone step in the loop.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencap.repair import normalised_edit_distance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def repairs_file_for(pid: str) -> Path:
    for f in glob.glob(str(ROOT / "results/repairs/*.json")):
        if pid in json.loads(Path(f).read_text()):
            return Path(f)
    raise SystemExit(f"pid {pid} not found in results/repairs/")


def main() -> int:
    pid, result, hypothesis, outcome, code_file = sys.argv[1:6]
    rf = repairs_file_for(pid)
    reps = json.loads(rf.read_text())
    prev = reps[pid]["code"]

    log_path = ROOT / "results/iterations" / f"{pid}.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {pid: []}
    log.setdefault(pid, []).append({
        "iter": len(log[pid]) + 1, "code": prev,
        "result": result, "hypothesis": hypothesis, "outcome": outcome,
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))

    new_code = Path(code_file).read_text().rstrip("\n")
    reps[pid]["code"] = new_code
    rf.write_text(json.dumps(reps, indent=2))

    fails = {x["pid"]: x for x in
             json.loads((ROOT / "results/failures.json").read_text())}
    orig = fails[pid]["code"]
    print(f"{pid}: attempt {len(log[pid])} recorded ({result})")
    print(f"  distance {normalised_edit_distance(orig, prev):.3f} "
          f"-> {normalised_edit_distance(orig, new_code):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
