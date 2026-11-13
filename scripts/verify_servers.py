"""Refuse to run a cell whose tier declares API servers that are not listening.

WHY THIS EXISTS
---------------
The whole S4 sweep -- 7 cells, ~2h of A100 -- was run without its perception
stack and had to be thrown away. The tier's config declares three api_servers;
only PyRoKi came up. SAM3 and Contact-GraspNet were never installed, so:

    the prompt documents  obs["robot0_robotview"]["images"]["depth"]
    the environment       raises KeyError: 'depth'

Models followed the documentation correctly and scored 0/30. That read as an
abstraction cliff -- A(4B) = +0.622, p < 0.0001 -- and was two steps from being
published as one.

Nothing in the harness noticed, because every existing guard checks the OUTPUT
of a run (episode counts, cross-checks, stderr coverage). This checks a
PRECONDITION: the environment a cell needs must exist before the cell runs.

    python3 scripts/verify_servers.py <task> <tier> [--root /content/cap-x]

Exit 0 = every declared server is listening. Exit 1 = do not run this cell.
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.capx_bridge import config_path  # noqa: E402


def declared_ports(cfg_text: str) -> list[tuple[str, int]]:
    """(server name, port) for every api_server the config declares.

    Parsed with regex rather than a YAML load so this stays dependency-free and
    usable from a bare pod shell during triage.
    """
    out: list[tuple[str, int]] = []
    block = cfg_text.split("api_servers:", 1)
    if len(block) < 2:
        return out
    name = None
    for line in block[1].splitlines():
        if re.match(r"^\S", line) and not line.startswith(" "):
            break                                   # left the api_servers block
        m = re.search(r"_target_:\s*\S*launch_(\w+?)_server", line)
        if m:
            name = m.group(1)
        p = re.search(r"^\s*port:\s*(\d+)", line)
        if p and name:
            out.append((name, int(p.group(1))))
    return out


def listening(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("tier")
    ap.add_argument("--root", default="/content/cap-x")
    a = ap.parse_args()

    cfg = Path(a.root) / config_path(a.task, a.tier)
    if not cfg.exists():
        print(f"MISSING CONFIG {cfg}")
        return 1
    ports = declared_ports(cfg.read_text())
    if not ports:
        print(f"{a.task}/{a.tier}: no api_servers declared -- nothing to check")
        return 0

    bad = []
    for name, port in ports:
        ok = listening(port)
        print(f"  {name:<20}{port:>6}  {'LISTENING' if ok else 'NOT LISTENING'}")
        if not ok:
            bad.append((name, port))
    if bad:
        print(f"\nREFUSING {a.task}/{a.tier}: "
              + ", ".join(f"{n} ({p})" for n, p in bad) + " not reachable.")
        print("The prompt will document APIs the environment cannot honour, and "
              "models that follow it will score 0 for reasons that are ours.")
        return 1
    print(f"\n{a.task}/{a.tier}: all {len(ports)} declared servers listening")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
