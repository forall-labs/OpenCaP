"""Raise CaP-X's hardcoded HTTP read timeout so a long generation can finish.

    python3 scripts/patch_llm_timeout.py "$CAPX_DIR"
    OPENCAP_LLM_READ_TIMEOUT=900 python3 scripts/patch_llm_timeout.py "$CAPX_DIR"

THE PROBLEM
-----------
`capx/llm/client.py` posts to the engine with a literal `timeout=200` at three
call sites. That is a read timeout, so it fires when the ENGINE is slow, not
when the model is wrong.

Measured 2026-09-27: the qwen3.5-27B sampled draw 3 of nut_assembly runs to the
2,048-token cap on 93% of its requests. At OPENCAP_WORKERS=2 behind a batch-1
shim the second worker queues behind a ~110-second generation, so its request
waits past 200s and dies. 29 of 30 trials came back
"Read timed out. (read timeout=200)" and run_cell.py refused the cell -- twice,
once on a card holding another account's job and once on an empty card with the
box load at 16. It is not contention; it is this draw's token budget meeting
this timeout.

WHY RAISING IT IS NOT A CHANGED CONDITION
-----------------------------------------
A timeout is a failure threshold, not a per-cell parameter. Every other cell in
the corpus answers well inside 200s, so their behaviour is byte-identical
whether the limit is 200 or 600 -- the threshold is simply never reached. The
alternative fix, OPENCAP_WORKERS=1, IS a changed condition: the leak A/B measured
worker count moving the nut_assembly reference from 16/30 at four workers to
14/30 at one, so a cell collected that way would not pool with the other 103.

Authorised in STATUS 2026-09-28. The default stays below CaP-X's own 1000s
per-trial timeout, so a stuck request still fails the trial rather than hanging
the run.

WHY A SCRIPT AND NOT AN EDIT
----------------------------
Same reason patch_seeding.py exists: cap-x is a dependency, a fresh checkout
comes back unpatched, and a patch nobody can re-apply is a patch that silently
stops being true. Idempotent, and it reports PATTERN NOT FOUND rather than
returning success when it changed nothing -- a check that cannot fail is not a
check.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

#: The literal in an unpatched checkout, at three call sites.
STOCK_TIMEOUT = 200

#: What we raise it to. Deliberately below CaP-X's 1000s per-trial timeout:
#: a request that will never answer should still fail its trial.
DEFAULT_TIMEOUT = int(os.environ.get("OPENCAP_LLM_READ_TIMEOUT", "600"))

TARGET = "capx/llm/client.py"

MARKER = "# opencap: read timeout raised"

#: Matches the stock literal AND an already-raised one, so the script is
#: idempotent and can also re-raise to a new value.
#:
#: The three sites are not formatted alike: two end the argument list
#: (`timeout=200\n    )`) and the streaming one at line 340 is followed by
#: another kwarg (`timeout=200,\n        stream=True,`). The first version of
#: this pattern required the closing paren and silently matched two of three
#: -- caught only because EXPECTED_SITES disagreed with the count. Anchor on
#: the keyword and its delimiter instead, and require a word boundary so
#: `read_timeout=` or `connect_timeout=` cannot be caught by accident.
PATTERN = re.compile(r"(?<![\w_])timeout=(\d+)(?=\s*[,)\n])")

#: How many sites the file is expected to carry. If cap-x grows a fourth, the
#: count check below says so instead of patching three and reporting success.
EXPECTED_SITES = 3


def patch_file(path: Path, timeout: int) -> tuple[str, int]:
    """-> (status, n_sites_changed). Idempotent."""
    text = path.read_text()
    hits = PATTERN.findall(text)
    if not hits:
        return "PATTERN NOT FOUND -- inspect manually", 0

    # PATTERN has one group, so findall gives strings, not tuples. The
    # first version read int(h[0]) and parsed "6" out of "600", so the
    # idempotency branch never fired and every run rewrote the file.
    already = [h for h in hits if int(h) == timeout]
    if len(already) == len(hits) and MARKER in text:
        return f"already at {timeout}s ({len(hits)} sites)", 0

    if len(hits) != EXPECTED_SITES:
        # Not fatal -- patch what is there -- but say it. The count changing
        # means cap-x moved underneath us and the next reader should know.
        print(f"  NOTE: expected {EXPECTED_SITES} call sites, found {len(hits)}")

    text, n = PATTERN.subn(f"timeout={timeout}", text)
    if MARKER not in text:
        banner = (
            f"\n{MARKER} from {STOCK_TIMEOUT}s to {timeout}s by "
            f"scripts/patch_llm_timeout.py.\n"
            f"# A read timeout fires when the ENGINE is slow, not when the "
            f"model is wrong: a\n"
            f"# draw that runs to the token cap on every request queues past "
            f"{STOCK_TIMEOUT}s behind\n"
            f"# another and its cell is refused as a throughput failure. Every "
            f"other cell in\n"
            f"# the corpus answers well inside {STOCK_TIMEOUT}s, so this "
            f"threshold never fires for\n"
            f"# them and raising it changes nothing they do. See STATUS "
            f"2026-09-28.\n"
        )
        # Put the banner after the module docstring / first import block so it
        # reads as provenance rather than as a stray comment mid-function.
        lines = text.splitlines(keepends=True)
        idx = 0
        for i, ln in enumerate(lines[:60]):
            if ln.startswith(("import ", "from ")):
                idx = i + 1
        text = "".join(lines[:idx]) + banner + "".join(lines[idx:])
    path.write_text(text)
    return f"patched {n} sites -> {timeout}s", n


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else os.environ.get("CAPX_DIR", "/data/cap-x"))
    timeout = DEFAULT_TIMEOUT
    print(f"[timeout] cap-x at {root}")
    # Record the value in the log. A patch whose effect is a number nobody
    # wrote down is indistinguishable from no patch at all when the run is
    # read back months later.
    print(f"[timeout] stock={STOCK_TIMEOUT}s  new={timeout}s  "
          f"(override with OPENCAP_LLM_READ_TIMEOUT)")

    p = root / TARGET
    if not p.exists():
        print(f"  MISSING {TARGET}")
        print("patch incomplete")
        return 1

    status, _ = patch_file(p, timeout)
    print(f"  {status:<40} {TARGET}")
    if "NOT FOUND" in status:
        print("patch incomplete")
        return 1
    print(f"llm read timeout patch applied ({timeout}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
