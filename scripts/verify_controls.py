"""Assert that CaP-X's own constants match the controls we claim to run under.

Every check here corresponds to a control in roadmap Sec. 05. They are asserted
against a real checkout rather than trusted, because each one is a place where
CaP-X's default would silently produce a different experiment than the one we
describe in the paper.

Usage:  python3 scripts/verify_controls.py /content/cap-x
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opencap.config import MAX_TURNS

# Long jobs are launched detached with stdout redirected to a file, where
# Python block-buffers by default -- the log stays empty until exit, so a
# multi-hour run shows no progress and a crash shows no context. Line
# buffering costs nothing and makes the log readable while it runs.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass


failures: list[str] = []
notes: list[str] = []


def check(ok: bool, msg: str) -> None:
    (notes if ok else failures).append(msg)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/cap-x")

    trial = (root / "capx/envs/trial.py").read_text()

    # --- turn cap ----------------------------------------------------------
    m = re.search(r"^MULTITURN_LIMIT\s*=\s*(\d+)", trial, re.M)
    check(m is not None, "trial.py defines MULTITURN_LIMIT")
    if m:
        limit = int(m.group(1))
        check(limit == MAX_TURNS,
              f"MULTITURN_LIMIT == {MAX_TURNS} (found {limit})")

    # The loop is `code_block_idx <= MULTITURN_LIMIT`, so the cap admits
    # indices 0..LIMIT inclusive -- i.e. up to LIMIT+1 executed code blocks.
    # Record the real bound so the paper reports it correctly.
    loop = re.search(r"while code_block_idx < len\(code_blocks\) and "
                     r"code_block_idx (<=?) MULTITURN_LIMIT", trial)
    check(loop is not None, "multi-turn loop found")
    if loop:
        inclusive = loop.group(1) == "<="
        notes.append(f"effective max code blocks = "
                     f"{MAX_TURNS + 1 if inclusive else MAX_TURNS} "
                     f"(loop uses '{loop.group(1)}')")

    # --- trial retry semantics --------------------------------------------
    runner = (root / "capx/envs/runner.py").read_text()
    r = re.search(r"^MAX_TRIAL_RETRIES\s*=\s*(\d+)", runner, re.M)
    t = re.search(r"^TRIAL_TIMEOUT_SECONDS\s*=\s*(\d+)", runner, re.M)
    check(r is not None and t is not None,
          f"retry constants found (retries={r.group(1) if r else '?'}, "
          f"timeout={t.group(1) if t else '?'}s)")
    # Retries must fire ONLY on timeout. Retrying on task failure would be a
    # free-retry scaffold that inflates success for whichever models fail more.
    retry_on_timeout_only = ("except TimeoutError:" in runner
                            and "timed out (attempt" in runner)
    check(retry_on_timeout_only,
          "retries fire on TimeoutError only, not on task failure")

    # --- we must be on the CLI path, not the web UI ------------------------
    web = root / "capx/web/async_trial_runner.py"
    if web.exists():
        wm = re.search(r"^MULTITURN_LIMIT\s*=\s*(\d+)", web.read_text(), re.M)
        if wm and int(wm.group(1)) != MAX_TURNS:
            notes.append(f"web UI path uses MULTITURN_LIMIT={wm.group(1)} -- "
                         f"never run the grid through --web-ui")

    # --- launch defaults we override ---------------------------------------
    launch = (root / "capx/envs/launch.py").read_text()
    for field, default, why in [
        ("temperature", "1.0", "we need greedy decoding"),
        ("max_tokens", "2048 * 10", "we need 2048"),
    ]:
        d = re.search(rf"^\s+{field}: \w+ = (.+)$", launch, re.M)
        if d and d.group(1).strip() == default:
            notes.append(f"launch.py {field} default is {default} -- {why}; "
                         f"capx_bridge overrides it")
        else:
            failures.append(f"launch.py {field} default changed "
                            f"(expected {default}, got {d.group(1) if d else '?'}) "
                            f"-- re-check capx_bridge overrides")

    for n in notes:
        print("  ok  ", n)
    for f in failures:
        print("  FAIL", f)
    print(f"\n{len(notes)} ok, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
