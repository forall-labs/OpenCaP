"""Check opencap.editclass.PROVIDED against the names cap-x actually injects.

    python3 scripts/verify_provided_api.py [CAPX_DIR] [--strict]

WHY THIS EXISTS
---------------
PROVIDED was a hand-written list of eight names. `goto_home_joint_position` was
not on it, so every program that called it was tagged `unbound_name` -- a
defect in the INTERFACE family -- when in fact the name is bound at runtime and
the human reference calls it. Nine programs were mis-tagged, all of them large
models, and the resulting "INTERFACE rises with scale, 69% at >=12B" was the
seventh headline finding in this project to invert.

Nothing in the pipeline could have caught that: the tagger takes a program and
a repair, and neither of them knows what the harness binds. This script closes
the loop by reading the binding site itself.

WHERE THE NAMES COME FROM
-------------------------
The execution namespace is built in `_init_exec_globals` -- the same method
scripts/patch_exec_globals.py anchors its numpy injection on. We parse it and
collect every string key assigned into the exec-globals dict. That is a
structural read, not a grep for names we already expect to find, so a primitive
nobody here has heard of still shows up.

Advisory by default (it prints the comparison); --strict exits non-zero on any
name the harness binds and PROVIDED omits, which is the direction that
manufactures a false defect. A name in PROVIDED that the harness does NOT bind
is reported too, but does not fail: the tagger uses PROVIDED for rebinding
checks as well, and over-listing there is harmless.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from opencap.editclass import PREIMPORTED, PROVIDED  # noqa: E402

ANCHOR = "_init_exec_globals"


def injected_names(root: Path) -> tuple[set[str], list[Path]]:
    """Every string key written into the exec-globals dict, and where."""
    found: set[str] = set()
    where: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if ANCHOR not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == ANCHOR):
                continue
            where.append(path)
            for sub in ast.walk(node):
                # self._exec_globals["name"] = ...
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) \
                        and isinstance(sub.slice.value, str):
                    found.add(sub.slice.value)
                # self._exec_globals.update({"name": ...}) / a dict literal
                elif isinstance(sub, ast.Dict):
                    for k in sub.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            found.add(k.value)
    return found, where


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    root = Path(args[0] if args else "/data/cap-x")
    if not root.exists():
        print(f"[provided] no checkout at {root} -- skipping (pass CAPX_DIR)")
        return 0

    found, where = injected_names(root)
    if not where:
        print(f"[provided] no `{ANCHOR}` found under {root}.")
        print("[provided] The binding site moved. Do NOT assume PROVIDED is")
        print("[provided] still right -- find it and repoint this script.")
        return 1 if strict else 0

    for p in where:
        print(f"[provided] binding site: {p.relative_to(root)}")
    # numpy is injected conditionally by our own patch, and is tracked apart.
    found -= set(PREIMPORTED)
    missing = found - set(PROVIDED)
    extra = set(PROVIDED) - found

    print(f"[provided] harness binds {len(found)}: {', '.join(sorted(found))}")
    print(f"[provided] PROVIDED lists {len(PROVIDED)}: {', '.join(sorted(PROVIDED))}")
    if missing:
        print(f"[provided] MISSING from PROVIDED: {', '.join(sorted(missing))}")
        print("[provided] Every program calling one of these is being tagged")
        print("[provided] unbound_name for using the API as intended.")
    if extra:
        print(f"[provided] listed but not bound here: {', '.join(sorted(extra))}")
    if not missing and not extra:
        print("[provided] OK -- PROVIDED matches the harness exactly.")
    return 1 if (missing and strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
